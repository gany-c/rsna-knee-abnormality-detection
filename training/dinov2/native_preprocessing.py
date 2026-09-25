from pathlib import Path
import hashlib, json, time, os, shutil, itertools
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
import pydicom
from pydicom.pixels import apply_modality_lut
ID='StudyInstanceUID'
SID='SeriesInstanceUID'
AXES = {
    'Sagittal': np.array([[1,0,0], [0,0,1], [0,-1,0]], float),
    'Coronal': np.array([[0,0,1], [1,0,0], [0,-1,0]], float),
    'Axial': np.array([[0,0,1], [0,1,0], [1,0,0]], float),
}
def validate_and_extract_geometry(headers, expected_study, expected_series, plane, cfg):
    """
    Validates DICOM header geometry, checks for spatial consistency, checks 
    slice-spacing consistency within configured tolerances, and extracts the physical coordinate basis matrix.
    
    Parameters:
        headers (list): List of pydicom dataset headers for a single series.
        expected_study (str): Expected StudyInstanceUID for validation.
        expected_series (str): Expected SeriesInstanceUID for validation.
        plane (str): Expected anatomical plane ('Sagittal', 'Coronal', 'Axial').
        cfg (dict): Pipeline configuration dictionary containing tolerances.
        
    Returns:
        dict: Spatial metadata including slice order, origin, basis matrix, and warnings.
    """
    if plane not in AXES:
        raise ValueError(f"Unknown plane: {plane}")
    if len(headers) < 2:
        raise ValueError("Need at least two spatial slices")

    first = headers[0]
    
    # iop is an array of 6 floating-point numbers (standard medical imaging metadata):
    # - The first 3 values (iop[:3]): Direction cosines for the image row.
    # - The last 3 values (iop[3:]): Direction cosines for the image column.
    iop = np.asarray(first.ImageOrientationPatient, float)
    
    spacing = np.asarray(first.PixelSpacing, float)

    # 1. Validate initial orientation and spacing parameters
    if iop.shape != (6,) or not np.isfinite(iop).all():
        raise ValueError("Invalid orientation")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Invalid pixel spacing")

    col, row = iop[:3], iop[3:]
    if (
        abs(np.linalg.norm(col) - 1) > 1e-3
        or abs(np.linalg.norm(row) - 1) > 1e-3
        or abs(col @ row) > 1e-3
    ):
        raise ValueError("Orientation vectors are not orthonormal")

    normal = np.cross(col, row)
    photo = str(first.PhotometricInterpretation)
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Only monochrome MRI supported")

    shape = (int(first.Rows), int(first.Columns))
    positions = []
    sops = []

    # 2. Iterate and validate consistency across all slices in the series
    for h in headers:
        if str(h.StudyInstanceUID) != expected_study or str(h.SeriesInstanceUID) != expected_series:
            raise ValueError("DICOM study/series ID mismatch")
        if int(getattr(h, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multiframe DICOM unsupported")
        if int(getattr(h, "SamplesPerPixel", 1)) != 1:
            raise ValueError("Non-grayscale DICOM")
        if (int(h.Rows), int(h.Columns)) != shape:
            raise ValueError("Mixed dimensions within series")
        if not np.allclose(h.ImageOrientationPatient, iop, atol=1e-4, rtol=0):
            raise ValueError("Mixed orientations within series")
        if not np.allclose(h.PixelSpacing, spacing, atol=1e-4, rtol=0):
            raise ValueError("Mixed pixel spacing within series")
        if str(h.PhotometricInterpretation) != photo:
            raise ValueError("Mixed photometric interpretation")
        
        positions.append(np.asarray(h.ImagePositionPatient, float))
        sops.append(str(h.SOPInstanceUID))

    if len(set(sops)) != len(sops):
        raise ValueError("Duplicate SOP instances")

    positions = np.asarray(positions)
    if positions.shape != (len(headers), 3) or not np.isfinite(positions).all():
        raise ValueError("Missing/nonfinite slice position")

    # 3. Sort slice positions and evaluate spatial gaps
    order = np.argsort(positions @ normal, kind="stable")
    positions = positions[order]
    distances = positions @ normal
    gaps = np.diff(distances)
    dz = float(np.median(gaps))

    if dz <= 0 or min(gaps) < 1e-3:
        raise ValueError("Duplicate/non-spatial slice positions")
    if np.max(abs(gaps - dz)) > max(0.05, cfg["gap_tolerance"] * dz):
        raise ValueError("Irregular slice spacing / missing slices; not silently interpolated")

    residual = positions - (positions[0] + np.arange(len(headers))[:, None] * dz * normal)
    if np.linalg.norm(residual, axis=1).max() > 0.2:
        raise ValueError("Nonparallel / shifted slice stack")

    # 4. Compute anatomical obliquity and flag warnings if needed
    angle = float(np.degrees(np.arccos(np.clip(abs(normal @ AXES[plane][:, 0]), 0, 1))))
    nearest = min(AXES, key=lambda p: np.degrees(np.arccos(np.clip(abs(normal @ AXES[p][:, 0]), 0, 1))))
    
    review_warning = ""
    if angle > cfg["max_obliquity_degrees"]:
        review_warning = (
            f"Orientation review: {angle:.1f} degrees from listed {plane}; "
            f"warning threshold {cfg['max_obliquity_degrees']:.1f}; nearest plane {nearest}. "
            "Native acquisition retained; plane label unchanged."
        )

    # 5. Construct the final 3D coordinate basis matrix
    basis = np.column_stack([normal * dz, row * spacing[0], col * spacing[1]])
    
    return dict(
        order=order, origin=positions[0], basis=basis,
        shape=(len(headers), *shape), photo=photo, angle=angle, dz=dz,
        review_warning=review_warning, nearest_plane=nearest,
        slice_thickness=float(getattr(first, "SliceThickness", 0) or 0)
    )


def native_grid(g, plane, cfg):
    """
    Performs spatial normalization on individual scan slices. It aligns, scales, 
    and computes the 2D affine transformation matrices needed to map heterogeneous 
    scanner layouts onto a uniform, standardized output grid without arbitrary 3D rotations.
    
    Parameters:
        g (dict): Spatial metadata dictionary extracted from geometry validation.
        plane (str): Expected anatomical plane ('Sagittal', 'Coronal', 'Axial').
        cfg (dict): Pipeline configuration dictionary containing target size and spacing.
        
    Returns:
        tuple: (matrix, offset, basis, crop) used for downstream image resampling.
    """
    # 1. Select native in-plane axes without through-plane interpolation or arbitrary rotations
    native = g['basis'][:, 1:]
    unit = native / np.linalg.norm(native, axis=0)
    
    # 2. Test valid axis permutations and sign flips to match the target anatomical orientation
    choices = []
    for perm in [(0, 1), (1, 0)]:
        for signs in itertools.product([-1, 1], repeat=2):
            directions = unit[:, perm] * np.array(signs)
            score = float(np.sum(directions * AXES[plane][:, 1:]))
            choices.append((score, directions))
            
    directions = max(choices, key=lambda x: x[0])[1]
    shape = np.array(g['shape'][1:])
    
    # 3. Calculate physical dimensions (extent) of the scan field of view (FOV)
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in shape])))
    projections = (corners @ native.T) @ directions
    extent = np.ptp(projections, axis=0)
    
    # 4. Determine pixel spacing, optionally scaling to fit the full FOV while preserving aspect ratio
    spacing = float(cfg['spacing_mm'])
    if cfg.get('fit_full_fov', False):
        spacing = max(spacing, float(extent.max()) / (cfg['size'] - 1))
        
    # 5. Compute the 2D affine transformation matrix:
    #    The resampling matrix maps output pixel coordinates back to native source coordinates 
    #    using scaling, axis swaps and flips; offset supplies the translation. No arbitrary rotation is used.
    basis = directions * spacing
    matrix = np.linalg.pinv(native) @ basis
    matrix[np.abs(matrix) < 1e-10] = 0
    matrix = np.where(np.abs(matrix - np.rint(matrix)) < 1e-10, np.rint(matrix), matrix)
    
    # 6. Calculate centering offsets (translation part of the spatial alignment)
    center = (shape - 1) / 2
    offset = center - matrix @ np.full(2, (cfg['size'] - 1) / 2)
    offset[np.abs(offset) < 1e-10] = 0
    offset = np.where(np.abs(offset - np.rint(offset)) < 1e-10, np.rint(offset), offset)
    
    # 7. Validate safety constraints to ensure excessive cropping does not occur
    crop = np.maximum(0, 1 - (cfg['size'] - 1) * spacing / np.maximum(extent, 1e-9))
    if crop.max() > cfg['max_crop_fraction'] + 1e-8:
        raise ValueError(f'Native FOV crop would be {crop.max():.1%}; increase size/spacing')
        
    return matrix, offset, basis, crop


def decode_native_slice(path):
    """
    Loads an individual DICOM slice file, masks out background padding values, 
    applies the DICOM modality LUT or rescale parameters, and returns the processed float32 
    image alongside a validity mask.
    
    Parameters:
        path (str or Path): File path to the DICOM slice.
        
    Returns:
        tuple: (image, valid_mask)
            - image (np.ndarray): 32-bit floating-point array after modality scaling; MRI values need not have standardized physical units.
            - valid_mask (np.ndarray): Boolean mask filtering out padding and non-finite values.
    """
    # 1. Load the DICOM file and validate that the pixel array is strictly 2D
    ds = pydicom.dcmread(path)
    raw = ds.pixel_array
    if raw.ndim != 2:
        raise ValueError("Expected single-frame 2D pixels")
        
    # 2. Mask explicit DICOM padding values; this does not identify all air or anatomical background
    valid = np.ones(raw.shape, bool)
    if hasattr(ds, "PixelPaddingValue"):
        a = float(ds.PixelPaddingValue)
        b = float(getattr(ds, "PixelPaddingRangeLimit", a))
        valid &= (raw < min(a, b)) | (raw > max(a, b))
        
    # 3. Apply the modality LUT or rescale slope/intercept when supplied by the DICOM metadata
    image = apply_modality_lut(raw, ds).astype(np.float32)
    
    # 4. Return the calibrated image and combine the padding mask with a finite-value check
    return image, valid & np.isfinite(image)


def series_intensity_limits(paths, cfg):
    """
    Calculates robust global pixel intensity clipping bounds across an entire series 
    of DICOM slices using a memory-efficient, deterministic sampling strategy.
    
    Parameters:
        paths (list): List of file paths to the DICOM slices in the series.
        cfg (dict): Pipeline configuration containing sampling targets and percentile definitions.
        
    Returns:
        tuple: (low, high) floating-point intensity limits for outlier clipping.
    """
    # 1. Determine a memory-efficient pixel sample budget per slice to prevent RAM spikes
    per_slice = max(1, cfg['intensity_sample_pixels'] // len(paths))
    samples = []
    
    # 2. Iterate through each slice, decode, and extract evenly-spaced valid pixel samples
    for path in paths:
        image, valid = decode_native_slice(path)
        values = image[valid]
        if len(values):
            # Select an evenly distributed subset of pixels using linear spacing
            ix = np.linspace(0, len(values) - 1, min(per_slice, len(values))).astype(int)
            samples.append(values[ix])
            
    # 3. Validate that a sufficient number of valid pixels were collected across the series
    if not samples or sum(map(len, samples)) < 32:
        raise ValueError('Insufficient valid pixels')
        
    # 4. Compute lower and upper percentiles (configured here as 0.5th and 99.5th) to establish robust intensity bounds
    low, high = np.percentile(np.concatenate(samples), cfg['percentiles'])
    
    # 5. Verify that the calculated limits are finite numbers and form a non-degenerate range
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Constant/invalid series intensity')
        
    return float(low), float(high)


def normalize_native_slice(image, valid, photo, low, high, matrix, offset, cfg):
    """
    Normalizes pixel intensities, handles photometric inversion, applies 
    conditional anti-aliasing for downsampling, and resamples the 2D slice 
    onto the standardized output grid using an affine transformation.
    
    Parameters:
        image (np.ndarray): 32-bit floating-point raw slice image.
        valid (np.ndarray): Boolean validity mask for the image pixels.
        photo (str): Photometric interpretation (e.g., 'MONOCHROME1' or 'MONOCHROME2').
        low (float): Lower intensity clipping threshold.
        high (float): Upper intensity clipping threshold.
        matrix (np.ndarray): 2D affine transformation matrix.
        offset (np.ndarray): Affine translation offset.
        cfg (dict): Pipeline configuration containing target size, antialiasing flags, etc.
        
    Returns:
        tuple: (output_slice, support_mask)
            - output_slice (np.ndarray): 16-bit floating-point normalized and resampled image.
            - support_mask (np.ndarray): Boolean support mask indicating valid resampled pixels.
    """
    # 1. Scale pixel values into a [0, 1] range using global intensity limits and clip outliers
    image = np.clip((image - low) / (high - low), 0, 1).astype(np.float32)
    
    # 2. Invert pixel intensities if photometric interpretation is MONOCHROME1 (lower values are intended to display brighter)
    if photo == 'MONOCHROME1':
        image = 1 - image
        
    # 3. Force invalid or background padding pixels to zero
    image[~valid] = 0
    
    # 4. Apply conditional anti-aliasing via Gaussian smoothing for in-plane reductions (downsampling)
    #    Anti-alias only in-plane reductions; no filtering is ever applied along depth.
    sigma = .5 * np.sqrt(np.maximum(np.sum(matrix ** 2, axis=1) - 1, 0))
    if cfg['antialias'] and sigma.max() > 0:
        numerator = gaussian_filter(image, sigma, mode='constant', cval=0)
        denominator = gaussian_filter(valid.astype(np.float32), sigma, mode='constant', cval=0)
        image = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)
        
    # 5. Resample the image and validity mask onto the standardized target grid using affine transformations
    shape = (cfg['size'], cfg['size'])
    output = affine_transform(image, matrix, offset, output_shape=shape, order=1, mode='constant', cval=0, prefilter=False)
    support = affine_transform(valid.astype(np.float32), matrix, offset, output_shape=shape, order=1, mode='constant', cval=0, prefilter=False) > .999
    
    # 6. Zero out unsupported regions, ensure all values are finite, clip, and cast to float16 for memory efficiency
    output[~support] = 0
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite output')
        
    return np.clip(output, 0, 1).astype(np.float16), support


def select_native_indices(n, depth):
    """
    Selects a uniform subset of slice indices from a 3D scan volume to match 
    a target depth, ensuring both ends of the volume are always included.
    
    Parameters:
        n (int): Total number of available native slices in the series.
        depth (int): Target number of slices required by the pipeline configuration.
        
    Returns:
        np.ndarray: 1D array of unique integer slice indices.
    """
    # 1. If the total available slices are fewer than or equal to the target depth, return all indices
    if n <= depth:
        return np.arange(n, dtype=int)
        
    # 2. Otherwise, use linear spacing to select evenly distributed actual slices across the volume
    #    (guaranteeing inclusion of both endpoints without artificial duplication or interpolation)
    indices = np.rint(np.linspace(0, n - 1, depth)).astype(int)
    
    # 3. Assert that all selected indices are unique to prevent overlapping or repeated slices
    assert len(np.unique(indices)) == depth
    
    return indices


def process_series(folder, study, series, plane, cfg):
    """
    Orchestrates the end-to-end preprocessing pipeline for a 3D DICOM series folder, 
    handling file discovery, fast header parsing, spatial validation, orientation 
    normalization, and cache array generation.
    
    Parameters:
        folder (str or Path): Directory path containing the DICOM series files.
        study (str): Study identifier.
        series (str): Series identifier.
        plane (str): Target anatomical viewing plane.
        cfg (dict): Pipeline configuration dictionary.
        
    Returns:
        tuple: (normalized image array, valid-pixel mask, metadata dictionary).
    """
    # 1. Discover and sort all DICOM files in the directory; raise an error if none are found
    paths = sorted(Path(folder).glob('*.dcm'))
    if not paths:
        raise FileNotFoundError(f'No DICOM slices in {folder}')
        
    # 2. Efficiently parse DICOM headers across all files without loading heavy pixel data into RAM
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    
    # 3. Extract and validate spatial geometry and orientation from the headers
    g = validate_and_extract_geometry(headers, study, series, plane, cfg)
    
    # 4. Reverse slice order if needed so the native normal has nonnegative projection on the listed plane normal
    if g['basis'][:, 0] @ AXES[plane][:, 0] < 0:
        g['origin'] = g['origin'] + g['basis'][:, 0] * (len(paths) - 1)
        g['basis'][:, 0] *= -1
        g['order'] = g['order'][::-1]
        
    # 5. Compute transformation matrices, offsets, and cropping parameters for the native grid
    matrix, offset, basis, crop = native_grid(g, plane, cfg)
    
    # 6. Reorder file paths based on the validated spatial sequence and select target depth slice indices
    ordered = [paths[i] for i in g['order']]
    indices = select_native_indices(len(paths), cfg['depth'])
    
    # 7. Execute the final slice-by-slice processing and build the native cache arrays
    return build_native_cache_arrays(ordered, headers, g, indices, matrix, offset, basis, crop, cfg)


def build_native_cache_arrays(paths, headers, g, indices, matrix, offset, basis, crop, cfg):
    """
    Executes the streaming core processing loop with an estimated-memory guard for selected slices, normalizes 
    and resamples each slice onto the target grid, validates valid-pixel coverage, 
    and compiles a comprehensive metadata audit log.
    
    Parameters:
        paths (list): Sorted list of file paths to DICOM slices.
        headers (list): List of parsed DICOM dataset headers.
        g (dict): Validated spatial geometry dictionary.
        indices (np.ndarray): Selected target slice indices.
        matrix (np.ndarray): 2D affine transformation matrix.
        offset (np.ndarray): Affine translation offset.
        basis (np.ndarray): Spatial basis vectors.
        crop (np.ndarray): Fraction of source extent cropped along each output in-plane axis.
        cfg (dict): Pipeline configuration dictionary.
        
    Returns:
        tuple: (output_tensor, support_mask, audit_metadata)
    """
    # 1. Enforce a defensive memory guardrail by estimating working memory consumption
    estimate = int(np.prod(g['shape'][1:])) * 64 + cfg['depth'] * cfg['size'] ** 2 * 6 + cfg['intensity_sample_pixels'] * 16
    if estimate > cfg['max_working_memory_gb'] * 1e9:
        raise MemoryError(f'Estimated streaming working memory {estimate / 1e9:.2f} GB exceeds configured guard')
        
    # 2. Compute global intensity clipping bounds and pre-allocate target 3D tensors
    low, high = series_intensity_limits(paths, cfg)
    shape = (cfg['depth'], cfg['size'], cfg['size'])
    output = np.zeros(shape, np.float16)
    support = np.zeros(shape, bool)
    
    # 3. Initialize tracking collections and normal vectors for spatial provenance
    origins = []
    positions = []
    sops = []
    normal = g['basis'][:, 0] / np.linalg.norm(g['basis'][:, 0])
    
    # 4. Iteratively decode, normalize, resample, and validate each selected native slice
    for slot, index in enumerate(indices):
        image, valid = decode_native_slice(paths[index])
        output[slot], support[slot] = normalize_native_slice(image, valid, g['photo'], low, high, matrix, offset, cfg)
        
        # Verify that the slice meets the minimum required valid-pixel coverage fraction (not anatomical segmentation)
        if support[slot].mean() < cfg['min_valid_fraction']:
            raise ValueError('Too little support in a selected native slice')
            
        # Compute physical LPS origin and relative position along the normal axis
        origin = g['origin'] + g['basis'][:, 0] * int(index) + g['basis'][:, 1:] @ offset
        origins.append(origin.tolist())
        positions.append(float((origin - g['origin']) @ normal))
        sops.append(str(headers[g['order'][index]].SOPInstanceUID))
        
    # 5. Compute slice spacing gaps and assemble the comprehensive audit metadata dictionary
    gaps = np.diff(positions)
    meta = dict(
        shape=list(shape),
        input_shape=list(g['shape']),
        representation='native-plane-full-fov-v3',
        intensity_low=low,
        intensity_high=high,
        intensity_limits_method='bounded deterministic all-slice sample',
        source_sop_order=[str(headers[i].SOPInstanceUID) for i in g['order']],
        selected_sop_uids=sops,
        selected_source_indices=indices.tolist(),
        selected_count=len(indices),
        padded_count=cfg['depth'] - len(indices),
        retained_slice_fraction=len(indices) / len(paths),
        slice_positions_mm=positions,
        slice_gap_min_mm=float(gaps.min()),
        slice_gap_max_mm=float(gaps.max()),
        spacing_summary_drc_mm=[float(np.median(gaps)), *np.linalg.norm(basis, axis=0).tolist()],
        requested_inplane_spacing_mm=float(cfg['spacing_mm']),
        fov_spacing_adjusted=bool(np.linalg.norm(basis, axis=0).max() > cfg['spacing_mm'] + 1e-8),
        output_slice_origins_lps=origins,
        output_inplane_basis_lps=basis.tolist(),
        input_origin_lps=g['origin'].tolist(),
        input_basis_lps=g['basis'].tolist(),
        source_slice_spacing_mm=g['dz'],
        source_slice_thickness_mm=g['slice_thickness'],
        obliquity_degrees=g['angle'],
        review_warning=g['review_warning'],
        nearest_plane=g['nearest_plane'],
        crop_fraction_rc=crop.tolist(),
        estimated_working_memory_gb=estimate / 1e9,
        valid_fraction=float(support[:len(indices)].mean()),
        tensor_valid_fraction=float(support.mean()),
        transfer_syntaxes=sorted({str(h.file_meta.TransferSyntaxUID) for h in headers})
    )
    
    return output, support, meta
