# Foundational Concepts for the MRI Preprocessing Notebook

Personal learning checklist, organized from the original concept list. The descriptions below preserve the original wording; they have not been revised to incorporate the subsequent accuracy review.

## 1. NumPy & Linear Algebra Concepts

- **The Matrix Multiplication Operator (@):** Replaces verbose function calls like np.matmul() with clean, readable mathematical notation.
- **Array Broadcasting:** Automatically stretches and aligns arrays of different shapes without explicit loops (e.g., using [:, None] to turn 1D arrays into column vectors).
- **Indirect Sorting (np.argsort):** Returns index permutations rather than sorting arrays directly, allowing parallel arrays to be reordered identically.
- **Floating-Point Tolerances (np.allclose):** Safeguards equality checks against microscopic decimal rounding errors using absolute tolerances (atol).
- **Moore-Penrose Pseudoinverse (np.linalg.pinv):** Solves least-squares linear transformation systems robustly for ill-conditioned or non-square matrices.
- **Peak-to-Peak Range (np.ptp):** A fast, vectorized NumPy routine (maximum - minimum) used to calculate physical FOV boundaries.
- **Numerical Noise Clean-up (np.rint & Masking):** Flattens microscopic floating-point noise into exact zeros and integers using conditional masking and rounding.
- **Evenly Spaced Sequences (np.linspace):** Generates representative, evenly spaced coordinate subsets for efficient data sampling.
- **Endpoint-Inclusive Subsampling:** Subsampling arrays while guaranteeing the absolute inclusion of both start and end boundaries.
- **Percentile-Based Statistics (np.percentile):** Computes robust distribution boundaries to isolate valid signal ranges from extreme outlier artifacts.
- **Range Bounding & Clipping (np.clip):** Forcibly constraining array values within a strict numerical range (like $[0, 1]$) to eliminate out-of-bounds anomalies.
- **Half-Precision Memory Optimization (np.float16):** Casting final processed image arrays into 16-bit floating point to cut tensor memory consumption in half before feeding data into deep learning frameworks.
- **Tensor Pre-Allocation:** Instantiating pre-sized target 3D arrays with efficient types (np.float16, bool) prior to iterative slice population.
- **Vectorized Boolean Accumulation (|=):** Iteratively combining multi-column criteria into unified boolean arrays using bitwise OR operators and NumPy conversions.
- **Vectorized Conditional Mapping (np.where):** Applying nested conditional statements to efficiently generate descriptive categorical state columns.

## 2. SciPy & Image Processing Concepts

- **Anti-Aliasing via Gaussian Smoothing (gaussian_filter):** Applying a low-pass Gaussian filter proportional to downsampling scale factors before resampling to prevent high-frequency artifacts (jaggies).
- **Mask-Normalized Filtering:** Dividing a smoothed image signal by a smoothed validity mask (np.divide with explicit masks and denominators) to prevent edge distortion and incorrect weighting near boundary regions.
- **Affine Image Resampling (affine_transform):** Using geometric transformation matrices and interpolation (e.g., bilinear/order=1) to map pixel arrays onto standardized target grids.
- **Support Mask Thresholding:** Binarizing interpolated floating-point masks (e.g., > 0.999) to create strict, reliable regions of anatomical validity after resampling.
- <span style="color: orange;"><strong>Principal Axis Orientation Normalization:</strong> Using vector dot products against canonical anatomical reference axes to enforce a consistent spatial depth direction.</span>

## 3. Python Standard Library & File System Concepts

- **pathlib Object-Oriented Paths:** Leverages clean, chainable path manipulation methods instead of raw string concatenation.
- **The Atomic File Swap Pattern:** Ensures file writing safety by saving data to a temporary .tmp file first and instantly swapping it into place.
- **Cartesian Product Generation (itertools.product):** Exhaustively loops through multi-dimensional combinations (like axis permutations and sign flips) without nested loops.
- **Zero-Copy File Hard-Linking (os.link):** Staging large files instantly via directory hard links to save disk space and I/O overhead.
- **Filesystem Device Boundary Detection (st_dev):** Comparing device identifiers to safely handle cross-device constraints before hard-linking or copying.
- **Cryptographic File Integrity Hashing (hashlib.sha256):** Computing secure SHA-256 digests of key files to guarantee absolute version provenance.
- **Portable Path Normalization (as_posix()):** Translating absolute paths into relative, OS-agnostic POSIX path strings inside exported manifests.

## 4. Pandas & Tabular Data Concepts

- **Optimized DataFrame Lookups (set_index & .at):** Setting unique column identifiers as DataFrame indices to perform lightning-fast scalar lookups during filtering.
- **Relational Table Integrity Checks:** Utilizing .is_unique and set operations (issubset) to validate primary keys and foreign key constraints across tables.
- **Cross-Table Relational Filtering (.isin):** Subsetting primary tables using filtered multi-criteria audit keys.
- **Sparse Summary Reindexing (reindex with fill_value):** Guaranteeing complete coverage across discrete categories (like shard ranges) by backfilling unrepresented groups.
- **Stacked DataFrame Validation (stack().dropna()):** Flattening multi-column structures for comprehensive multi-attribute validation.
- **Conditional Series Selection (.where()):** Conditionally keeping or masking column values based on external boolean criteria.
- **Index Intersection Filtering (.intersection()):** Efficiently computing common subsets between DataFrame indices and unique identifiers.

## 5. DICOM & Medical Imaging Concepts

- **DICOM Dataset Parsing (pydicom.dcmread & pixel_array):** Standard workflow for reading medical imaging metadata and raw pixel arrays.
- **Lazy Header-Only Parsing (stop_before_pixels=True):** Inspecting metadata headers across an entire series without loading heavy pixel arrays, optimizing initial scan discovery speed.
- **Background Pixel Padding Masking (PixelPaddingValue & PixelPaddingRangeLimit):** Standard tags defining background air or scanner borders to exclude non-tissue pixels.
- **Modality Look-Up Table Calibration (apply_modality_lut):** Standardized transformation using rescale slope/intercept metadata to convert raw integers into physical units.
- **Photometric Interpretation Handling (MONOCHROME1 vs MONOCHROME2):** Correcting luminance conventions where raw DICOM standards invert pixel intensities (e.g., white representing air instead of bone) to ensure consistent training signals.
- **DICOM Provenance & SOP Tracking:** Extracting unique instance identifiers (SOPInstanceUID) and transfer syntaxes for rigorous medical data traceability.

## 6. Architectural & Pipeline Design Patterns

- **Fail-Fast & Defensive Programming:** Aggressively raises explicit exceptions (ValueError, MemoryError) when irregular data or memory constraints are encountered.
- **Pre-Execution Memory Guardrails:** Estimating peak memory footprints beforehand to protect host environments from unhandled allocation spikes.
- **Uniqueness Assertion Guards:** Enforces structural integrity checks (such as np.unique assertions) to prevent silent data corruption during index mapping.
- **Spatial Normalization & Grid Standardization:** Abstracts scanner-specific variations into a uniform output coordinate space.
- **Defensive Bounding & Guardrails:** Enforces strict configuration limits (like max crop fractions, minimum sample counts, and support thresholds) to catch severe edge cases early.
- **RAM-Conscious Sampling & Streaming Pattern:** Trades minor I/O overhead to iteratively process slices and maintain a flat, highly predictable memory footprint.
- **High-Level Pipeline Orchestration Pattern:** Centralizing multi-step execution flows (discovery, validation, normalization, caching, and auditing) into unified driver functions.
- **Comprehensive Metadata Audit Logging:** Assembling deep dictionaries of spatial, intensity, and structural metrics to ensure complete reproducibility.
- **Stratified Pilot Subsampling:** Partitioning data into logical training and validation partitions with guaranteed proportional quotas for controlled dry-runs.
- **Supervised Data Curation Gatekeeping:** Enforcing rigorous multi-label qualification rules (gold standard verification vs. generated mask acceptance) coupled with explicit audit reason logging.
- **Hash-Based Deterministic Sharding:** Assigning datasets to partitions via stateless hash digests modulo $n$.
- **Iterative Capacity-Based Autoscaling:** Dynamically scaling partition counts in a feedback loop to respect storage constraints.
- **Patient-Level Group Stratification:** Grouping records by patient identifiers to prevent data leakage across cross-validation splits.
- **Deterministic Hash-Seeded Group Ordering:** Shuffling and ordering group keys using hash digests with a fixed seed for reproducible partitioning.

## 7. PyTorch & Training Data Concepts

- **PyTorch Custom Dataset Integration (torch.utils.data.Dataset):** Encapsulating data extraction and tensor formatting via __len__ and __getitem__ for DataLoader compatibility.
- **Multi-Series Tensor Stacking & Padding:** Structuring variable-length multi-series study records into uniform, zero-padded PyTorch tensors.
- **Categorical Protocol Encoding:** Mapping categorical string descriptors into integer feature vectors for neural network feature extraction.
