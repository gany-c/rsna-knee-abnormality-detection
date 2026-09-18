# RSNA Knee MRI: Preprocessing Algorithm

This guide describes [normalize-mri-for-training.ipynb](normalize-mri-for-training.ipynb), including its full-field-of-view resizing policy. The notebook prepares data for study-level abnormality classification; it does not train a model.

## 1. Constraints: the starting data

A **study** is one examination. It contains multiple **series**, or scan sequences, each containing individual image **slices** stored as DICOM files. DICOM files hold both image pixels and acquisition metadata.

The source data varies in:

- Number of series per study and slices per series.
- Image dimensions, physical pixel spacing, slice spacing and field of view.
- Scan plane, acquisition tilt, MRI contrast and intensity scale.
- Availability and reliability of labels.
- Geometry and decoding quality: some series have inconsistent metadata or unreadable pixel data.

The observed inventory contains **4,407 studies and 24,371 series**, including **58 studies with verified labels**. With the current accepted GPU labels, selection retains **3,919 studies and 21,749 series** before image checks. These counts describe the current inputs, not guaranteed final training coverage.

### Inputs

| Input | Purpose |
|---|---|
| `train_series/` | Original MRI DICOM images |
| `train_series.csv` | Study/series IDs, listed plane and scan-type flags |
| `train.csv` | Study IDs and verified abnormality labels |
| `qwen_training_labels.csv` | Accepted generated targets, masks, weights and label sources |
| Optional patient-group mapping | Keeps studies from a known patient together when assigning folds |

A study is not necessarily a unique patient. The current run uses study-level grouping; without a patient mapping, patient-level independence is not established.

## 2. Goal: the training-ready representation

Produce a predictable array shape while preserving the metadata needed to interpret it:

- **One `64 × 320 × 320` image array per successful series**, ordered as slices, rows and columns.
- Actual physical spacing, selected slice positions and source references.
- Masks identifying valid pixels, real slices and real series within a padded batch.
- Study-level targets, known-label masks, training weights and fold assignments.
- Training manifests containing only studies whose expected series all succeeded.
- Audit tables for errors, warnings and study-selection decisions.
- A cumulative private Kaggle dataset, split into manageable shard packages.

**Equal array dimensions do not imply equal physical spacing.** In-plane spacing is usually 0.75 mm, but larger scans use coarser spacing to fit. Slice spacing follows the source acquisition and selected slices. Padding is not an acquired physical location.

The result is not an isotropic 3D volume (equal spacing in all directions), anatomical segmentation (outlines of structures), or registration between studies (precise anatomical alignment across examinations).

## 3. Algorithm at a broad level

### Step 1: Prepare labels, assign folds and select studies

Read the metadata and label tables. Verified labels override generated labels where available. Assign folds before filtering to preserve the split across shards.

Include a study if it has either:

- At least one verified target; or
- At least one accepted binary target with an enabled mask and a positive training weight.

Unknown targets are ignored during training; they are not treated as negative diagnoses. All series of an included study are considered for processing.

Verified groups are distributed across five folds. Groups without verified labels normally receive fold `-1`; if patient grouping links them to a verified group, they share that group's fold. The training loader holds out the configured validation fold, and validation uses verified targets only.

### Step 2: Assign studies to shards

Assign each study deterministically using its ID, keeping all its series in the same shard. The automatic planner chooses a shard count using estimated cache sizes and a target of 4.7 GB per shard.

The current plan has **12 shards, numbered 0-11**, with **282-351 studies per shard**. Shard sizes are estimates, not exact balance or storage guarantees. Time, memory and free-disk guards limit each run.

### Step 3: Read and validate each series

Treat each supported series as a spatial stack: an ordered set of slices along the acquisition direction, which is not necessarily the global Z-axis.

Use DICOM physical positions to order slices rather than filenames. Check study/series IDs, orientation, dimensions, pixel spacing and slice positions. Decode grayscale pixels, apply modality scaling, and account for photometric interpretation and explicit pixel padding.

Checks reject unsupported or invalid data, including duplicate slice positions, mixed geometry, missing required metadata, irregular spacing beyond tolerance, shifted stacks and decoding failures. Small spacing differences within tolerance are allowed. The current reader does not support enhanced/multiframe DICOM.

**Orientation deviations above 35° are review warnings.** This angle compares the actual slice orientation with the standard plane named in the series table. The notebook retains the native acquisition and leaves the listed plane unchanged; the warning alone does not exclude a series.

### Step 4: Normalize spatial representation

Standardize the two in-plane axes using flips or axis swaps. Do not rotate the stack into another anatomical plane or interpolate new slices through its depth.

**Example: rotating a photo on an iPhone.** This is similar to turning a photo by 90 degrees or flipping it in the photo editor. The notebook can flip an MRI slice horizontally or vertically, or swap rows and columns; a swap combined with a flip can produce a 90-degree rotation. It uses DICOM orientation metadata to choose a consistent direction.

Turning a side-view photo on your phone does not turn it into a front-view photo. Likewise, this step does not convert a sagittal (side-view) scan into a coronal (front-view) scan. It changes how the existing slice is arranged, not the viewing angle through the knee.

“Do not interpolate through depth” means the notebook does not invent new slices between acquired slices. It can still interpolate pixel values when resizing **within** each slice.


Fit each image into a **320 × 320** grid:

1. Use **0.75 mm** spacing in both in-plane directions when the image fits.
2. For a larger field of view, increase both spacings equally just enough to fit the full source pixel-center extent.
3. Center the image on the output grid and mark outside-image padding as invalid.
4. Save the actual spacing and physical mapping.

This preserves aspect ratio and avoids centered cropping. Larger spacing reduces spatial resolution. In-plane anti-aliasing reduces aliasing during downsampling, but does not guarantee preservation of fine findings.

- **Downsampling:** Representing an image with fewer pixels, which reduces detail.
- **Aliasing:** False patterns or jagged edges that can appear when fine image details are represented with too few pixels.


### Step 5: Normalize intensity

Estimate the **0.5th and 99.5th intensity percentiles** from a bounded, deterministic sample of valid pixels across every source slice. Clip intensities to those limits and scale them to **0-1**, accounting for grayscale inversion when required.

Use one pair of intensity limits for the series, rather than normalizing each slice independently. This reduces numerical intensity differences between acquisitions; it does not eliminate all scanner effects, bias fields or differences between MRI scan types.

### Step 6: Standardize slice count and create masks

**In this step, padding means adding blank slices.** Each added slice contains zeros, not real or synthesized anatomy.

Keep up to **64 actual slices**. If a series is longer, select slices across its depth, including both ends. If it is shorter, append zero-filled placeholder slices.

| Source series | Output |
|---|---|
| 30 slices | 30 real slices + 34 blank slices |
| 50 slices | 50 real slices + 14 blank slices |
| 64 slices | 64 real slices |
| 100 slices | 64 selected real slices across the series |

Padding adds no anatomy and does not fill gaps between acquired slices. The cache stores a packed valid-pixel mask and metadata including the real slice count; the loader derives slice masks and adds series masks when batching studies with different series counts.

**The model must use the supplied masks.** Merely saving a mask does not automatically prevent padding from influencing model predictions. Image-support masks are also distinct from label masks, which identify known targets.

### Step 7: Save caches and identify complete studies

Write one compressed `.npz` cache per successful series. Record successes, errors, review warnings and pending work in the manifest.

Under the current complete-study policy, **a study enters the training-ready manifest only when every expected series succeeds**. A failed or pending series prevents that study from entering the training set. Warning-only series remain eligible.

Excluded studies remain visible in audit tables. Successfully processed series from those studies are still cached and published, but the training loader uses the filtered training manifest. This policy defines completeness; it does not prove that every accepted examination or label is clinically correct.

### Step 8: Publish and load for training

Package each shard and publish a new version of the same private dataset:

`gany24558/rsna-knee-normalized-training-data`

Each publication includes the current shard and previous compatible shards. The index checks preprocessing identity, labels and shard plan to avoid mixing incompatible data. The current migration also preserves the old shard 0 under a legacy filename outside the active training index.

A **processed** shard has no pending work but can still contain recorded errors and excluded studies. A **partial** shard must be resumed. Successful upload submission is separate from Kaggle finishing server-side processing.

The notebook's training loader reads NPZ files directly from the archives and requires all planned shards to be present and processed. It combines series by study and supplies images, masks, scan metadata, spacing, targets and weights. Loading training data does not require decoding the original DICOMs again.

## 4. Output structure and what is uploaded

### Published dataset after all shards finish

```text
rsna-knee-normalized-training-data/
├── README.md
├── dataset_index.json              # Active shards and compatibility information
├── shard-000.tar.bin
├── shard-001.tar.bin
├── ...
├── shard-011.tar.bin
└── legacy-...-shard-000.tar.bin     # Old backup; excluded from active training
```

Each `.tar.bin` is a standard tar archive. The binary suffix prevents automatic archive expansion during publication.

### Inside one shard archive

```text
shard-005.tar.bin
├── cache/
│   └── <run_id>/
│       ├── <cache-hash>.npz
│       └── ...
├── study_labels_and_folds.csv
├── training_series_shard_005.csv
├── study_preprocessing_selection.csv
├── series_manifest_shard_005.csv
├── orientation_and_error_review_shard_005.csv
├── preprocessing_config.json
├── supervised_shard_plan.json
├── study_shard_assignments.csv
├── supervised_shard_summary.csv
└── publication_manifest.json
```

### Image cache contents

A cache is a reusable saved result of preprocessing. It contains actual processed image pixels, not just filenames or thumbnails.

| NPZ field | Contents |
|---|---|
| `image` | Normalized `float16` pixels, shaped `64 × 320 × 320`, including padding |
| `valid_bits` | Bit-packed mask of valid pixels; unpacked by the loader |
| `meta` | JSON metadata: shape, real slice count, source IDs, positions, physical spacing, scan details, warnings and preprocessing identity |

These caches contain processed arrays rather than the original DICOM files. Compression is lossless for the saved arrays, but preprocessing itself can reduce detail through resizing, slice selection, clipping and conversion to `float16`.

### Training tables and quality outputs

| Output | Uploaded? | Purpose |
|---|---|---|
| Series image caches | Yes | Actual pixels, valid-data masks and metadata |
| Labels and folds | Yes | Study-level targets, known-target masks, weights and split assignments |
| Training-series table | Yes | Cache references for complete, usable studies |
| Selection and shard tables | Yes | Selection decisions and study-to-shard assignments |
| Series and review manifests | Yes | Success, error, warning and pending-work records |
| Configuration and publication metadata | Yes | Settings, compatibility checks and package inventory |
| Original/normalized preview images | **No, in the current packaging code** | Visual review in notebook outputs; normalized montages also saved under working `qc/` |
| Python training-loader code | **No** | Defined in the notebook; reuse its helper cells in the training notebook |

The labels table covers the full study set and is repeated in shard packages. It is not a list of every study ready for training; the filtered training-series tables establish that coverage.

## 5. Run and resume notes

- For the next shard, wait for dataset processing to finish, start a fresh session, change only `SHARD_INDEX`, and keep inputs and other settings unchanged.
- If a run is interrupted, completed caches can be reused only if their files survive and their preprocessing identity matches. Unsaved session files may be lost.
- Resume the same shard before advancing if any series remain pending.
- Review error and warning tables after every run. A notebook reaching its last cell does not by itself prove full training coverage.
- For inference on new scans, apply the same image transforms and masks. The current notebook selects labeled training studies; inference needs its own input selection and failure-handling workflow.

## 6. Limitations to retain in training design

- Physical spacing can vary between series even though array shapes match. Use saved spacing and positions if the model accounts for physical scale.
- Depth spacing in the batch metadata is a median summary; selected slice positions preserve more detail.
- Long-series sampling and in-plane downsampling can remove small findings.
- MRI contrast differences remain; scan metadata should not be discarded without evaluation.
- Generated labels remain weak supervision, and validation uses the limited verified-label set.
- Study-disjoint folds alone do not establish patient-disjoint evaluation.

Implementation reference: [preprocessing notebook](normalize-mri-for-training.ipynb). Counts and defaults above reflect the current supervised, full-field-of-view configuration.
