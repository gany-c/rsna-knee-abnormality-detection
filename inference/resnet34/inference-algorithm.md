# ResNet34 scoring algorithm and training/scoring differences

## Model and data sources

Use only the attached `gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil` package and the supplied test files. Package family must be resnet34-study-mean-max-v1. Verify every manifest hash, the target list, input contract and checkpoint architecture before importing the saved modules. Instantiate ResNetKnee with pretrained=False and load model.pt strictly. Test predictions against synthetic_smoke.pt on CPU; compare streaming and original model forward on multiple synthetic series on the inference device.

No fitting, label extraction, score-based adaptation or calibration uses hidden data. The runtime never reads train.csv or reports. The validation score is not a claim about the public/private leaderboard.

## Discrepancies and policies

| Difference or failure | Handling |
|---|---|
| Three example studies vs roughly 1,300 hidden studies | Discover current inputs; enumerate sample IDs dynamically; require coverage agreement with test.csv. No fixed study count. |
| No Report at scoring time | No report dependency. |
| Fluid_Sensitive and Fat_Suppression differ | This CNN uses neither flag; no equality assumption, imputation between them or series filtering. |
| Different prevalence | Preserve raw sigmoid scores. No threshold, class balancing at inference, prior correction or batch-dependent ranking. |
| Different numbers of series/slices | Validate each series independently, select deterministic centers from acquired slices, use all usable series as in training validation. |
| Training augmentation and series subsampling | Disable augmentation; no random four-series cap at scoring. |
| Native DICOMs vs normalized training caches | Execute exported native_preprocessing.py with its saved config, including float16 intermediate representation; then call exported prepare_series with rng=None. |
| Scanner intensity, resolution, orientation differences | Use original series percentile scaling, modality LUT, photometric inversion, native orientation handling, full-FOV resize and pixel support rules. Do not invent new resampling rules. |
| Compressed transfer syntaxes | Startup inventory for all documented formats, then per-series syntax checks and pydicom's decoder dispatch. Missing dependency errors remain fatal and actionable. |
| Missing/nonstandard Anatomical_Plane | Normalize case/whitespace. If still absent/unknown, infer from an unambiguous normal in ImageOrientationPatient, and log it. Ambiguous/missing geometry makes the series unusable. Valid supplied planes are not overridden. |
| Missing series table | Discover directories only for requested test studies and infer plane from DICOM headers. |
| Extra series-table studies | Ignore records outside requested sample IDs and record their count. |
| Exact duplicate metadata rows | Deduplicate and report. Conflicting duplicate series records make that series unusable. |
| Unreadable DICOMs, irregular spacing, unsupported geometry | Skip the affected series and record the exact exception. Do not interpolate gaps or mix inconsistent slices. |
| No usable series | Emit configured emergency scores (0.5 per target) and count a fallback study. No all-empty pooling. |
| Corrupt weights/code or nonfinite model output | Stop with diagnostic receipt. Do not hide model/infrastructure defects behind fallback. |
| Large hidden dataset | Process one series at a time; retain only 1,024 features per valid series, append audit records to disk, bound image batches. |
| CUDA image batch allocation failure | Retry the same image range with a smaller batch down to one. Persistent failure remains fatal. |

Fallback and partial-study inference preserve execution coverage but their predictive accuracy has not been validated on a broad real-data set. Training filtered for usable normalized studies, so these policies represent a deliberate extension beyond the healthy validation path.

## Per-study inference

1. Sort the study's series by SeriesInstanceUID. Read headers and check pixel-decoder availability.
2. Run the saved strict native preprocessing to reproduce the [64,320,320] float16 representation and valid acquired slice count.
3. Use the saved center count (four for the current experiment), rounded evenly spaced centers and clipped preceding/center/following slice indices. These refer to retained acquired slices, exactly as during validation.
4. Use the saved full-FOV bilinear-antialiased resize (224×224 for the current model) and mean/std buffers. No crop, random intensity scaling or test-time augmentation.
5. Run ResNet34 in evaluation mode. Concatenate mean and maximum 512-dimensional slice features to form a 1,024-dimensional series vector.
6. Average all usable series vectors equally. Apply the trained linear head once, then sigmoid once in float32.
7. Place the twelve probabilities in the exact sample row and column order.

The streaming implementation reproduces the exported model's feature aggregation. It is tested against that original forward method rather than merely duplicating a shape check. GPU autocast matches training validation; full GPU and decoder parity still need verification in Kaggle.

## Output checks

Require one row per requested ID, unique nonempty IDs, exact sample columns/order and finite scores between zero and one. Write a temporary CSV, read it back, revalidate, then atomically rename it to submission.csv. Archive a prior submission before a new run so a failure cannot leave a stale file masquerading as the current result. Record a failure receipt and traceback if an error occurs, including the processing stage and current study when known.

An 8.5-hour software budget leaves some margin under the saved nine-hour competition rule; it is not a performance guarantee. A single slow series is not forcibly interrupted. No artificial timing/score signals are used to probe hidden data.

## Appendix

- **Transfer syntax:** DICOM pixel encoding/compression format.
- **Decoder:** Software converting stored pixel data into image arrays.
- **Triplet / 2.5D:** Adjacent slices used as three channels of a 2D CNN input.
- **Pooling:** Mean/maximum operations that combine features.
- **Partial study:** Prediction based on valid series after one or more series were rejected.
- **Fallback:** Explicit emergency scores when no image-based prediction is possible.
- **Manifest:** Saved model configuration, identity and file hashes.
- **Sigmoid:** Conversion from a model logit to a score between zero and one.
