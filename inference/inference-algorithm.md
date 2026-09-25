# Knee MRI inference and submission notebook specification

Status: implemented in `submit-knee-dinov2-attention.ipynb`; synthetic end-to-end validation passed, real Kaggle validation pending. Companion: `../training/dinov2/training-algorithm.md`.

## 1. Purpose

Load the exported model package, prepare unseen studies using the training image contract, predict twelve abnormalities, and write a validated `submission.csv`. No labels or radiology reports are required at inference. No fitting, calibration, or model selection takes place on test data.

The notebook must work with the actual Kaggle offline assets and runtime limits. Verify current competition rules, inputs, output schema, permitted weights and runtime before implementation; historical notebook runtimes do not establish hidden-test feasibility.

## 2. Required inputs

Attach the custom Kaggle Model `gany24558/gc-rsna-knee-dinov2/pyTorch/five-fold-attention` at a selected version. It contains the shared encoder, all five heads and exact preprocessing/runtime code. The submission notebook reads its mounted files offline; it does not require the original Meta model or training caches.

- Test-study table and official sample submission.
- Test series metadata and DICOM files, or compatible normalized caches where explicitly available.
- Exported model manifest, fold head weights, metadata transforms, and encoder weights.
- Pinned dependencies available offline.

The model manifest must identify target order, encoder architecture and hash, image transforms, input channels, patch resolution, feature aggregation, metadata encodings, deterministic slice policy and selected weight type. Fine-tuned models need their own encoder per fold. Refuse incompatible or incomplete packages rather than silently substituting settings.

## 3. Preflight checks

1. Confirm unique requested study IDs and required submission columns.
2. Verify artifact hashes, fold completeness and compatible architecture/state dictionaries.
3. Check available CPU, GPU, RAM and disk; select supported precision for the actual device.
4. Load weights and set every model to evaluation mode.
5. Run the exported smoke-test reference and compare predictions within the declared numerical tolerance.
6. Estimate runtime and memory on representative studies before a full run. Scale measurements by slice and series counts, not only study count.

Use one GPU initially where feasible; the model design must not inherit another notebook's two-T4 requirement unnecessarily.

## 4. Prepare images consistently

Reuse the same versioned preprocessing implementation as training instead of reimplementing it independently.

For each study:

1. Discover series and decode images.
2. Apply the established orientation, slice ordering, intensity handling, full-field-of-view resizing and cache representation.
3. Preserve acquired-pixel support, valid-slice masks, physical slice positions and acquisition metadata.
4. Apply the same quality eligibility rules. Do not introduce a label-based selection gate at inference.
5. Select up to the manifest-defined number of centers, initially 16 per usable series, deterministically over valid depth.
6. Build the manifest-defined channel representation: repeated center slice or preceding/center/following valid slices.
7. Repeat boundary neighbors only as specified; never treat padded depth as acquired slices.
8. Apply the recorded spatial padding, initially 320 to 322 pixels, and encoder normalization.

Do not silently switch to center cropping, laterality flips, different percentiles or denser sampling. Such changes alter the model input distribution and require validation. The 32-center variant is a separately validated configuration.

## 5. Extract and aggregate features

Process selected slice inputs in bounded image microbatches without gradients. Use supported reduced precision for encoder execution if verified; perform numerically sensitive aggregation in float32 as needed.

For each slice input, concatenate the CLS token and support-weighted patch mean. For each series, concatenate the mean and maximum of its valid slice vectors. Project and incorporate metadata using the corresponding fold's trained layers and saved transforms.

A generic frozen encoder can supply shared raw image features to every fold head. If encoders were fine-tuned separately, compute features with each fold's own encoder. Never reuse generic or another fold's cached features for different encoder weights.

The image-support mask affects feature readout but does not automatically suppress padding inside transformer attention. Use the exact behavior established during training.

## 6. Generate study predictions

For each fold:

1. Construct variable-length series representations and explicit validity masks.
2. Apply the twelve finding queries and attention/fusion head.
3. Obtain twelve logits.
4. Apply sigmoid once to get scores in [0,1].

Average the sigmoid outputs equally across the selected fold models. Do not average logits or introduce per-target weights unless that precise method was selected and exported using validation data.

Do not use report information, test labels, public leaderboard feedback, or batch-dependent rank transformations in this initial pipeline. Raw sigmoid outputs are prediction scores; clinical probability calibration has not been established.

## 7. Missing data and failures

Recoverable series-level DICOM/geometry failures are recorded and excluded; available valid series are retained. Missing series metadata for a study is allowed, and no all-masked attention is executed. If no usable series remains, the configured twelve fallback scores (0.5 each by default) are emitted and the study is counted explicitly. These emergency scores are not calibrated probabilities or training-derived priors. Partial-study prediction accuracy remains unvalidated on real competition data.

Model integrity, dependency, encoder/head and nonfinite prediction failures remain fatal. Missing codec errors must be fixed with offline dependencies, not hidden by fallback. Diagnostics count skipped series, partial studies and fallback studies. `audit-knee-training-scans.ipynb` uses the exported preprocessing to evaluate real training scans on Kaggle before resubmission. The actual hidden-run exception is unknown.

## 8. Runtime and reproducibility

- Stream studies in bounded blocks; release decoded images and temporary features promptly.
- Decode each required image once where practical and reuse generic frozen features across heads.
- Use deterministic center selection and evaluation mode; training dropout and feature noise are disabled.
- Record artifact hashes, dependency versions, device, precision, elapsed time, peak memory and input counts.
- Verify that predictions agree within tolerance across microbatch sizes and save/reload cycles.
- Benchmark the complete pipeline, including decoding and preprocessing, not only head execution.

## 9. Validate and publish submission

Verify target names and their order against the actual sample submission. The initial training target list is:

`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.

Before writing:

- Exactly one output row per requested study.
- No duplicate, missing or unexpected IDs.
- Rows aligned to the required submission order.
- Exact required columns, no accidental DataFrame index column.
- All scores finite and within [0,1].
- Failure/fallback counts recorded and consistent with the configured policy.

Write to a temporary CSV, read it back and validate it, then atomically rename it to `submission.csv`. Keep a separate run manifest and failure report; do not add audit columns to the submission.

## 10. Deliverables and notebook layout

Deliver `submission.csv`, a run manifest, per-study processing/failure audit, and runtime summary. Optional per-fold predictions are debugging artifacts and must not be confused with the final submission.

Cell groups: configuration → artifact/input preflight → shared preprocessing → model loading → sample smoke test → batched inference → schema and coverage checks → atomic export and run receipt.

## Appendix: terms

| Term | Meaning |
|---|---|
| Inference | Applying an already-trained model to unseen inputs. |
| Artifact / manifest | Saved model files / their machine-readable configuration and identity. |
| State dictionary | Saved parameter values loaded into a matching model architecture. |
| Hash | A fingerprint used to detect changed or mismatched files. |
| DICOM | Medical imaging file format containing pixels and acquisition metadata. |
| Study / series / slice | Examination / acquisition / individual 2D image. |
| Mask | Explicit indication of which image data or model inputs are valid. |
| Microbatch | A small image group processed to bound memory usage. |
| Reduced precision | Lower-precision computation used for speed or memory after compatibility checks. |
| Feature cache | Saved encoder outputs tied to exact weights and preprocessing. |
| Logit / sigmoid | Raw prediction / mapping into the range 0–1. |
| Fold ensemble | Averaging predictions from separately trained cross-validation models. |
| Calibration | Matching predicted probabilities to observed outcome frequencies. |
| Fallback | A declared alternate output path when normal inference fails. |
| Prior | A training-derived estimate of how common a target is. |
| Deterministic sampling | Selecting the same slice positions on repeated runs. |
| Atomic export | Replacing the final file only after a complete temporary file is ready. |
| Smoke test | A small end-to-end check of compatibility and expected behavior. |
