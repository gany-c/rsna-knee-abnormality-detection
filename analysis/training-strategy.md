# Training Strategy

**Created:** October 7, 2026 (Asia/Kolkata)

Improve the current ResNet34 systematically toward a public score of 0.90. This is an objective, not a promised outcome. Experiment history is recorded in [Training Progress Results](training-progress-results.md).

## 1. Current baseline and completed experiments

| Experiment | Public score | Finding |
|---|---:|---|
| Resumed four-triplet ResNet34 — submission V9, model `study-mil/4` | **0.839** | Keep as the reference model. |
| Eight-triplet ResNet34 — submission V14 | 0.831 | More triplets did not improve this run. |
| Earlier ResNet + DINOv2 hybrid — submission V10 | 0.830 | Did not beat its standalone ResNet component (0.831). |
| DINOv2: original 3,917 studies / expanded 4,407 studies | 0.756 / 0.748 | Expansion did not improve this experiment; continue with the original image set for controlled comparisons. |

The baseline uses community soft labels alongside official competition labels. Keep the four-triplet configuration and original 3,917-study image set fixed while establishing better validation. Preserve the eight-triplet notebook as a completed experiment.

## 2. Next priority: stronger validation

The current ResNet split evaluates against only **nine officially labelled studies**. That is too small to confidently rank many model changes.

- Build **five-fold validation across all 58 officially labelled studies**. For each fold, train a separate model and predict the official studies held out from that model. Concatenate these out-of-fold (OOF) predictions for evaluation.
- Exclude all supervision for held-out studies from that fold's training, including community labels. Group by patient where reliable patient identifiers exist; otherwise split by study and document that patient separation cannot be verified.
- Keep preprocessing, four-triplet sampling, labels, and training settings fixed for the baseline comparison. Save fold assignments, predictions, checkpoint details, and configuration.
- Report overall and per-finding AUROC, label coverage, and uncertainty. Skip and report findings without both positive and negative official examples. Do not average fold AUROCs as a substitute for the pooled OOF metric.
- Track agreement with community labels separately using a fixed report-labelled holdout. Soft-label loss measures agreement with that supervision, not independently verified medical truth.
- Account for checkpoint selection using validation labels: these OOF results remain development estimates, not an untouched final test. The 58 official studies have already influenced our experiments.

This requires additional training: the existing single-split checkpoint cannot produce genuine five-fold OOF predictions by itself. Use resumable checkpoints and plan the GPU budget before starting. Establish a validation recipe first, then compare candidates using the same folds and selection rules.

## 3. Next model experiment: cover the imaging planes

Keep **four triplets per series**. Change series selection to cover available sagittal, coronal, and axial views instead of choosing up to four series purely at random.

Respect missing views and available series; keep fluid sensitivity and fat suppression as separate attributes. Compare against the baseline with the same folds and training budget. Verify that inference uses compatible preparation and measure runtime.

This is a separate experiment from increasing four to eight triplets. It does not inherently require renormalizing the images if the required series and metadata are already in the cache.

## 4. Subsequent experiments, one change at a time

| Priority | Experiment | What to check |
|---|---|---|
| 1 | Image framing | Inspect exact model inputs; compare full-field images with a checked physical knee crop at 224 pixels. Ensure matching training and inference transforms. |
| 2 | Higher resolution | Test 288 only after framing. Upsizing the existing cache cannot recover lost detail; genuine additional detail may require rebuilding from DICOM. |
| 3 | Supervision and loss weighting | Audit masking and normalization. Uniform community weight 0.25 cancels under the current per-study weighted-loss normalization; it is not a quarter-strength study contribution. Test explicit study-level weighting separately if desired. |
| 4 | Label smoothing | Compare epsilon 0.05 with 0.0 while holding other settings fixed. |
| 5 | Finding-specific attention | Let each abnormality learn which slice or series features to emphasize instead of relying on the same averaged representation. |
| 6 | Ensembling | Revisit after stronger individual models exist; choose blend weights using compatible held-out predictions rather than repeated public-score tuning. |

Larger backbones and large ensembles are lower priority until these experiments identify the current limitation.

## 5. Radiologist review: ready, not yet labelled

**No radiologist labels are available yet.** The local viewer is ready, with **20 studies from the excluded 490** downloaded and grouped under “Excluded training studies.”

Reserve these studies from training if we intend to use their future labels as an additional evaluation set. Confirm the reviewed findings and handle uncertain or unanswered findings as missing, not negative. Freeze the evaluation labels and membership before comparing models against them.

This is a deliberately selected excluded-study set, not a random sample of the competition population. Report its results separately. If we later use these labels to train a model, those studies cease to be an independent evaluation set for that model.

## 6. Experiment discipline

Record the exact data subset, folds, model version, configuration, validation results, runtime, and public score in the progress report. Change one major factor at a time, retain the best existing checkpoint, and assess improvements on consistent validation data. A single public-score change does not establish that a modification caused a general improvement.
