# Kaggle training notebook

Upload `train-knee-dinov2-attention.ipynb` into a new **private** Kaggle notebook. The file is self-contained; no local Python support files need to be attached.

## Inputs and settings

Attach:

- The complete latest `gany24558/rsna-knee-normalized-training-data` dataset. All active shards must have status `processed`. Legacy archives are ignored. Labels, weights, masks and folds are already included.
- The competition dataset, for verification against the original `train.csv`.
- Generic public DINOv2 Small weights: a Hugging Face `facebook/dinov2-small` folder containing config/weights, or Meta's `dinov2_vits14_pretrain.pth`. Set `ENCODER_PATH` if discovery finds multiple choices. Do not substitute an already fine-tuned knee encoder.

Enable a GPU, keep Internet off, then **Save Version → Run All**. No online installation or download is performed. The notebook has explicit messages for missing inputs/packages. Use a Kaggle runtime with torch, numpy, pandas, sklearn and either transformers or timm; record the image/version used for a reproducible full training run.

Default: frozen DINOv2, all valid candidate slice features, 16 centers per series during head training, finding-specific series attention, masked target-balanced BCE, five folds, EMA and early stopping. The first configuration cell controls paths and hyperparameters. Optional ranking loss and verified-only refinement are disabled by default; image-encoder fine-tuning is not implemented in this first baseline.

## Long runs and resumption

The conservative session budget is 7.5 hours; this is not a prediction that the full dataset fits one session. Feature extraction is resumable per series, and head training resumes from the last completed epoch. Save partial output privately and attach it to the next run. Compatible previous outputs are discovered automatically, or specify their run directory in `RESUME_ROOTS`.

Each normal resumed run carries cached features forward into its own output, so retain the newest cumulative output. A run folder contains `run_identity.json`, `features/`, `folds/` and status/configuration files. The notebook never changes mounted inputs. Do not delete old output until the resumed output has been saved successfully.

`MODE='features'` only builds the feature cache; `MODE='train'` requires the complete existing feature cache; `MODE='all'` does both. `FOLDS=[0]` is a one-fold pilot; a later all-fold run can reuse its checkpoint. The manifest explicitly marks incomplete OOF coverage.

If preprocessing shards are incomplete, finish preprocessing first. If the preprocessing implementation fingerprint changes, update the embedded inference transforms before proceeding. Shape compatibility alone is insufficient.

## Outputs

Under `/kaggle/working/rsna-knee-training/<run-id>/`:

- `model_package/`: shared encoder, selected fold heads, spacing scalers, ordered target schema, manifest and hashes, runtime and native preprocessing code/configuration, recorded dependencies, and synthetic head-reload reference.
- `features/`, `folds/`: resumable feature and training state.
- `oof_private.csv`, `validation_private.json`, `label_audit_private.csv`: development evaluation and coverage.
- `status.json`: complete/partial execution status.

Keep the notebook and complete output private because they contain derived competition data. The model package omits scans, reports, cached patient features, labels and study-level OOF rows. Review model licenses and competition distribution requirements before publishing any artifact. This training notebook does not upload, submit or produce `submission.csv`; inference is a separate deliverable.

## Validation performed locally

- Notebook schema validation and compilation of every code cell.
- Eleven synthetic tests covering archive/label contracts, missing labels, target-balanced loss, group isolation, padding and permutation behavior, slice sampling, metrics, training/checkpoint resume, and offline HF/timm encoder loading and export.
- Full execution of all notebook cells with synthetic tar shards and locally initialized DINOv2 weights, completing five folds, OOF export, runtime compilation and feature-cache resumption.

These tests verify mechanics, not real-data model accuracy, GPU runtime, or the currently attached Kaggle dataset version. The real-data Kaggle GPU run remains to be performed. The current small verified set also has prior development exposure; no independent clinical-validation claim is made.
