# ResNet34 knee MRI training algorithm

## Objective

Train a separate CNN that predicts the twelve competition findings from MRI studies. Unlike the frozen DINOv2 baseline, this experiment updates the ResNet34 backbone and its classification head together. The pretrained starting point is torchvision's ImageNet ResNet34.

Notebook: `train-knee-resnet34.ipynb`.
Kaggle model destination: `gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil`.
The model is created/uploaded only after actual training produces a validated checkpoint and the final upload cell succeeds. No CNN weights have been trained on the real competition dataset locally.

## What is borrowed from BirdCLEF

The reference notebooks are `BirdClef2026/training/rare-amphibian-oversample-training-notebook.ipynb` and `family-specific-b0-training-noteboo.ipynb`. Their executable code uses EfficientNet-B3 or B0, Adam parameter groups, cosine warm restarts, best-validation checkpointing and Kaggle uploads.

This experiment incorporates those training features, rather than copying the BirdCLEF algorithm. It does not transfer bird weights, audio transforms, species-specific oversampling, or spectrogram augmentations. MRI studies require grouped splitting, multi-series aggregation and explicit masks for incomplete labels. Patience is three here, as requested, rather than six in the BirdCLEF notebooks.

## Inputs and setup

Attach the complete `gany24558/rsna-knee-normalized-training-data` dataset and the original competition data. Generated report labels, verified-label overrides, confidence weights, fold/group identifiers and normalized MRI tensors are already in the custom dataset. The notebook validates all active shards, archive offsets, label checksums and preprocessing identity. Verified labels must match official train.csv exactly.

Enable a GPU. Internet is enabled for downloading the official ImageNet weights and uploading the trained package. An attached official torchvision ResNet34 state dictionary may instead be supplied through PRETRAINED_PATH. Do not download training packages or new dependencies implicitly. Kaggle's submission notebook must later run offline; this is a training notebook.

The original DINO encoder and its cached feature vectors are not used. CNN features change as its weights learn.

## Train/validation split

1. Start with studies having eligible normalized series.
2. Use GroupShuffleSplit with seed 42 and 20% of groups held out. The remaining approximately 80% of groups form the training pool. Group sizes mean the study percentages can differ.
3. Use the supplied patient_group identifier. If it is only StudyInstanceUID, report study separation without claiming patient separation.
4. Exclude **every study in held-out groups** from training, including weak-labelled studies.
5. Train on studies with at least one known, positive-weight target.
6. Evaluate only verified targets in verified studies from the held-out groups. Held-out weak labels are never used for checkpoint selection.
7. Save the complete split to split_private.csv and report per-target verified positive/negative counts. Fail if no verified validation studies exist.

This is a train/validation holdout, not an independent final test set. The competition hidden test set stays untouched. The small verified validation set makes metrics noisy; targets with only one class have undefined AUROC and are reported as such. This new split is not directly comparable with the previous DINO five-fold pooled OOF score.

## Image preparation

Use the normalized native-plane full-field-of-view float16 arrays already generated for training. Keep the exported native preprocessing functions/configuration for eventual DICOM inference.

For each training study:

1. Randomly choose up to four available MRI series; keep all if fewer exist.
2. Pick up to four slice centers in evenly spaced depth bins. Training randomly chooses a center within each bin; validation uses rounded evenly spaced indices.
3. Build three input channels from the previous, center and next acquired slices. Repeat boundary slices when needed. Never sample depth padding.
4. Resize the full 320×320 image to 224×224 using bilinear interpolation with antialiasing. Do not center-crop.
5. During training, apply a small shared intensity scale of 0.9–1.1 within each series and clamp to [0,1]. No anatomical flips or rotations.
6. Apply ImageNet channel normalization: mean [0.485,0.456,0.406], standard deviation [0.229,0.224,0.225].

Validation uses every usable series, deterministic centers and no augmentation. The size, sampling and channel construction are recorded in the package and must be reproduced in CNN inference.

## CNN architecture

- ResNet34 convolutional backbone initialized from ImageNet.
- Remove its original 1,000-class classifier; each slice triplet becomes 512 features.
- For each series, concatenate the mean and maximum slice feature vectors, producing 1,024 features.
- Average these series vectors to obtain one study representation.
- Apply dropout 0.2 and a linear layer producing twelve logits.
- Use sigmoid only when converting logits to prediction probabilities.

This head uses pooling and a linear classifier, not Transformer attention. All convolutional weights and BatchNorm affine parameters train. BatchNorm running statistics stay at their pretrained values to avoid unstable updates with small study batches. Protocol/spacing metadata is not included in this first CNN head.

## Loss and optimization

For each target, the label mask indicates whether a usable label exists. Unknown targets contribute zero loss; they never become negative examples. Verified labels keep weight 1; accepted generated-label confidence weights are retained.

Training loss is binary cross-entropy with logits, multiplied by the target mask and confidence weight and divided by the sum of active weights within the study. Accumulate gradients across four studies, including correct normalization for the last shorter accumulation window. Every training study is visited once per epoch in shuffled order.

Use Adam with backbone learning rate 1e-5 and head learning rate 1e-4. Use cosine warm restarts (10-epoch cycles, minimum learning rate 1e-6), mixed precision on CUDA, gradient clipping at norm 1, and image microbatches of eight. Activation checkpointing recomputes CNN activations during backward to reduce memory. These are starting settings, not validated optimal hyperparameters.

## Validation and stopping

After each complete epoch:

1. Evaluate in inference mode, with no augmentation, dropout or gradient updates.
2. Compute one masked BCE over all verified validation targets. Report per-target AUROC and macro AUROC across targets with both classes.
3. If validation loss is strictly below the best previous value, save the best weights and reset the counter to zero.
4. Otherwise increment the counter; equal loss counts as no improvement.
5. **Stop once the counter reaches three consecutive epochs.** Any strict improvement resets it. There is no minimum-delta threshold.

Examples: losses 0.60, 0.55, 0.56, 0.55, 0.57 stop after the fifth epoch. Losses 0.60, 0.60, 0.59 reset the counter at epoch three.

Maximum 35 epochs, with a 7.5-hour session budget. Export only the best fully validated epoch. If time expires mid-epoch, that incomplete epoch is discarded for export; resume from the previous last.pt. If no validated epoch exists, do not upload a model.

## Checkpoints and reproducibility

`best.pt` holds the best validation model. `last.pt` records the last complete epoch, optimizer, scheduler, AMP scaler, random states, patience counter, history and best state. Resume only from your own trusted checkpoint, with the identical configuration, data identity, split and runtime source. Save/attach the private output before starting another Kaggle session. This checkpoint includes a trusted Python serialization and is not intended for arbitrary external sources.

`history.csv`, `validation_predictions_private.csv`, `split_private.csv`, `run_identity.json` and `training_status.json` remain outside the uploaded package. Inspect curves and per-target support; lower training loss alone does not establish better validation ranking.

## Export and separate Kaggle Model

Restore the best epoch and write `model_package/` containing:

- model.pt: trained ResNet34 plus classifier weights;
- cnn_runtime.py and knee_data.py: architecture and preparation helpers;
- native_preprocessing.py and preprocessing_config.json: original DICOM transform contract;
- manifest.json: target order, configuration, aggregate validation metrics, environment, identities and file hashes;
- synthetic_smoke.pt: synthetic inputs and reference outputs;
- requirements.txt and README.md.

Reload the saved checkpoint into a fresh model and compare synthetic predictions before upload. Upload only model_package, never the entire training directory. The final cell uses kagglehub.model_upload to create the separate CNN variation or add a version. New models default to private; existing visibility is retained. Internet/authentication failures leave the local export intact, so only the upload cell needs retrying.

The existing DINOv2 submission notebook does **not** support this architecture. A CNN inference adapter must load this ResNet34 package and reproduce its validation image preparation and pooling before evaluating it on Kaggle. This training notebook does not produce submission.csv.

## Validation status

Local tests use synthetic normalized MRI shards and a real randomly initialized ResNet34, without network access or uploads. They cover forward/backward training, masked-label gradients, grouped holdout exclusion, exact three-epoch stopping, checkpoint restore, export reload and file hashes. Real pretrained-model accuracy, GPU memory, full-dataset speed and successful Kaggle upload remain to be measured when the notebook runs there.

## Appendix: terms

| Term | Meaning |
|---|---|
| CNN | Neural network using learned convolution filters to detect image patterns. |
| ResNet34 | A 34-layer residual CNN; skip connections help optimize deep networks. |
| Backbone / head | Image feature extractor / network turning features into predictions. |
| Fine-tuning | Updating pretrained weights for the new task. |
| 2.5D triplet | Three adjacent 2D slices used as channels to supply nearby depth information. |
| MIL | Multiple-instance learning: a study has many slices but study-level labels. |
| Pooling | Summarizing features with operations such as mean or maximum. |
| Grouped split | Keeping all records from one patient/group on the same side of a split. |
| Validation | Held-out labelled examples used to select the best checkpoint. |
| Weak label | Automatically generated label that may be noisy. |
| Masked BCE | Binary classification loss computed only where labels are known. |
| Logit / sigmoid | Raw model score / conversion to a score between zero and one. |
| AUROC | Ranking metric; 0.5 is random ranking, 1 is perfect ranking. |
| Adam | Optimizer adapting parameter updates using gradient statistics. |
| Scheduler | Rule that changes learning rates over training. |
| Patience | Consecutive non-improving epochs allowed before early stopping. |
| Gradient accumulation | Combining gradients from several small batches before updating weights. |
| Activation checkpointing | Recomputing activations during backward to save GPU memory. |
| BatchNorm | Layer that normalizes features; running statistics are held fixed here. |
| Mixed precision | Using lower precision where suitable to reduce GPU memory and computation. |
| Checkpoint | Saved model/training state for reuse or resumption. |

Reference: [torchvision ResNet34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html). The documented pretrained transform uses a center crop; this MRI experiment deliberately uses the full field of view, and records that difference.
