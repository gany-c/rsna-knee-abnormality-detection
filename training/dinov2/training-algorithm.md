# Knee MRI training notebook specification

Status: proposed implementation, not an executed or validated training recipe.

## 1. Purpose and scope

Build a reproducible study-level classifier using a frozen DINOv2 Small image encoder and a finding-specific attention head. Train and validate five fold models, export their weights and configuration, and optionally test encoder fine-tuning later. The existing preprocessing notebook supplies normalized images; the separate inference notebook consumes the trained model package.

This design adapts cached-feature training and target-conditioned attention from `resources/reference_notebooks/roman-tamrazov-rsna-knee-dinosaur-v5-train.ipynb`. That notebook trains a head on an already-trained DINOv3 encoder; it does not establish a complete, leakage-free encoder-training recipe. `resources/reference_notebooks/evgendvorkin-rsna-baseline.ipynb` is an inference-ensemble reference. Neither notebook's score establishes the performance of this proposed model.

## 2. Targets and input contract

Use this exact initial target order, and verify it against the current competition submission schema before implementation:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

One example is one study containing multiple series. Inputs comprise the complete-study training manifest, normalized series arrays, image-support and valid-slice masks, physical positions/spacing, acquisition metadata, labels, reliability weights, and fold assignments. Current documented arrays are `64 × 320 × 320`; verify the actual cache contract instead of assuming every depth entry is acquired data.

For each study–target pair, use a verified label first, otherwise an accepted generated label, otherwise an unknown-label mask. Unknown targets never become negatives or supervised 0.5 targets. Reports supply labels, not image-model input features.

The documented accepted generated-label export primarily covers ACL and medial meniscus. Audit the current files and report positives, negatives, unknowns, and sources per target/fold; do not claim comparable supervision for all twelve outputs.

## 3. Data and validation audit

- Validate preprocessing identity, shard completeness, unique study/series keys, cache readability, shapes, finite pixels, and mask consistency.
- Preserve existing five-fold assignments only after auditing patient grouping and duplicate examinations.
- For fold k, exclude every study of held-out patients from training, including weakly labeled studies. Fold -1 is not permission to include a related held-out patient.
- Evaluate against verified labels only. If patient identity is unavailable, state that evaluation is study-separated.
- Fit numeric metadata normalization and any sampling statistics on training-fold data only.
- Use generic pretrained DINOv2 weights initially. A fixed generic encoder may be shared across folds; a task-adapted encoder must exclude the held-out groups for each fold.
- Record that the small verified set already informed label-pipeline development. These folds are development evaluation, not an untouched independent test.

## 4. Image inputs and frozen feature cache

1. Retain the existing full-field-of-view normalization and orientation conventions.
2. Identify real slices using explicit masks, not pixel intensity alone.
3. Baseline: repeat one grayscale slice across three channels. Comparison: preceding/center/following valid slices, repeating the nearest valid slice at boundaries.
4. Preserve slice positions: adjacent cached indices need not be physically adjacent original slices.
5. Pad 320-pixel images symmetrically to 322 pixels, divisible by DINOv2's 14-pixel patch size. Apply the chosen encoder's expected channel normalization. Record both transformations.
6. Run a pinned generic DINOv2 Small ViT-S/14 checkpoint in evaluation mode without gradients.
7. Concatenate the 384-dimensional CLS token and 384-dimensional support-weighted patch mean to obtain a 768-dimensional vector.
8. Cache features for all valid candidate centers, in image microbatches. This permits changing center subsets without rerunning the encoder.
9. Key the cache by input/preprocessing identity, encoder hash, library/model implementation, image representation, resolution, normalization, and feature-pooling definition. Store study/series IDs, positions and metadata with features.

Support-weighted pooling reduces padding's contribution at readout; it does not make the encoder's internal attention ignore padded patches. Keep padding consistent and do not claim full padding invariance.

## 5. Series and study model

Select up to 16 centers per series. During training, sample one center per approximately equal interval of valid depth; during validation use deterministic evenly spaced centers. Short series use fewer real centers and masks, rather than duplicated padded instances. Keep all usable series initially.

For each series, concatenate the mean and element-wise maximum of its valid 768-dimensional slice features. Project the resulting 1,536-dimensional vector to 256 dimensions with layer normalization, a linear layer and GELU. Add learned embeddings for plane, fat suppression, and fluid sensitivity/sequence category; use explicit unknown categories. Add a small projection of training-normalized numeric spacing metadata.

Across series, use multihead attention with 256 hidden dimensions, four heads and twelve learned finding queries. Mask missing series. For each finding, concatenate its attended representation, query, global valid-series mean, absolute attended-minus-mean difference, and element-wise attended-times-mean product. Feed the 1,280-dimensional result through a small shared fusion layer and finding-specific output weights to produce twelve logits.

Reject all-empty studies rather than permitting all-masked attention. Attention is a learned aggregation mechanism, not verified lesion localization.

## 6. Loss and sampling

Use unreduced binary cross-entropy with logits. For target t:

`L_t = sum_i(mask_it * reliability_it * BCE(logit_it, label_it)) / sum_i(mask_it * reliability_it)`

Average `L_t` across targets with positive total weight in the batch. Skip a batch with no supervision. Replace missing numeric labels before BCE, then mask them; NaN multiplied by zero remains NaN.

Verified reliability starts at 1. Audit and preserve accepted generated-label weights on a documented scale. Do not infer confidence from distance from 0.5 for binary labels. Start with approximately 25% verified and 75% weak-only studies per batch, sampling verified studies with replacement if necessary. Report actual exposure; do not combine extreme oversampling with an untested large verified-label multiplier.

## 7. Initial head-training settings

| Setting | Starting value |
|---|---|
| Optimizer | AdamW |
| Head learning rate | 7e-4 |
| Weight decay | 2e-3 |
| Batch | 64 study representations; adjust for memory |
| Maximum epochs | 32 |
| Attention dropout | 0.10 |
| Fusion dropout | 0.15 |
| Series dropout | 0.10; retain at least one originally valid series |
| Gradient norm limit | 2.0 |
| EMA decay | 0.995 |
| Validation | Every epoch |
| Early stopping | Five epochs without validation-loss improvement |

Pin seeds and record library versions. Log actual learning rate, losses, label exposure, time and memory. A fixed learning rate is the initial recipe; scheduler changes are separate experiments.

For each batch: select cached centers, pool series features, apply training-only regularization, compute logits and masked loss, backpropagate, clip gradients, update parameters, then update EMA. Evaluate current and EMA weights under the same predetermined checkpoint rule. Save the best checkpoint, not simply the last epoch.

## 8. Optional experiments, in order

1. Compare a simple pooled head with finding-specific attention.
2. Compare repeated single-slice input with three-slice input; these require distinct feature caches.
3. Add verified-only positive–negative ranking loss: mean `softplus(-(positive_logit - negative_logit))`, averaged across eligible targets. Start at coefficient 0.05; skip targets without both verified classes in the batch.
4. Test verified-only refinement from the best mixed-supervision checkpoint: learning rate 1e-4, at most five epochs, validation each epoch. Preserve the pre-refinement checkpoint as a candidate.
5. Test central-patch features or learned within-series attention separately.
6. Only after the frozen baseline: unfreeze the final two encoder blocks and final normalization, start encoder learning rate at 5e-6 and head rate at 1e-4, and recompute image features with gradients. Train separately per fold. Use memory-aware microbatches/gradient accumulation and checkpoint activations if needed.
7. Compare 16 versus 32 deterministic inference centers for accuracy/runtime.

Limit comparisons to avoid repeatedly overfitting the small validation set. Do not adopt Roman's final blend weights or encoder without provenance checks.

## 9. Validation and deliverables

Checkpoint selection initially minimizes verified-label validation BCE, averaged equally across evaluable targets. Report per-target AUROC, precision–recall AUC, positive/negative counts, and selection loss. AUROC is undefined with only one class; record it as unavailable. State the averaging and PR-AUC implementation explicitly.

After five folds, save out-of-fold predictions and calculate pooled metrics with patient-group bootstrap uncertainty where identity is available. Preserve per-fold results because cross-fold score calibration may differ.

Export a model package containing:

- Manifest with ordered targets, architecture, encoder provenance/hash, preprocessing contract, input representation, feature pooling, metadata vocabulary/scalers, sampling policy, fold identifiers, selected epochs and weight type.
- Selected head weights per fold; encoder weights or an exact offline-resolvable encoder reference, and per-fold encoder weights if fine-tuned.
- Fold-specific metadata transforms and validation predictions/metrics.
- Training configuration, source/data fingerprints, environment versions, seeds and logs.
- An inference smoke-test reference: a permitted sample input and expected outputs/tolerance, or its reproducible identifier and preparation recipe.

Test a save/reload round trip, variable series counts, all-empty handling, unknown-label masking, held-out-group exclusion, and deterministic inference. Do not require a full training run to check these contracts.

## 10. Suggested notebook cell groups

Configuration → input audit → folds/labels → encoder loading → feature-cache build → model/loss definitions → single-fold smoke test → five-fold training → optional controlled experiments → OOF evaluation → package export and reload verification.

## Appendix: terms

| Term | Meaning |
|---|---|
| Study / series / slice | One examination / an acquisition within it / one 2D image. |
| Target | One abnormality to predict; multiple targets can be positive together. |
| Weak label | An imperfect training label, here derived from a report. |
| Mask | An explicit indicator of valid images, series, or known labels. |
| Backbone / encoder | The image network producing numerical features. |
| Frozen / fine-tuned | Parameters unchanged / adapted through training. |
| ViT-S/14 | Small Vision Transformer with 14-pixel-square patches. |
| CLS token / patch feature | Image-summary token / representation of an image region. |
| Pooling | Combining several feature vectors into a summary. |
| Embedding | A learned numerical representation of an input or category. |
| Query / attention | A learned request for information / weights combining available representations. |
| Head / logit / sigmoid | Prediction network / raw score / conversion to the interval 0–1. |
| BCE | Binary cross-entropy, the main prediction-error objective. |
| Ranking loss | Encourages verified positive cases to score above negative cases. |
| AdamW / learning rate | Parameter-update algorithm / update scale. |
| Weight decay / dropout | Parameter regularization / randomly hiding inputs or features during training. |
| Epoch / batch | One scheduled training-data pass / examples processed together. |
| Gradient clipping | Limiting update-driving gradients to stabilize training. |
| EMA | Exponential moving average of trained parameters. |
| Fold / OOF | Held-out partition / predictions made without training on that partition. |
| Leakage / provenance | Inappropriate held-out information exposure / training-history record. |
| AUROC / PR-AUC | Ranking discrimination / precision–recall summary. |
| Macro average | Equal-weight average across targets. |
| Checkpoint / early stopping | Saved parameters / stopping after validation stops improving. |
| Ablation | A controlled experiment changing one component. |
| Calibration | Agreement between predicted probabilities and observed frequencies. |

## References

- Local Roman and evgendvorkin notebooks in `resources/reference_notebooks/`.
- [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).
- [Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html).
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
