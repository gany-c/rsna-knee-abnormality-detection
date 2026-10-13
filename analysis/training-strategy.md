# Training Strategy

**Created:** October 7, 2026 (Asia/Kolkata)

Improve the current ResNet34 systematically toward a public score of 0.90. This is an objective, not a promised outcome. Experiment history is recorded in [Training Progress Results](training-progress-results.md).

## 1. Current baseline and completed experiments

| Experiment | Public score | Finding |
|---|---:|---|
| ResNet34 members 0 + 2 — ensemble submission V3 | **0.851** | Completed; equal average of member probabilities. |
| Resumed four-triplet ResNet34 — submission V9, model `study-mil/4` | 0.839 | Earlier single-model reference. |
| Eight-triplet ResNet34 — submission V14 | 0.831 | More triplets did not improve this run. |
| Earlier ResNet + DINOv2 hybrid — submission V10 | 0.830 | Did not beat its standalone ResNet component (0.831). |
| DINOv2: original 3,917 studies / expanded 4,407 studies | 0.756 / 0.748 | Expansion did not improve this experiment; continue with the original image set for controlled comparisons. |

The baseline uses community soft labels alongside official competition labels. Keep the four-triplet configuration and original 3,917-study image set fixed while establishing better validation. Preserve the eight-triplet notebook as a completed experiment.

## 2. Completed ResNet34-only ensemble training and inference — October 13, 2026

Trained three ResNet34 members using the original **3,917-study image set** and **four triplets per series**. Each member used **3,802 training studies**, **15 officially labelled validation studies**, and the same **100 non-golden selection studies**, excluded from training and validation.

Member 0 retained its original holdout. Members 1 and 2 used the revised balanced plan. The final three validation sets are disjoint and cover **45 of 58** officially labelled studies; **13 remain unvalidated**. This is three holdouts, not exhaustive cross-validation. Members 1 and 2 have at least three positives and three negatives per finding in validation; member 0 retains only one MCL-positive validation study. Groups use study IDs, so patient-level independence is unverified. See the [balanced holdout report](outputs/balanced-holdouts-2026-10-10/report.md).

### Training checkpoints

| Member | Model version | Best epoch | Official holdout BCE | Official holdout macro AUROC |
|---|---:|---:|---:|---:|
| 0 | 2 | 15 | 0.493499 | 0.797465 |
| 1 | 2 | 10 | Not available in the locally saved outputs reviewed | Not available in the locally saved outputs reviewed |
| 2 | 2 | 12 | 0.507696 | 0.780401 |

Members 0 and 2 stopped by early stopping in the saved training notebooks. Best epochs and model versions for all three members are confirmed by the saved comparison notebook. Holdout metrics are measured on different study sets and are not directly comparable. Checkpoint selection used validation loss.

### Inference on the common 100-study selection set

| Members | Soft-target BCE |
|---|---:|
| 0 | 0.490621 |
| 1 | 0.491369 |
| 2 | 0.491280 |
| 0 + 1 | 0.487255 |
| 0 + 2 | 0.486586 |
| 1 + 2 | 0.488955 |
| 0 + 1 + 2 | 0.486506 |

Ensembles use equal probability averages. These results measure agreement with community soft labels on a selection set, not an independent test or verified clinical accuracy. Source: [saved member comparison](compare-resnet-members.ipynb).

### Competition inference

| Submission | Version | Configuration | Status | Public score |
|---|---:|---|---|---:|
| submit-knee-resnet34-ensemble-0-2 | 3 | 50% member 0 (epoch 15) + 50% member 2 (epoch 12); both model version 2 | Succeeded | **0.851** |

The result was recorded on October 13. See [Training Progress Results](training-progress-results.md). The earlier five-fold validation proposal has not been completed; the holdout procedure above is the experiment actually run.

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
| 6 | Further ensembling | Members 0 + 2 completed with public score 0.851; use compatible held-out predictions for any further blend selection. |

Larger backbones and large ensembles are lower priority until these experiments identify the current limitation.

## 5. Radiologist review: eight studies labelled — October 13, 2026

**Eight studies were labelled in today’s review, as reported by the user.** The local viewer has **20 studies from the excluded 490** downloaded and grouped under “Excluded training studies.” The saved labels and completion status have not been audited as part of this update.

Reserve these studies from training if we intend to use their future labels as an additional evaluation set. Confirm the reviewed findings and handle uncertain or unanswered findings as missing, not negative. Freeze the evaluation labels and membership before comparing models against them.

This is a deliberately selected excluded-study set, not a random sample of the competition population. Report its results separately. If we later use these labels to train a model, those studies cease to be an independent evaluation set for that model.

## 6. Proposed next-iteration pooling experiment — October 13, 2026

Start with a small **gated-attention pooling layer across series**, replacing the current equal average of series vectors. Retain the existing mean/max pooling of triplet features within each series. Keep slice sampling, data, validation splits, and training settings fixed to isolate the fusion change; train and evaluate a new model.

If this improves validation results, test **finding-specific series attention**, allowing each of the twelve targets to learn its own series weights. Plane and sequence metadata can be considered as a separate extension. A small Transformer for interactions between series is a later, more complex option.

These are proposed experiments, not implemented changes or established improvements. Attention weights are not proof of diagnostic importance or lesion localization, and pooling cannot recover findings absent from the sampled images.

## 8. Proposed sequence-aware enhancements — October 13, 2026

Treat **fluid sensitivity** and **fat suppression** as separate sequence properties: emphasizing water signal and suppressing fat signal are not equivalent. Intensity normalization does not substitute for either acquisition property.

**Training-data limitation:** The data description states that fluid sensitivity and fat suppression perfectly coincide in this training data. Their flags therefore provide the same information here; this dataset cannot establish their independent effects. Preserve separate metadata fields because the properties can differ in unseen data, but do not assume that separate flags alone provide robustness to unfamiliar combinations.

1. **First experiment: complementary series selection.** Extend the plane-coverage proposal in Section 3 to consider sequence contrast as well. Select a complementary set of available planes and sequence types, including fluid-sensitive/fat-suppressed and useful non-fat-suppressed series, rather than choosing up to four series purely at random. Select only among combinations actually available; do not require fluid-sensitive, non-fat-suppressed series or other combinations absent from the dataset. Keep the series budget and triplets per series unchanged. Handle missing sequence types without discarding otherwise usable studies.
2. **Separate experiment: metadata-aware fusion.** Add plane, fluid-sensitivity, and fat-suppression descriptors to series vectors for the attention layer proposed in Section 7. Audit DICOM metadata and series-name mappings first; retain an explicit unknown category for uncertain or missing attributes.
3. **Later extension: finding-specific attention.** Let each target learn which series to emphasize from image features and sequence descriptors instead of imposing fixed diagnostic weights by sequence type.

These are future experiments, not implemented changes or demonstrated gains. Compare each change separately on the same validation splits, with compatible training and inference handling. Random splits of this dataset cannot establish performance on sequence combinations absent from training. Evaluating that generalization requires additional examples containing those combinations. Complementary series selection remains the first experiment; metadata-aware attention remains a later experiment without an assumption of independent information from the two flags.

## 9. Radiologist discussion and proposed sampling experiment — October 13, 2026

The following are observations reported from today's eight-study labelling session, not established rules for every study:

1. **Baker’s cyst:** Axial images around the middle of the stack are useful; sagittal images may also provide a good view.
2. **ACL and PCL:** The middle slices of the sagittal series are particularly important.
3. **Contusion:** Both sagittal and coronal **fat-suppressed** images are useful.
4. **Bone fracture:** The coronal plane was emphasized as particularly useful.
5. **Meniscal tears:** Reviewing the entire sagittal stack is important because different portions of the menisci appear across successive slices.
6. **Sampling implication:** Consider giving more sampling coverage to the middle slices, particularly in sagittal and coronal series, while preserving coverage throughout the stack so that peripheral findings and meniscal tears are not missed.

**Proposed experiment:** compare the current evenly distributed prediction sampling with sampling that allocates more triplets to the middle of coronal and sagittal stacks while retaining some peripheral coverage. Keep the total triplet budget, training data, and validation splits fixed. Define the central region and sampling allocation before the experiment; no exact percentages have been chosen. Use compatible training and inference policies and train a new model rather than silently changing sampling for existing checkpoints.

Before translating “middle” into indices, check the viewer's slice ordering against the normalized stack's anatomical direction and field of view. Do not assume a displayed end corresponds to the same anatomy across studies. Plane and sequence observations should guide experiments without excluding all other views or choosing inputs using the true diagnosis. Findings outside the emphasized region remain a reason to preserve peripheral sampling.

These observations have already informed development proposals, so the eight reviewed studies should not be described as an untouched independent test for those proposals. No sampling or model code was changed by this note.

## 10. CoPAS paper and proposed plane-aware fusion experiments

Source: Qiu et al., **Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities**, Nature Communications (2024). [Published paper](https://doi.org/10.1038/s41467-024-51888-4); [local PDF](../resources/NaturePaper.pdf).

### What the paper implemented

CoPAS uses 3D ResNet18 encoders and three branches corresponding to sagittal, coronal, and axial planes. Cross-plane attention lets each branch incorporate information from other views. Co-plane cross-sequence attention combines different contrasts within a plane, including coronal PDW/T1W and sagittal PDW/T2W. Rotating volumes presents existing information along other axes; it does not create new acquired detail.

Each branch produces twelve predictions. Their products form a 12 × 12 × 12 array, and a learned correlation module adjusts the final predictions using interactions between planes and abnormalities. This is learned fusion, not a fixed probability boost based on co-occurrence. Training supervises the individual branches with BCE and the final prediction with focal loss.

### Reported results and limits

The study included **1,748 patients from five centres**: 773 training, 110 validation, 220 internal-test, and 645 external-test patients.

| Evaluation | CoPAS average AUROC |
|---|---:|
| Internal test | 0.812 |
| External centres B + C | 0.721 |
| External centres D + E | 0.726 |

External evaluations involved different sequence availability and centre populations. These scores are not directly comparable with our 0.851 public score: the populations, label definitions, and twelve targets differ. CoPAS includes PCL, LCL, and synovial plica, whereas our task includes fracture, synovitis, and separate compartment-specific OA targets. Its cohort included patients undergoing arthroscopy and excluded motion/metal-artifact cases. The paper acknowledges parameter count, overfitting risk, and dependence on three imaging planes. It does not establish that middle-only sampling improves performance.

### Proposed experiments for our pipeline

1. **Preserve plane identity before fusion.** Build sagittal, coronal, and axial summaries instead of immediately averaging all series together. Handle missing or unknown planes explicitly.
2. **First CoPAS-inspired experiment: plane-aware learned fusion.** Use a small learned fusion module over those summaries, retaining our existing 2D ResNet encoder, triplet sampling, data, and validation splits. This extends the simpler series-attention proposal in Section 7; compare the variants separately rather than combining them with sampling changes.
3. **Later: auxiliary predictions per plane.** Test supervision of both plane-specific outputs and the final output. Preserve unknown-label masks at every supervised output. Do not automatically switch to focal loss at the same time.
4. **Later: condition-dependent fusion.** Investigate learned interactions between condition predictions, using the paper's correlation module as a reference rather than imposing fixed clinical probability boosts. Evaluate against simpler fusion with the same held-out data.
5. **Separate larger experiment: volumetric context.** Consider 3D encoders only as a distinct change requiring assessment of preprocessing, spatial geometry, memory, and runtime. Do not copy the paper's volume rotations or ROI cropping without checking compatibility and anatomical coverage for our targets.

These are proposals, not implemented changes or demonstrated improvements on our dataset. The immediate architectural experiment is plane-aware learned fusion; the radiologist-inspired sampling experiment in Section 9 remains a separate comparison.

## Learning strategy

1. Understand fluid sensitivity and fat suppression 
2. Read the 2 new documents provided by Dr. Amitesh Khare
3. Understand how missing labels are masked during training.
4. Explore new ideas from the discussion forums.
5. Read through the training notebook code.
