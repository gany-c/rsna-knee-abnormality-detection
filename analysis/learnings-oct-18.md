# Learnings — October 18, 2026

Compilation of our discussions on MRI series, model design, supervision, and possible experiments. Proposed changes below are not descriptions of features already implemented in the current ResNet model.

## 1. MRI series types and metadata

Fat suppression and fluid sensitivity describe different properties:

- **Fat suppression (FS)** reduces the signal from fat, making fat darker. It does not necessarily make the whole image dark.
- **Fluid sensitivity** makes fluid or increased tissue water conspicuous, typically bright.
- Overall brightness alone cannot reliably identify FS status; sequence contrast and viewer settings also affect appearance.

Each anatomical plane can theoretically contain four combinations:

| Fat suppression | Fluid sensitivity |
| --- | --- |
| No | No |
| No | Yes |
| Yes | No |
| Yes | Yes |

Across sagittal, coronal, and axial planes, this gives 12 possible groups. According to the competition description, FS and fluid sensitivity coincide in the training data, so only the matching combinations populate the six plane × FS groups. This does not establish that every group occurs in every study.

For the current experiment, expanding to 12 slots would add permanently empty groups without teaching the model how to interpret them. Retain both metadata fields independently for auditing. A new combination at inference remains unfamiliar even if the code accepts it.

## 2. Combining information across series

| Approach | Meaning |
| --- | --- |
| Concatenation | Put group feature vectors side by side in fixed slots. A prediction network learns how to use them. |
| Attention | Learn how strongly each available series or group contributes, potentially differently for each study and condition. |
| Separate prediction heads | Each series group predicts conditions first; a subsequent rule or network combines those predictions. |
| Learned fusion | Any trained mechanism for combining features or predictions. Separate prediction heads are optional. |

Concatenation itself does not learn weights; the network after it does. A nonlinear network after concatenation can also learn combinations that depend on the input. Attention provides an explicit weighting mechanism.

A proposed experiment is to pool series within each plane/FS group, then apply condition-specific attention across the available groups. Multiple series of one type become one group vector. Missing groups are masked out. Pooling within groups prevents their representation from growing simply because more series were acquired, although averaging may dilute a particularly useful acquisition.

For fixed six-group concatenation, slot positions already identify plane and FS status. Six availability flags indicate which groups are present; additional FS flags per slot are redundant. If individual series remain separate instead, each series carries its own metadata.

A study with two sagittal series and one coronal series can be handled: place them in the appropriate groups, pool within a group if necessary, and mark absent groups as missing. Reliability still depends on whether similar acquisitions and missing-group patterns were represented in training.

### Shared heads versus independent models

Multiple heads sharing an encoder remain coupled through the encoder. They can be trained with a final combined loss, or with individual head losses as additional supervision. They need not improve at the same speed. Freezing a head does not freeze its predictions if its shared encoder continues changing.

Six independent models each have their own encoder, loss, optimiser, learning rate, and stopping point. One can stop while others continue. Their frozen outputs can later be combined; a learned combiner should use appropriately held-out or out-of-fold predictions.

## 3. Unfamiliar inputs and uneven training coverage

An FS-only model will generally still output predictions for non-FS images, but those predictions may be unreliable or confidently wrong. Intensity normalization does not remove differences in which tissues are bright.

None of attention, concatenation, or separate heads automatically solves a 95% FS / 5% non-FS imbalance. If each study contains only one type, attention across types has no alternative to choose: the available group receives all the attention. That weight is not a reliability score.

A shared encoder and shared prediction head with series metadata is a useful baseline for this scenario. It shares learning across types, but does not create missing training evidence. Evaluate FS-only, non-FS-only, and mixed studies separately, and distinguish a change in their proportions from a genuinely unseen sequence appearance.

## 4. Missing labels and backpropagation

Missing labels are unknown, not negative. A label mask excludes them from the loss and their direct gradients. The model still predicts all 12 conditions.

An internal zero placeholder for a missing target does not become a negative example because its mask is zero. A partially labelled study remains useful through its known targets. Updates to the shared encoder can indirectly change predictions for conditions whose labels were missing.

Community labels may supply targets where official labels are missing. Validation uses verified labels only.

## 5. Image validity masks in current training

| Information | Current use |
| --- | --- |
| Pixel/voxel support mask | Generated during preprocessing and available with cached data, but not explicitly used by the ResNet encoder to exclude invalid regions from features. |
| Real slice count | Keeps sampling within acquired slices and excludes padded slices. A slice mask is not passed into the model. |
| Series-padding mask | Not required currently because actual series are processed without empty series slots. |
| Label mask | Excludes unknown targets from loss and direct gradients. |

Invalid or padded regions inside a selected real slice can still influence the encoder. Filling pixels with zero does not instruct a network to ignore them. These masks describe validity, not lesion outlines; real dark pixels and background are not automatically invalid.

Padding is common and is not inherently a training error. Its importance depends on its extent, patterns, and consistency between training and inference. Inspect and measure it before changing the model; we have not established that it harms current results.

## 6. Co-occurrence of conditions

The radiologist suggested that effusion is more likely alongside ACL tear, MCL tear, or contusion.

The original EDA, based on 58 observed-label studies, showed the following correlations with effusion:

| Condition | Correlation with effusion |
| --- | ---: |
| ACL tear | 0.25 |
| Contusion | 0.12 |
| MCL tear | -0.04 |
| Synovitis | 0.40 |

Correlation is not a conditional probability. Effusion was already present in 35/58 studies (60.3%).

The later 20-study radiologist export showed:

| Finding | Effusion when present | Effusion when absent |
| --- | ---: | ---: |
| ACL tear | 4/4 (100%) | 7/16 (44%) |
| Contusion | 6/6 (100%) | 5/14 (36%) |
| MCL tear | 1/1 (100%) | 10/19 (53%) |
| Any of those three findings | 7/7 (100%) | 4/13 (31%) |

Effusion occurred in 11/20 studies overall. The injury groups overlap, so their counts must not be added together. This batch supports the observation but is too small to establish a dependable rule, especially with only one MCL-positive case. Effusion also occurs without these injuries.

Sources: [EDA notebook](sep-5-exploratory-data-analysis.ipynb), section 4, and the radiologist export at `/Users/ganapathychidambaram/Desktop/excluded_labels.csv`.

Co-occurrence could be tested through learned prediction refinement using initial predictions and image features. Avoid fixed probability boosts. Count only jointly known labels when measuring associations, distinguish verified from community labels, and keep validation/test labels out of fitting the refinement mechanism.

## 7. Roles of preprocessing and label files

| File | Role |
| --- | --- |
| `train_series.csv` | Identifies study/series relationships and supplies anatomical plane, fat suppression, and fluid sensitivity. Metadata is carried into processed manifests. |
| `qwen_training_labels.csv` | Helped select studies with usable generated labels, alongside officially labelled studies, when building the original 3,917-study image subset. |
| `labels_v1_blend.parquet` | Supplies current community training targets, while official labels are preserved. |

Qwen labels retain an indirect effect through which studies were included, but their predictions are no longer the current training targets. Keeping metadata in manifests does not mean the current ResNet explicitly consumes it as model features; that is part of the proposed metadata-aware fusion experiment.

## 8. Image framing and cropping

The model receives 224 × 224 images. Resizing the full scan preserves coverage, but surrounding space can leave fewer pixels for knee detail. A crop could enlarge anatomy while risking loss of useful structures, including regions containing Baker’s cysts or effusion.

Coverage changes naturally through a series: peripheral slices contain less anatomy, and middle slices show the knee prominently. This alone does not indicate poor framing.

Next steps:

1. Inspect actual model inputs after preprocessing across beginning, middle, and end slices and all three planes.
2. If cropping is justified, test one consistent physical crop across each series, checked with the radiologist. Avoid independently zooming each slice, which changes scale between neighbouring slices and can magnify noise.
3. Compare full-field and cropped inputs under otherwise identical training settings, including per-condition validation results.
4. Use the same adopted crop procedure during training and inference.

Central-emphasis sampling already prioritises middle slices while retaining peripheral coverage. Cropping addresses the separate question of whether unnecessary surrounding space can be removed safely.

## 9. Higher-resolution inputs

Current inputs are 224 × 224, resized from the 320 × 320 cache. After checking framing, test 288 × 288 to preserve more detail already present in that cache.

Higher resolution requires more GPU memory and computation; accuracy may not improve. Enlarging cached images cannot recover detail lost during preprocessing. Recovering that detail may require rebuilding the cache from original DICOM images.

Compare 224 and 288 under otherwise identical settings and match training and inference preparation.

## 10. Supervision and loss weighting

The current per-study loss is:

$$
L = \frac{\sum_c \mathrm{BCE}_c\,m_c\,w_c}{\sum_c m_c\,w_c}
$$

Here, the mask excludes unknown labels and the denominator sums active label weights.

If every active label in a study has community weight 0.25, that factor cancels between numerator and denominator. A community-only study therefore does not have quarter-strength influence. For example, an average error of 0.6 remains a loss of 0.6, rather than 0.15.

When official and community labels coexist within a study, weights of 1 versus 0.25 give official labels greater relative influence. To give a community-only study quarter-strength influence, explicitly multiply its final normalized study loss by 0.25.

Audit masking and normalization to confirm the intended behaviour. Test explicit study-level weighting as a separate experiment.

## 11. Label smoothing

Label smoothing makes community training answers slightly less absolute. For binary targets, epsilon 0.05 changes 100% present to 95%, and 0% present to 5%, allowing a little uncertainty because community labels may contain mistakes. Official labels remain unchanged.

For a general community target probability, the current transformation is:

$$
p_{\mathrm{smooth}} = p(1-2\epsilon)+\epsilon
$$

Compare epsilon 0.05 with epsilon 0.0 (no smoothing), holding all other settings fixed, to see which performs better.

## 12. MRNet, MPFuseNet, ELNet, and our ResNet34 baseline

Added October 20, 2026. MRNet, MPFuseNet, and ELNet are earlier knee MRI deep-learning systems used as comparison baselines in the CoPAS Nature paper. The suffixes 24, 26, and 27 in that paper are citation numbers, not model versions. The authors adapted the systems to their own 12-condition dataset.

ResNet34 is our image encoder; MRNet and MPFuseNet describe broader systems for processing slices and combining views. Their fusion ideas can be explored while retaining a ResNet34 encoder.

| Approach | Main idea | Comparison with our current baseline |
| --- | --- | --- |
| Our ResNet34 system | Pretrained ResNet34 encodes slice triplets; within-series mean/max pooling produces series vectors; series vectors are averaged before predicting 12 conditions. | Shares one encoder across series and uses a simple series-averaging rule. |
| MRNet | The original system uses pretrained AlexNet slice features, max pooling across slices, and logistic regression to combine plane-specific predictions. | Shares the slice-encoding/pooling principle, but keeps planes separate until prediction fusion. |
| MPFuseNet | Uses a pretrained CNN with spatial attention and fusion across MRI planes. | Adds mechanisms to emphasise useful image regions and combine views. These ideas could be incorporated around a ResNet encoder. |
| ELNet | A purpose-built, lightweight architecture trained from scratch; the authors report a model size below 1 MB. | More directly an alternative image-processing architecture, with an emphasis on efficiency. |

We started with ResNet34 as a practical baseline: pretrained image features and a manageable foundation for variable series counts, 12 targets, and masked missing labels. We have not tested these three alternatives on our competition data, so we have not established that they are worse or that ResNet34 is the best choice.

Published scores cannot directly rank these systems against our 0.851 public ensemble score because datasets, target definitions, and evaluation procedures differ.

A controlled next experiment is to retain ResNet34 and replace series averaging with plane/sequence-aware fusion. This tests the value of combining views differently while keeping the encoder fixed. Alternative encoders can then be compared separately.

Sources:

- [MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance imaging](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)
- [MPFuseNet: Optimising Knee Injury Detection with Spatial Attention and Validating Localisation Ability](https://arxiv.org/abs/2108.08136)
- [ELNet: Knee Injury Detection using MRI with Efficiently-Layered Network](https://arxiv.org/abs/2005.02706)
