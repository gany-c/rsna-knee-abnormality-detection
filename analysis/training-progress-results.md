# Training Progress Results

## 1. First pass, self generated labels + competition labels

We used an LLM to generate labels from the radiology reports and combined them with the official competition labels. We normalized and used images only for studies with usable labels, giving **3,917 studies**. Images from studies without usable labels were not used. Hybrid predictions used probability averaging.

| Date (IST) | Model / Submission | Public score |
|---|---|---:|
| Sep 25 (approx.; 11d ago from Oct 6) | DINOv2 V5 | 0.607 |
| Sep 26 (approx.; 10d ago) | ResNet34 V3 | 0.679 |
| Sep 26 (approx.; 10d ago) | Hybrid V2: 80% ResNet + 20% DINOv2 | **0.688** |
| Sep 27 (approx.; 9d ago) | Hybrid V4: 70% ResNet + 30% DINOv2 | 0.685 |
| Sep 27 (approx.; 9d ago) | Hybrid V6: blend unconfirmed | 0.687 |

## 2. Using community provided labels - DinoV2

Replaced our self-generated labels with community soft targets, smoothed using `p * 0.9 + 0.05`; official competition labels were preserved.

**Note:** Although community and competition labels now covered all **4,407 studies**, we still used only the **3,917 original studies**, because those were the images we had already normalized.

**Approximate submission date:** September 28, 2026 (8d ago from October 6).

**DINOv2 V7:** OOF AUROC **0.7356**; public score **0.756** (**+0.149** over baseline).

Configured target weights were 0.25 for community labels and 1.0 for official labels. In ResNet, normalization within each study cancels a uniform weight; this does not downweight a wholly community-labelled study by four times.

## 3. Community Labels: All 4,407 Studies - DinoV2

We reran normalization to create a new image set covering all **4,407 studies / 24,371 series / 13 shards**, retaining the same community and official label strategy.

**Approximate submission date:** September 30, 2026 (6d ago from October 6).

**DINOv2 V9:** OOF AUROC **0.7228**; public score **0.748** (**−0.008** versus V7).

**Decision:** Expanding the dataset to **4,407 studies did not help in this experiment**. We are therefore not using the new image set going forward and will continue with the original **3,917-study image set**.

Both OOF scores used the same 58 officially labelled studies. The paired difference's 95% interval included zero, so the additional training data was not proven harmful. [Detailed comparison](outputs/comparison-report.md).

## 4. Oct 2-4, Community dataset, 3917 Studies - Resnet

**Misguided restart, not a continuation:** after the first run stopped, training was unintentionally rerun from scratch with `RESUME_CHECKPOINT=None`. This repeated epochs 1–10 instead of resuming from the saved stopping point, consuming another training session. It was not a new experiment or evidence that further training made the model worse. **Both runs were incomplete:** each achieved its best validation loss at its last completed epoch and had not reached early stopping.

Both used **3,133 training studies**, a 784-study holdout, and **9 officially labelled validation studies in a single split**. Each session had a configured 7.5-hour training budget.

| Metric | Earlier run (reviewed Oct 2) | Repeat run (started Oct 3 night; reviewed Oct 4) |
|---|---:|---:|
| Best completed epoch | **11** | 10 |
| Validation loss ↓ | **0.528455** | 0.536805 |
| Validation macro AUROC ↑ | **0.838228** | 0.803042 |
| Time budget reached during epoch | 12 | 11 |
| Export/reload and upload | Passed / accepted | Passed / accepted |

### October 4 — Submission Results

ResNet and hybrid inference used existing checkpoints, before resumed training produced a new model. DINOv2 rows are earlier results included for comparison.

| Submission | Configuration | Public score |
|---|---|---:|
| [ResNet V6](https://www.kaggle.com/code/gany24558/submit-knee-resnet34-2?scriptVersionId=355041495) | Community labels; epoch 11 | **0.831** |
| [Hybrid V10](https://www.kaggle.com/code/gany24558/submit-knee-hybrid?scriptVersionId=355067685) | 80% improved ResNet + 20% DINOv2 V7 | 0.830 |
| [DINOv2 V7](https://www.kaggle.com/code/gany24558/submit-knee-dinov2-attention?scriptVersionId=353624830) | 3,917-study subset | 0.756 |
| [DINOv2 V9](https://www.kaggle.com/code/gany24558/submit-knee-dinov2-attention?scriptVersionId=354213399) | 4,407-study dataset | 0.748 |

ResNet improved **+0.152** over its original baseline. The hybrid didnot get a better score over the best stand alone resnet

## 5. October 4, 5 - Full training of resnet

The mistake was then corrected: the repeated run's **epoch-10 `last.pt`** was attached as input, `RESUME_TRAINING=True` was set, and the next session genuinely continued at epoch 11. The earlier epoch-11 model was used for the initial 0.831 inference result.

Both initial training runs stopped at our configured **7.5-hour time budget**. A separate Kaggle GPU-quota restriction temporarily blocked inference; it did not cause those training stops.

| Experiment | Exported epoch | Val loss | Val macro AUROC | Submission | Public score |
|---|---:|---:|---:|---|---:|
| Earlier four-triplet model | 11 | 0.528455 | 0.838228 | ResNet V6 | 0.831 |
| **Resumed four-triplet model** | **15** | **0.503081** | 0.807209 | [**ResNet V9**](https://www.kaggle.com/code/gany24558/submit-knee-resnet34-2?scriptVersionId=355315473) | **0.839** |

- **Monday, Oct 5:** the run resumed on Oct 4 completed through **epoch 18**, then early-stopped and exported **epoch 15**. Thus “15 epochs total” in the journal refers to the selected checkpoint, not the final completed epoch. Inference achieved **0.839**. Checkpoint selection uses minimum validation loss, not maximum AUROC.

## 6. Oct 6 - Resnet 8 triplet experiment

| Experiment | Exported epoch | Val loss | Val macro AUROC | Submission | Public score |
|---|---:|---:|---:|---|---:|
| Eight-triplet model | 7 | 0.522174 | 0.821958 | [ResNet V14](https://www.kaggle.com/code/gany24558/submit-knee-resnet34-2?scriptVersionId=355642800) | 0.831 |

- **Oct 5:** started the fresh eight-triplet experiment using the same normalized images. The change was **four → eight triplets per series**, each still containing **three adjacent slices**; it was not three → eight individual slices.
- **Oct 6:** reviewed the eight-triplet run after nine completed epochs, resumed for epoch 10, and reached early stopping with epoch 7 still best. Its inference scored **0.831**. No renormalization was needed.
- **Model packages used:** four-triplet `study-mil/4`; eight-triplet `comm-dataset-slice-8/2`. Eight-triplet model V1 and V2 retained the same best epoch-7 weights. Model versions differ from submission versions.

## 7. October 13 — ResNet34 Ensemble Submission Result

| Submission | Version | Configuration | Status | Public score |
|---|---:|---|---|---:|
| submit-knee-resnet34-ensemble-0-2 | 3 | 50% member 0 + 50% member 2 probabilities; four triplets per series | Succeeded | **0.851** |

## 8. October 14–15, 2026 — Radiologist Labelling and Dataset

- **October 14:** Completed the remaining **12 studies** with radiologist **Dr. Amit Yadav**, bringing the total to **20 labelled studies**. Created the [radiologist-verified 20-study dataset](https://www.kaggle.com/datasets/gany24558/rsna-knee-radiologist-verified-20/data).
- **October 15:** Fixed the dataset after finding that it had not been created correctly.

## 9. October 14–16, 2026 — GPU Quota Interruption

- **October 14:** Ran out of weekly GPU quota after the all-planes-central training/resumption, despite purchasing additional quota through Colab. Patience had been increased to 5 in configuration, but a hard-coded early-stopping check still used 3.
- **October 15:** Created the all-planes inference notebook, but exhausted quota prevented submission.
- **October 16:** Still out of quota, with **19 hours remaining until reset** according to the journal. Inference and further training were postponed. This was the reported roughly two-day interruption; exact exhaustion and reset times were not recorded.

## 10. October 17, 2026 — ResNet34 All-Planes-Central Submission Result

| Submission | Version | Configuration | Status | Public score |
|---|---:|---|---|---:|
| [submit-knee-resnet34-all-planes-central](https://www.kaggle.com/code/gany24558/submit-knee-resnet34-all-planes-central?scriptVersionId=359216319) | 3 | Single ResNet34; all usable series; four centrally weighted triplets per series; inward-shifted boundary triplets | Succeeded | **0.844** |

**October 17 — Further training:** Continued the all-planes-central model with early-stopping patience increased to 5 and then 7. Training ended at epoch 21 without improving the best validation loss of **0.495103 at epoch 14**. The epoch-14 checkpoint was retained; the continuation produced no improved model.


## 11. October 18–20, 2026 — Transferred-Label ResNet34 Experiment

Generated CoAtNet teacher soft labels for all **4,407 studies** and trained the same all-planes-central ResNet34 with corrected weighting: transferred targets have weight **0.25**, official targets **1**, and loss is divided by the number of active labels so pseudo-label weights do not cancel. Training used **4,369 studies**, including **40 official-gold studies**; validation held out **18 official-gold + 20 radiologist-labelled studies**, excluding their pseudo-labels from training.

After the first interactive session's full checkpoint was lost, fine-tuning started from its published epoch-4 weights with fresh optimizer and scheduler state. The selected fine-tune epoch 4 achieved combined validation **BCE 0.496086 / macro AUROC 0.838431**; later training resumed from a saved full checkpoint.

| Date (IST) | Submission | Version | Status | Public score |
|---|---|---:|---|---:|
| October 20, 2026 | [submit-knee-resnet34-transferred-labels](https://www.kaggle.com/code/gany24558/submit-knee-resnet34-transferred-labels?scriptVersionId=360274151) | 2 | Succeeded | **0.859** |

This is our highest recorded public score: **+0.015** over all-planes-central (**0.844**) and **+0.008** over the ResNet34 ensemble (**0.851**).
