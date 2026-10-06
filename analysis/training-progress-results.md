# Training Progress Results

**Created:** October 1, 2026 · **Last updated:** October 6, 2026 (Asia/Kolkata)

*Submission timing is sourced from the Kaggle history supplied on October 6, 2026. Calendar dates inferred from its rounded ages are approximate. Training chronology comes from notebook logs and the journal. V-numbers are submission versions unless stated otherwise.*

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

## Stage Comparison

| Date (IST) | Labels / image subset | Studies | DINOv2 OOF | DINOv2 public | ResNet public | Best hybrid public |
|---|---|---:|---:|---:|---:|---:|
| Sep 25–28 (approx.) | Our labels + official | 3,917 | — | 0.607 | 0.679 | 0.688 |
| Sep 28–Oct 6 | Community + official | 3,917 | 0.7356 | 0.756 | **0.839** | 0.830 |
| Sep 30–Oct 1 | Community + official | 4,407 | 0.7228 | 0.748 | Untested | Untested |

## 6. Oct 6 - Resnet 8 triplet experiment

| Experiment | Exported epoch | Val loss | Val macro AUROC | Submission | Public score |
|---|---:|---:|---:|---|---:|
| Eight-triplet model | 7 | 0.522174 | 0.821958 | [ResNet V14](https://www.kaggle.com/code/gany24558/submit-knee-resnet34-2?scriptVersionId=355642800) | 0.831 |

- **Oct 5:** started the fresh eight-triplet experiment using the same normalized images. The change was **four → eight triplets per series**, each still containing **three adjacent slices**; it was not three → eight individual slices.
- **Oct 6:** reviewed the eight-triplet run after nine completed epochs, resumed for epoch 10, and reached early stopping with epoch 7 still best. Its inference scored **0.831**. No renormalization was needed.
- **Model packages used:** four-triplet `study-mil/4`; eight-triplet `comm-dataset-slice-8/2`. Eight-triplet model V1 and V2 retained the same best epoch-7 weights. Model versions differ from submission versions.
