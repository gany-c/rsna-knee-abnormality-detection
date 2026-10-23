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

## 12. October 20–22, 2026 — ConvNeXt and Colab Experiment

1. **What changed in training**
   - **Backbone:** ImageNet-pretrained ConvNeXt-Tiny; fresh attention/classifier. Did not initialize from the transferred-label ResNet model.
   - **Sampling:** Fixed four triplets → **15% of retained slice depth, bounded to 4–12 triplets per series**.
   - **Central emphasis:** Targeted 70% of samples within the middle 30% of depth, subject to available slices and outer-coverage constraints.
   - **Fusion:** Condition-specific attention over six **plane × FS/non-FS groups**, using all usable series.
     - Inspired by the [CoPAS paper: Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities](https://www.nature.com/articles/s41467-024-51888-4) (Nature Communications, 2024). Our group-attention implementation draws on its plane/sequence distinctions; it is not a reproduction of CoPAS.
   - **Labels/split unchanged:** Transferred labels with corrected 0.25 weighting; **4,369 training studies**, **18 official-gold + 20 radiologist validation studies**.
   - Multiple changes were combined; this was **not an isolated backbone ablation**.

2. **Training became substantially slower on Kaggle**
   - ConvNeXt: approximately **3–3.8 hours per epoch**.
   - Transferred-label ResNet, same 4,369-study cohort: approximately **73–75 minutes**.
   - Earlier all-planes ResNet, 3,133 studies: approximately **50 minutes**; this comparison also includes a dataset-size difference.

3. **October 20 — Preserved Kaggle quota and moved to Colab**
   - **Useful decision by Ganapathy:** Stopped Kaggle training with **16–17 GPU minutes remaining**, preserving capacity for the visible-test runs needed before submission.
   - Purchased additional **Colab compute units**. These did not replenish the displayed Kaggle allowance, so training moved to native Colab, including an **A100** session.
   - Cost: dataset downloads and checkpoint-transfer setup time. Benefit: continued training without consuming the reserved Kaggle quota.
   - Verified full checkpoints on Drive after epochs, enabling continuation across interrupted sessions.

4. **Training result — reviewed October 22**
   - Completed **epoch 24**, then early-stopped; selected **epoch 17** by minimum validation BCE.
   - Selected model: **BCE 0.493520 / macro AUROC 0.865843**.
   - Published as **central-group-attention model version 3**. Its epoch-17 inference weights differ from the full epoch-24 resume checkpoint.

5. **Why inference became a risk**
   - Previous ResNet inference took roughly **2.5 hours**; ConvNeXt adds a heavier encoder and more sampled triplets.
   - Submission inference also preprocesses **raw DICOM volumes**, unlike training from normalized caches. Hidden-run bottlenecks were not profiled.
   - Selecting **T4 ×2** does not automatically parallelize our single-device inference code.
   - Competition limit: **9 hours**. Original code raised an exception at its **8.5-hour internal budget**, without exporting a completed CSV.
   - Original submission **Version 3 failed with “Notebook Threw Exception.”** The exact exception remains unconfirmed without the traceback.
     - **Timing distinction:** Ganapathy reported approximately **10.5 hours on the UI before failure**, versus **9.6 hours in the submission system’s internal check**. UI elapsed time and internal execution time are different measures; neither should be confused with our configured 8.5-hour budget.

6. **Lightweight inference contingency**
   - Notebook: **submit-knee-convnext-tiny-central-attention-light.ipynb**.
   - Same Colab-trained **epoch-17 weights**; sampling reduced to **6% of retained depth, bounded to 2–4 triplets per series**.
   - Kept resolution, central-sampling policy, all usable series and plane/FS attention. Raw DICOM preprocessing remains; speedup and accuracy are unverified.
   - Saves a complete CSV progressively, replacing initial **0.5 probabilities** as studies finish.
   - At **8.5 hours**, exits cleanly with **partial_timeout**; unfinished studies retain fallbacks, with coverage recorded separately.
   - Nominal **30-minute margin** covers setup/finalization. Checks occur between series; a long decoder call, hard timeout or unrelated exception can still fail the run.
   - Syntax and simulated completion/timeout tests passed. **Lightweight Version 2 was still running at this update.**

**Status as of October 22:** No ConvNeXt public score yet. Transferred-label ResNet remains our best recorded submission at **0.859**.

## 13. October 23, 2026 — Competition Complete

- **The competition is over.** The submissions page reported two selected submissions for final evaluation.
- **Lightweight ConvNeXt:** [Version 2](https://www.kaggle.com/code/gany24558/submit-knee-convnext-tiny-central-attention-light?scriptVersionId=361120786) **succeeded**, with a displayed score of **0.817**.
  - The pasted results did not identify whether this was the **public or private score**; do not compare it directly with a different leaderboard split.
  - Coverage logs have not been reviewed. Success does not establish whether all studies received model predictions or some retained timeout fallbacks.
- **Original ConvNeXt:** Versions **2 and 3** failed with **“Notebook Threw Exception”** and produced no score.
- **Best recorded public score:** Transferred-label ResNet34 **0.859**. If ConvNeXt's 0.817 is public, it is **0.042 lower**.
  - The result does not isolate backbone quality: lightweight inference changed slice sampling, and fallback coverage remains unconfirmed.
- **Final private score and leaderboard rank:** Not yet recorded in this document.
