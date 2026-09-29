# Single-Image CNN vs. Multi-Image MRI Model

The 2D CNN backbone can be the same in both setups. The difference is how the inputs are assembled, how their features are combined, and which input the label describes.

## 1. Conceptual Shift: Single Input vs. Multiple-Instance Learning

**Single-image CNN (BirdCLEF example)**

```text
1 spectrogram → 2D CNN → features → classifier → bird species probabilities
```

- **Input:** One spectrogram or other image-like tensor.
- **Label:** Describes that input. The prediction can still be multi-label if several birds are audible in the clip.

**Multi-image MRI model (RSNA knee example)**

```text
Slices 1, 2, …, N
↓ shared 2D CNN applied to each slice or slice group
Features 1, 2, …, N
↓ pooling within each series
One summary per series
↓ fusion across series
Study vector
↓ classifier
12 abnormality probabilities
```

- **Input:** A study containing multiple series, each with multiple slices.
- **Label:** Describes the whole study. It does not say which slice shows the abnormality.
- **Learning setup:** This is a form of multiple-instance learning (MIL): the model learns from a labeled collection of images without an individual label for every image.

## 2. Detailed 2.5D Processing Flow

**Series 1 — for example, sagittal**

```text
3 adjacent slices → shared ResNet-34 → feature vector 1
Next selected 3-slice group → shared ResNet-34 → feature vector 2
Next selected 3-slice group → shared ResNet-34 → feature vector 3
…
All feature vectors → average/max pooling → series 1 summary
```

**Series 2 — for example, coronal**

```text
3-slice groups → same shared ResNet-34 → feature vectors
Feature vectors → average/max pooling → series 2 summary
```

**Series 3 — for example, axial**

```text
3-slice groups → same shared ResNet-34 → feature vectors
Feature vectors → average/max pooling → series 3 summary
```

**Final study prediction**

```text
Series 1 summary + series 2 summary + series 3 summary
↓ study-level fusion, such as mean pooling
Study vector
↓ classifier
12 abnormality probabilities
```

### What happens at each stage

1. **Build 2.5D inputs.** Stack three adjacent MRI slices as the three channels of one input. This gives a standard 2D CNN limited information about neighboring slices.
2. **Extract features.** Run every three-slice group through the same CNN backbone. “Shared” means the groups use the same learned weights.
3. **Summarize each series.** Average pooling captures features spread across a scan. Max pooling can retain a strong feature that appears in only a few slice groups.
4. **Fuse the series.** Combine the summaries from different series and views into one study vector. Mean pooling is a simple option; learned fusion is another.
5. **Predict and train.** The classifier produces 12 study-level probabilities. The loss is computed from the available study labels, and its gradients flow back through both aggregation stages into the shared CNN. In implementation, the classifier outputs logits; training uses a logits-based loss, and sigmoid converts logits to probabilities for prediction.

### Details specific to our ResNet implementation

- “Adjacent” means neighboring slices in the preprocessed cache. When an original series has more than 64 slices, preprocessing samples it down to 64, so cache neighbours may not be consecutive original slices.
- Slice groups are centered on selected positions and can overlap. At a stack boundary, the edge slice is repeated as needed.
- Average and maximum feature vectors are concatenated to form each series summary. Series summaries are then averaged to form the study vector.
- A study can contain more than three series, including multiple acquisitions in the same plane. The three series above are illustrative.
- This is a 2D CNN with 2.5D inputs, not a true 3D CNN that applies filters through the volume's height, width and depth.

## 3. Quick Comparison

**Single-image CNN**

- Input: one image-like tensor per example, often `B × 3 × H × W`
- CNN: extracts features from that input
- Aggregation: pools spatial features within the image
- Label: applies to the individual input

**Multi-image MRI model**

- Input: multiple series and slice groups per study
- CNN: the same backbone processes each group
- Aggregation: combines groups into series summaries, then series into a study summary
- Label: applies to the whole study

Here, `B` is batch size, `3` is the channel count, and `H` and `W` are image height and width. Three channels are a common model-input convention, not a requirement for every spectrogram model.

**Key idea:** The MRI model makes one study-level decision from many images, even though it is not told exactly which slices contain each abnormality.

**Missing-label note:** If a study has unknown values among the 12 targets, exclude those targets from its training loss. A blank label does not mean “no abnormality.”
