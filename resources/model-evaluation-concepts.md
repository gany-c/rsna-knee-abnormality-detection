# Model Training and Evaluation: A Simple Guide

This guide explains common machine-learning concepts for someone familiar with software development. The examples use images, but many of the ideas also apply to audio and text.

## 1. Training, validation and inference

**Training** is the process of adjusting a model using examples and their expected answers, called **labels**.

**Validation** checks the model on examples held out from training. It helps us choose settings and decide when to stop training.

**Inference** means using a trained model to make predictions on an input. Validation also involves inference, but with known answers so we can measure performance.

```text
Training examples + labels → adjust model → trained model
                                              ↓
New input ──────────────────────────────→ prediction
```

A separate **test set** is used for a final evaluation. If we repeatedly use its results to choose changes, it starts functioning as another validation set.

A useful software analogy is that training examples help build the implementation, while held-out examples test how well it generalises. Unlike ordinary software tests, a finite validation set cannot establish correctness on every possible input.

## 2. Weights and features

**Weights** are the numbers inside a model that training adjusts. Saving the weights preserves what the model learned.

**Features** are numerical descriptions the model produces for a particular input. For example, an image may become a vector containing hundreds of numbers describing useful visual patterns.

```text
Image + model weights → feature vector for that image
```

Weights belong to the model; features describe an input. Individual feature values usually do not have simple names such as “tear” or “bird wing.”

## 3. Encoder, backbone and prediction head

Many models can be understood as two parts:

```text
Input → encoder/backbone → features → prediction head → prediction
```

- **Encoder:** converts the input into features.
- **Backbone:** another common name for the main feature-extraction network, especially in image models.
- **Prediction head:** converts those features into the outputs needed for a particular task.

For example, the same image encoder might support a head that predicts animal species or a different head that predicts image quality. Each task needs suitable training.

A classifier is the part or complete model that predicts categories. A **multi-label classifier** can predict several categories at once rather than choosing exactly one.

## 4. Frozen encoders and fine-tuning

A **pretrained model** has already learned from an earlier dataset.

There are two common ways to adapt it:

| Approach | What changes during training? |
|---|---|
| Frozen encoder | Only the prediction head; encoder weights stay fixed |
| Fine-tuning | Some or all encoder weights, usually together with the head |

Freezing can reduce computation and the number of parameters we need to learn from a small dataset. Fine-tuning lets the encoder adapt its features to the new task, but usually costs more computation and can overfit.

Neither approach is always better.

## 5. Feature extraction and caching

When the encoder is frozen, we can calculate features once and save them:

```text
First pass:
Images → frozen encoder → saved features

Repeated training passes:
Saved features → trainable head → predictions → training loss
```

This is **feature caching**. It avoids running the expensive encoder during every training epoch.

Strictly speaking, the encoder weights are frozen. Cached features are saved outputs of that fixed encoder.

Reusing a cache requires compatible inputs, preprocessing, encoder weights and feature-extraction settings. Changing the labels alone does not change the image features, although it changes what the head must learn.

If we fine-tune the encoder, its features change as its weights change. A fixed cache would no longer represent the current encoder. Likewise, caching one fixed image representation limits the image-level augmentations we can use during head training.

At inference time, a new input still needs to pass through the encoder before reaching the trained head.

## 6. Epochs, batches, loss and checkpoints

- **Batch:** a group of examples processed together during training.
- **Epoch:** usually one pass through the training examples. Sampling-based pipelines may define an epoch differently.
- **Loss:** a numerical penalty describing how far predictions are from the training targets. Training attempts to reduce it.
- **Checkpoint:** saved model weights and, sometimes, the state needed to resume training.
- **Early stopping:** ending training after a chosen validation measure stops improving for a specified number of epochs.

The saved “best” checkpoint depends on the chosen measure. The epoch with the lowest validation loss need not have the highest AUROC.

Lower training loss does not guarantee better predictions on unseen examples.

## 7. Folds and cross-validation

A **fold** is a group of examples used in cross-validation.

In five-fold cross-validation, we divide the evaluation-eligible dataset into five groups and train five model variants:

| Model | Held-out validation group | Training groups |
|---|---|---|
| 0 | A | B, C, D, E |
| 1 | B | A, C, D, E |
| 2 | C | A, B, D, E |
| 3 | D | A, B, C, E |
| 4 | E | A, B, C, D |

Every example is evaluated by a model that did not train on it.

Some pipelines also use an auxiliary training dataset that is not part of the validation folds. That is a specific design choice, not a requirement of cross-validation. Auxiliary data must not leak information about held-out examples.

**Keep related examples together.** If one person contributes several scans, splitting those scans across training and validation can make performance look better than it really is. A patient-grouped split keeps all their scans in the same fold.

## 8. OOF: out-of-fold predictions

An **out-of-fold prediction** is a prediction made by the model that held that example out of training.

```text
Group A → predictions from model 0
Group B → predictions from model 1
Group C → predictions from model 2
Group D → predictions from model 3
Group E → predictions from model 4
                  ↓
Combine into one OOF prediction table
```

We compare this table with the known labels to evaluate the training approach.

Do not average all five models on these training examples and call the result OOF: four of the models ordinarily trained on each example.

OOF predictions reduce direct training/evaluation overlap. They do not eliminate all bias: repeatedly using OOF results to tune settings can still overfit the validation process.

**The number of training examples need not equal the number of OOF-evaluated examples.** A pipeline might train with many weakly labelled examples but calculate OOF metrics only on a smaller set with trusted labels.

## 9. AUROC: measuring ranking quality

A classifier may output a score interpreted as a probability, such as 0.8 for “condition present.”

**AUROC** means *area under the receiver operating characteristic curve*. A practical interpretation is:

> Choose one positive example and one negative example. How often does the model give the positive example a higher score?

Ties receive half credit.

| Example | Actual label | Predicted score |
|---|---|---:|
| A | Positive | 0.90 |
| B | Negative | 0.70 |
| C | Positive | 0.60 |
| D | Negative | 0.20 |

There are four positive–negative pairs. The ordering is correct for A–B, A–D and C–D, but wrong for C–B. AUROC is therefore **3/4 = 0.75**.

- **1.0:** perfect ordering on the evaluated examples.
- **0.5:** chance-level ordering.
- **Below 0.5:** worse-than-chance ordering.

An AUROC of 0.75 does **not** mean 75% of diagnoses are correct. AUROC does not require choosing a Yes/No threshold, and it does not measure whether a predicted 80% probability really corresponds to an 80% event rate. That latter property is called **calibration**.

AUROC is undefined when the evaluation data contains only positives or only negatives.

## 10. Macro AUROC

In a multi-label problem, calculate AUROC separately for each label, then average them equally:

```text
Macro AUROC = sum of per-label AUROCs / number of evaluable labels
```

For example, AUROCs of 0.80, 0.70 and 0.90 give a macro AUROC of 0.80.

“Macro” means every label has equal weight, regardless of how common it is. Any undefined label scores must be handled explicitly and reported; one common policy is to skip them and average the evaluable labels.

**OOF macro AUROC** therefore means: calculate each label's AUROC using held-out predictions, then average across labels.

Calculating AUROC on the combined OOF table is not generally the same as averaging the AUROCs calculated separately within each fold.

## 11. Ensembles and five-fold inference

An **ensemble** combines predictions from several models. A simple approach averages their probabilities:

```text
New input → model 0 → 0.70 ┐
          → model 1 → 0.80 ├→ mean probability = 0.75
          → model 2 → 0.75 ┘
```

For a genuinely unseen test example, all five fold models can contribute because none trained on that example. For OOF evaluation, use only the model that held the example out.

Ensembles can help when models make different errors, but more models do not guarantee an improvement.

**Ensemble weights** are different from the millions of learned weights inside a neural network. A blend such as 80% model A and 20% model B assigns two weights to their final predictions.

## 12. Five frozen heads versus five complete models

| Setup | What is trained? | What happens for a new input? |
|---|---|---|
| Shared frozen encoder + five heads | Five heads, each with its own fold split | Encode the input once, then run all five heads |
| Five fine-tuned image models | Five encoder/head combinations | Run each model's encoder and head, then combine predictions |

Both approaches support cross-validation and OOF evaluation. The frozen approach can share cached features; the fine-tuned approach generally cannot because each encoder learns different weights.

These choices are not tied to a particular architecture. A CNN can be frozen or fine-tuned, and so can a vision transformer.

## 13. Single-image versus multi-image predictions

A single-image classifier follows:

```text
One image → encoder → features → head → labels
```

A multi-image model can follow:

```text
Many images → shared encoder for each image → many feature vectors
           → aggregation → one summary → head → labels
```

**Average pooling** takes the average value of each feature across inputs. **Maximum pooling** takes the largest value of each feature. **Attention** learns how to combine information based on the inputs and task.

When a label applies to a collection rather than to each image individually, this is often called **multiple-instance learning**. A positive collection does not imply every image shows the finding.

A 2D CNN can process three neighbouring slices as three input channels, often called **2.5D**. A true **3D CNN** instead applies filters across height, width and depth.

## 14. Shards versus folds

| Term | Purpose |
|---|---|
| Shard | A storage or processing chunk, used to keep jobs and files manageable |
| Fold | A training/validation partition, used to evaluate generalisation |

Think of shards as storage boxes and folds as exam groups. One shard can contain examples from several folds. Changing storage boxes should not silently change which examples are held out for evaluation.

## 15. Missing labels and soft labels

- **Hard label:** a definite target such as 0 or 1.
- **Soft label:** a target between 0 and 1, such as 0.8, often supplied by another model or a labelling process.
- **Missing label:** no usable target is available.

Missing is not the same as negative. A **mask** tells the loss function which targets to include. A **weight** controls how much an included target contributes.

Binary cross-entropy can train on soft labels. Its logits-based form takes raw model outputs and handles the sigmoid calculation internally.

Soft labels are training signals, not automatically trustworthy ground truth for evaluation.

## 16. Comparing two model versions fairly

Use the same evaluation examples, reference labels and metric calculation. Otherwise, a score change might reflect a different evaluation population rather than a different model.

A **paired bootstrap** repeatedly samples evaluation units with replacement, keeping the two models' predictions paired for each unit. It estimates uncertainty in their score difference. If several studies belong to one patient, resampling by patient can account for that dependence.

A confidence interval spanning zero is inconclusive; it does not prove the models are equivalent. Bootstrap intervals on fixed predictions also do not capture variation from retraining the models, choosing different folds or selecting different checkpoints.

With only one training run per version, a decline does not establish that a particular change caused harm. Record the data, labels, split, configuration and checkpoint-selection rule so comparisons can be interpreted correctly.
