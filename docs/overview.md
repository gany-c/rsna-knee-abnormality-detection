# RSNA Knee Abnormality Detection — Overview

A single knee scan can reveal a dozen different problems. In this competition, you are tasked to build machine learning models that detect a defined set of clinically important abnormalities on knee MRI examinations.

## Description

The knee is the most commonly injured and imaged joint in the body. Osteoarthritis alone affects an estimated 654 million people worldwide, while acute knee injuries account for 15 to 40 percent of all sports-related trauma. MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without exposing patients to radiation.

Reading those scans isn’t always straightforward. ACL and MCL tears, meniscal damage, cartilage loss, fractures, and other abnormalities can be subtle, and radiologists don’t always interpret them the same way. Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.

In this competition, you will develop multimodal machine learning models to detect twelve clinically important knee abnormalities. You'll work with the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report, enabling your models to learn from both visual scans and written diagnostic text.

High-performing models can act as robust decision support tools, delivering the accuracy, consistency, and speed needed to elevate expert-level knee MRI interpretation and improve care across disparate clinic settings.

## Evaluation

Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets.

The final score is, in other words, the macro-averaged AUC ROC.

## Submission File

For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...
```

## Timeline

| Date | Milestone |
|---|---|
| July 30, 2026 | Start Date. |
| October 15, 2026 | Entry Deadline. You must accept the competition rules before this date in order to compete. |
| October 15, 2026 | Team Merger Deadline. This is the last day participants may join or merge teams. |
| October 22, 2026 | Final Submission Deadline. |
| November 5, 2026 | Winners' Requirement Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video and method description. |

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

### Main Leaderboard

| Prize | Amount |
|---|---:|
| First Prize | $9,000 |
| Second Prize | $7,000 |
| Third Prize | $6,500 |
| Fourth Prize | $6,000 |
| Fifth Prize | $5,500 |
| Sixth Prize | $5,000 |
| Seventh Prize | $5,000 |
| Eighth Prize | $5,000 |
| Ninth Prize | $5,000 |
| Tenth Prize | $5,000 |

### Efficiency Track

| Prize | Amount |
|---|---:|
| First Efficiency Prize | $7,000 |
| Second Efficiency Prize | $6,000 |
| Third Efficiency Prize | $5,000 |

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA) Annual Meeting, winners will be invited and strongly encouraged to attend the AI Challenge Recognition Event with waived fee, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the competition rules, in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, solution packaging/delivery, presentation to host), the host team also asks that you:

1. Create a short video presenting your approach and solution, and

2. Publish a link to your open sourced code and the weights on the competition forum

3. Share final version of model as publicly available for open distribution and validation. Please see [example model](https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/PyTorch/default) as an example.

## Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook ≤ 9 hours run-time
- GPU Notebook ≤ 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

## Efficiency Prize Evaluation

We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy.

For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance.

To be eligible for an Efficiency Prize, a submission:

- Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the My Submissions tab.
- Must be ranked on the Private Leaderboard higher than the sample_submission.csv benchmark.

All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize.

An Efficiency Prize will be awarded to eligible submissions according to how they are ranked by the following evaluation metric on the private test data. See the Prizes tab for the prize awarded to each rank. More details may be posted via discussion forum updates.

### Efficiency Score

The efficiency score combines predictive performance and evaluation runtime. **Lower is better.**

> The equation and variable symbols were missing from the saved page extract. They have not been reconstructed here.

The accompanying description refers to:

- The submission's score on the main competition metric.
- The score of the `sample_submission.csv` benchmark.
- The maximum main-metric score across submissions on the private leaderboard.
- The number of seconds needed to evaluate the submission.

During the training period of the competition, you may see a leaderboard for the public test data in the following notebook, updated daily: Efficiency Leaderboard. After the competition ends, we will update this leaderboard with efficiency scores on the private data. During the training period, this leaderboard will show only the rank of each team, but not the complete score.

## Citation

Po-Hao “Howard” Chen, Naveen Subhas, Robyn Ball, Pieter Baeyens, Errol Colak, Ali Emami, Hillary Garner, Jacob Kazam, Hui-Ming Lin, Luciano Prevedello, Daniel Schneider, Jason Sho, Ryan Holbrook, and María Cruz. RSNA Knee Abnormality Detection. [Kaggle competition](https://kaggle.com/competitions/rsna-knee-abnormality-detection), 2026. Kaggle.
