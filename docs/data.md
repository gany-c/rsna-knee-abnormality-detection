# RSNA Knee Abnormality Detection — Dataset Description

This dataset contains knee MRI studies annotated for twelve common findings: ligament and meniscus injuries, three compartments of osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture. Each study comprises a collection of individual MRI sequences from a single scanning session formatted as DICOM series. Your task is to predict the per-study probability of each of the twelve findings.

Studies come from a diverse international mix of imaging sites and span a wide range of scanners, protocols, and populations. Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies.

## Files

### `train.csv`

One row per training study.

| Column | Description |
|---|---|
| `StudyInstanceUID` | unique identifier for the study; matches the folder name under train_series/. |
| `PatientSex` | patient sex (Male or Female; may be blank). |
| `Report` | the free-text radiology report. May be in any of several languages, depending on the reporting institution. |

Twelve binary labels:

| Label | Finding |
|---|---|
| `ACL` | anterior cruciate ligament injury (0/1). |
| `MCL` | medial collateral ligament injury (0/1). |
| `Medial Meniscus` | medial meniscus tear (0/1). |
| `Lateral Meniscus` | lateral meniscus tear (0/1). |
| `Medial OA` | osteoarthritis of the medial tibiofemoral compartment (0/1). |
| `Lateral OA` | osteoarthritis of the lateral tibiofemoral compartment (0/1). |
| `PF OA` | patellofemoral osteoarthritis (0/1). |
| `Effusion` | joint effusion / excess fluid (0/1). |
| `Synovitis` | inflammation of the joint lining (0/1). |
| `Baker's` | Baker's cyst (0/1). |
| `Contusion` | bone contusion / bone bruise (0/1). |
| `Fracture` | fracture (0/1). |

### `train_series.csv`

One row per training series. Each series is a single MRI acquisition and each study comprises several series.

| Column | Description |
|---|---|
| `StudyInstanceUID` | study this series belongs to. |
| `SeriesInstanceUID` | unique identifier for the series; matches the folder name under `train_series/<StudyInstanceUID>/`. |
| `Fluid_Sensitive` | 1 if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), 0 otherwise. |
| `Fat_Suppression` | 1 if the sequence applies fat suppression, 0 otherwise. |
| `Anatomical_Plane` | imaging plane: Sagittal, Coronal, or Axial. |

### `train_series/`

Training DICOMs are organized as:

```text
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```
 Each .dcm is a single image slice. Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred.

### `test.csv`

Example test file with three study IDs from the public test set. During scoring, this example data will be replaced with the actual test data. There are about 1300 studies in the test set.

- `StudyInstanceUID`: unique identifier for a test study.
### `test_series.csv`

Same schema as train_series.csv, for the example test studies. Replaced with the real test-series descriptors during scoring.

### `test_series/`

Example test DICOMs, same layout as train_series/. Replaced with the real test DICOMs during scoring.

### `sample_submission.csv`

A valid submission with all label columns set to 0.5.

## Dataset Distribution Notice

Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets.

## DICOM Notes

Intensities, orientations, and resolutions vary across series and studies. Series come in a mix of transfer syntaxes (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian). Every DICOM has been stripped to an allowlisted set of 86 metadata tags.

## Dataset Summary

| Property | Value from the saved dataset page |
|---|---|
| Files | 819,640 (approximately 820k) |
| Size | 569.76 GB |
| Types | DICOM (`.dcm`) and CSV (`.csv`) |
| Columns reported by the data explorer | 38 |
| License | Subject to Competition Rules |
| Sample submission | 470 B; 3 example studies and 13 columns |

The sample submission has one `StudyInstanceUID` column and twelve finding columns, each initially set to **0.5**. The copied preview displayed only 10 of the 13 columns.

Example study IDs shown in that preview:

```text
1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191
1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776
1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402
```

## Directory Layout

```text
rsna-knee-abnormality-detection/
├── train.csv
├── train_series.csv
├── train_series/
│   └── <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
├── test.csv
├── test_series.csv
├── test_series/
│   └── <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
└── sample_submission.csv
```
