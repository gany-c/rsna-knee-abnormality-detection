# ResNet34 inference and submission

Upload **submit-knee-resnet34.ipynb** to Kaggle. It is self-contained; do not upload the development Python files separately.

Attach:
1. RSNA Knee Abnormality Detection competition data.
2. `gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil`, choosing the uploaded version.

Enable a GPU and **disable Internet**. No DINO model, ImageNet download, training data cache, report, or label input is needed. The notebook loads the best epoch already saved in the custom CNN package. Model version, manifest hash and best epoch are recorded in the run receipt.

Run all cells, inspect the audit counts, and save a completed notebook version. Select that version's `submission.csv` for competition submission. The notebook itself never submits to Kaggle or changes the model.

## Decoder setup — required for the reported Kaggle environment

The supplied run failed at preflight because JPEG Lossless UIDs .57 and .70 were unavailable. It stopped before loading the CNN. The prepared `rsna-dicom-decoders.zip` contains python-gdcm 3.2.6 from PyPI, a CPython 3.12 Linux x86_64 wheel (glibc >=2.27), README and SHA256 checksum. No scans, credentials or NumPy wheels are included.

Create a private Kaggle dataset named `rsna-dicom-decoders` from the ZIP and attach it alongside competition data and the CNN model. The revised notebook discovers the wheel and installs it with `--no-index --no-deps --target /kaggle/working/_dicom_deps`, then imports GDCM before pydicom. Start a fresh session and Run All with Internet disabled. The configured strict decoder checks remain enabled.

Wheel archive integrity and metadata were verified locally. The Linux binary cannot be executed on this Mac; confirm `Unavailable standard DICOM decoders: none` in Kaggle before submitting. Local synthetic tests continue to use native/RLE files and bypass only the startup requirement for absent local JPEG plugins. Runtime encountering a missing decoder remains fatal.

Reference: [pydicom decoder API](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.get_decoder.html).

## Outputs

- submission.csv: the competition artifact, exactly one row per sample ID, exact sample column order, finite scores in [0,1].
- series_audit.csv: series success/failure, transfer syntax and geometry warnings.
- study_audit.csv: complete, partial and fallback studies.
- decoder_report.json: dependency availability.
- run_manifest.json: completion/failure, model identity, counts, runtime and output hash.
- failure_traceback.txt: diagnostic on a failed accessible run; Kaggle may not expose hidden-run logs.

Fallback is explicit: unusable series are skipped; studies with no usable series get 0.5 for each target. This prevents missing rows, but those are emergency scores rather than model predictions. Inspect the counts. Corrupt model files, missing codecs, nonfinite model outputs and exhausted runtime are not silently converted to fallback predictions.

## Validation and limitations

Every notebook cell passed local synthetic end-to-end tests using real ResNet34 forward passes and a package produced by the training exporter. Checked DICOM-to-exported-forward agreement, multi-series streaming pooling, microbatch variation, five requested studies rather than three, absent reports, unequal fluid/fat flags, reordered output columns, RLE decoding, missing/invalid series, corrupt files, irregular geometry, plane recovery, fallback counts, model hash failure and stale-submission protection.

This does not test the user's actual trained weights in Kaggle, CUDA numerical behavior, real JPEG Lossless/JPEG 2000 pixel files, broad real-scanner coverage, partial-study accuracy or the full hidden-test runtime. Those remain Kaggle checks. The notebook dynamically runs the actual package's saved-weight smoke test before inference.

The local development test harness is `cnn_inference_build/test_resnet_submission.py` in the workspace; it depends on synthetic fixtures there. `build_notebook.py` embeds resnet_inference.py into the deliverable notebook.
