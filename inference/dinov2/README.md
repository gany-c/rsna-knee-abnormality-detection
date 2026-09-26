# Inference and competition submission

Upload `submit-knee-dinov2-attention.ipynb` to a Kaggle competition notebook.
Attach the competition data and the private Kaggle Model
`gany24558/gc-rsna-knee-dinov2/pyTorch/five-fold-attention` (select an explicit version).
Enable a GPU and disable Internet. Run all cells, save a completed version, and submit
its `submission.csv`. No separate Meta encoder or training dataset is needed.

The model root is discovered from its mounted slug and variation. If discovery is
ambiguous, set `MODEL_PACKAGE` to the attached version's directory containing
`manifest.json`. Set `COMPETITION_ROOT` only if competition discovery is ambiguous.
The notebook never calls an upload/download API or installs packages online.

Outputs in `/kaggle/working`:
- `submission.csv`: required competition artifact; exact twelve target columns and sample ID order.
- `processing_audit.csv`: per-series timings, counts, and warnings.
- `run_manifest.json`: completion/failure receipt, model identity, output hash, device and runtime.

The package supplies the exact preprocessing/runtime source, shared frozen encoder,
five head checkpoints and fold-specific spacing scalers. All manifest hashes and
the exported synthetic head reference are checked before prediction. Image features
are rounded through float16, then aggregated in float32 exactly as in training.
The five sigmoid outputs are averaged equally. Recoverable series decoding/geometry errors are logged and excluded. Remaining
series are used; if none remain, twelve explicit 0.5 emergency scores preserve
coverage. These are not calibrated estimates. Missing decoder dependencies and
model errors remain fatal. Partial-study accuracy is not yet established.

Local validation: every notebook cell executed against synthetic DICOM studies,
a real randomly initialized DINOv2 architecture and five random heads. Checked
training-batch parity, microbatch variation, row ordering, invalid score/coverage
rejection, and corrupted-package rejection. This does not establish competition
accuracy, real-package compatibility in Kaggle, GPU parity, codec coverage, or the
full hidden-test runtime. The actual model's reported OOF macro AUROC was 0.5389.

Dependencies must be available in the Kaggle image: torch, numpy, pandas, scipy,
pydicom, scikit-learn, plus transformers for HF exports or timm for timm exports.
Compressed DICOM transfer syntaxes also require compatible pixel decoder plugins.
If Kaggle reports a missing decoder, attach offline wheels for that dependency;
do not enable Internet in the submission run.

Development files: `inference_runtime.py` is embedded in the notebook by
`build_notebook.py`; `inference_build/test_submission.py` in the development workspace is the local synthetic integration test
and uses the project's training runtime and synthetic preprocessing fixtures.

## Hidden-run failure investigation

Kaggle returned only `Notebook Threw Exception`; the root cause remains unknown.
The revised notebook handles corrupt DICOMs, irregular geometry and missing series,
and reports skipped series, partial studies and fallback studies in its receipt.
No test labels or leaderboard probes are used. No competition resubmission was made.

Run `audit-knee-training-scans.ipynb` on Kaggle with the competition data and custom
model attached. It checks 250 deterministically sampled training series across planes
by default; set MAX_SERIES=0 for all training series. It produces
`training_scan_audit.csv` and `training_scan_summary.json`. Review failures before
resubmitting. It does not assess predictive accuracy of partial studies.

Only synthetic knee DICOMs were available locally. Fault injection covers missing
series, malformed DICOMs, missing study metadata and irregular slice spacing;
healthy predictions are checked against the original training aggregation.
Real competition coverage, codec availability and hidden-test runtime remain unverified.

## ResNet34 submission

The separately trained CNN uses [its own submission notebook](../resnet34/submit-knee-resnet34.ipynb). See [ResNet34 setup and validation notes](../resnet34/README.md). Attach gc-rsna-knee-resnet34 / PyTorch / study-mil; the DINO model and training caches are not used.
