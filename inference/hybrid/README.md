# Hybrid inference: ResNet34 + DINOv2

Upload **submit-knee-hybrid.ipynb** to Kaggle. It embeds all required Python code; the other files in this folder support local maintenance and testing.

Attach:

1. RSNA Knee Abnormality Detection competition data.
2. `gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil` — the version used for the successful ResNet submission.
3. `gany24558/gc-rsna-knee-dinov2/pyTorch/five-fold-attention` — the version used for the successful DINOv2 submission.
4. `gany24558/rsna-dicom-decoders` with the Python 3.12 Linux GDCM wheel.

Enable GPU, disable Internet, start a fresh session, and Run All. The notebook defaults to:

```python
WEIGHTS = {'resnet34': 0.80, 'dinov2': 0.20}
```

The same weights apply to all twelve findings. Change them in the configuration cell; they must be nonnegative and sum to one. Zero skips a model. The defaults are an experiment, not optimized weights, and may score below ResNet alone.

## Algorithm

Run the original ResNet34 inference in a child process, preserving its exported preprocessing and prediction head. After it exits and releases GPU memory, run the original DINOv2 encoder and five attention heads in another child process. DINOv2 internally averages its five sigmoid outputs. Match the resulting model probabilities by study ID and label name, then compute `0.8 * ResNet34 + 0.2 * DINOv2`. No extra sigmoid or binary threshold is applied.

Both component runtime files are byte-for-byte snapshots of the existing local inference runtimes, which match their respective local notebooks. This does not verify that a later Kaggle website edit is identical: attach the intended successful model versions and test the example run.

Component preprocessing is intentionally independent. Decoder installation happens before runtime imports and is passed to both processes. Existing package checksum and saved-weight smoke tests are retained. Missing or ambiguous model inputs fail clearly.

An unusable study retains each component's original 0.5 fallback and configured ensemble weight. There is no unvalidated dynamic switching to the other model. Component audits expose fallback and skipped-series counts. Decoder/model errors stop the submission. A failed new attempt archives any old top-level CSV, preventing it from being mistaken for the new result.

## Outputs

- `/kaggle/working/submission.csv`: the only file to select for the hybrid competition submission.
- `/kaggle/working/hybrid_run_manifest.json`: status, weights, exact mounted packages, component receipts, timing, checksum.
- `/kaggle/working/resnet34/`: original CNN predictions and diagnostic files.
- `/kaggle/working/dinov2/`: original transformer predictions and diagnostic files.

The combined budget defaults to 8.5 hours, starting at the configuration cell. A subprocess timeout prevents either model consuming a separate full budget. Total time includes both passes over the scans; full scoring runtime and model accuracy are unverified locally. The three example studies cannot establish full hidden-test runtime. Keep the successful standalone ResNet submission as your baseline.

No report, training label, API key, model upload, network download, or automatic competition submission is used. Save a completed notebook version with Internet off and select the **top-level** `submission.csv`.

## Files and checks

- `build_notebook.py`: regenerates the self-contained notebook using nbformat.
- `hybrid_runtime.py`: orchestration, validation and weighted probability blend.
- `resnet_inference.py`, `dinov2_inference.py`: unchanged component runtimes.
- `install_decoders.py`: existing offline wheel setup.
- `test_hybrid.py`: unit tests with synthetic probabilities and mocked model processes.

```sh
python -m unittest discover -s . -p test_hybrid.py -v
python build_notebook.py
```

Local tests cover row/column reordering, exact coverage, duplicate/missing IDs, nonfinite/out-of-range probabilities, invalid weights, zero-weight behavior, sequential orchestration and stale-output failure handling. Notebook schema validation and code compilation pass. Actual model inference still requires the attached packages and Kaggle GPU.

## Terms

**Ensemble:** combines predictions from multiple models. **Weight:** a model's fraction of the final probability. **Inference:** applying trained weights to new images. **Attention head:** the DINOv2 model's trained aggregation/classification component. **Fallback:** an emergency probability when usable scans are absent. **Manifest:** machine-readable model/run details. **Sequential:** one model runs after the other; this saves memory but not total compute.
