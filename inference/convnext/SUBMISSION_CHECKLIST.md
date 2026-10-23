# ConvNeXt submission checklist — 22 October 2026

Notebook: `submit-knee-convnext-tiny-central-attention.ipynb` in this folder.
Do not run the training notebook for submission. Keep the remaining ~17 minutes of Kaggle GPU time for the final run.

## 1. Attach inputs with the accelerator OFF

- Upload this updated inference notebook.
- Attach the RSNA Knee Abnormality Detection competition.
- Attach exactly one version of `gany24558/gc-rsna-knee-convnext/pyTorch/central-group-attention`: the newly uploaded export whose manifest says `best_epoch=17` and `best_validation_loss=0.49352002143859863`. The completed training epoch was 24, but the exported weights are epoch 17.
- Attach the offline `rsna-dicom-decoders` wheel dataset. The installer expects `python_gdcm-3.2.6-*.whl`, compatible with the runtime's Python and Linux x86_64 tags. A macOS or different Python ABI wheel will not work. Python 3.13 requires a compatible cp313 wheel.
- No training caches, labels, ImageNet download, or student model are required.
- Disable Internet. Keep the accelerator OFF during the initial checks.

## 2. Run cells individually through CPU preflight

Do NOT select Run All yet: the final prediction cell requires a GPU.

1. Run the configuration cell. Keep `REQUIRE_GPU=True`, `MICROBATCH=8`, `MAX_HOURS=8.5`, and `REQUIRE_STANDARD_DECODERS=True`.
2. Run the offline decoder-install cell BEFORE the runtime-import cell. Expect `Offline GDCM installed`, or all required decoders must already be available. If it reports incompatible wheels or cached decoder availability, resolve it in a fresh CPU session before enabling GPU.
3. Run the runtime-definition cell. This imports dependencies and defines inference functions; it does not predict.
4. Run the new CPU preflight cell. Require `CPU PREFLIGHT PASSED`, epoch 17, correct BCE, positive test-study and series counts. Resolve any warning about studies with no metadata before proceeding.
5. If model auto-discovery is ambiguous, set `MODEL_PACKAGE` in the configuration cell to the exact attached folder containing `manifest.json`, then rerun from a fresh CPU session. Do not guess a model-version number or path.

Preflight checks model hashes, embedded Python syntax, expected model metadata, test/sample ID agreement, target order, binary Fat_Suppression metadata, and decoder availability. It does not load the weights or decode actual images.

Optional stronger CPU check after preflight (may take time, uses no GPU):

```python
cpu_bundle = load_package(MODEL_PACKAGE, torch.device('cpu'))
print('CPU model load and synthetic prediction checks passed')
del cpu_bundle
```

Run this in a temporary diagnostic cell and remove that cell before the final saved GPU run to avoid repeating a redundant model load.

## 3. Validate actual-image inference outside the scarce Kaggle GPU session

Use Colab with the same exported epoch-17 package, inference source, and a representative raw DICOM study arranged as `test_series/<study>/<series>/*.dcm`, with matching test/sample/series CSVs. A normalized NPZ cache alone does not test DICOM inference.

Check that a complete prediction succeeds, probabilities are finite and within [0,1], and output IDs/columns match the sample. Time preprocessing plus prediction. A Colab A100 timing is not a reliable bound for Kaggle T4 runtime. The local synthetic tests cannot establish that Kaggle's preliminary run fits 17 minutes or that hidden evaluation fits its limit.

## 4. Final GPU run

- Confirm the correct model version and decoder dataset remain attached in the version being saved.
- Enable the permitted GPU accelerator. Use a fresh session so decoder installation precedes imports.
- Keep Internet disabled and save a version that executes all cells once. Avoid an extra interactive GPU Run All followed by another full saved run.
- Confirm logs show `CPU PREFLIGHT PASSED`, `Loaded best CNN epoch 17; saved-weight and streaming checks passed`, and study progress.
- `MAX_HOURS=8.5` is the inference safety ceiling, NOT a 17-minute quota guard. Do not reduce it to 17 minutes: the same notebook runs over the larger hidden test set.
- If a preflight error occurs, stop the GPU session promptly and investigate on CPU. Do not repeatedly retry with GPU enabled.

## 5. Before selecting the saved version for submission

- Run completed successfully; no traceback.
- `/kaggle/working/submission.csv` exists.
- `run_manifest.json`: `status=complete`, `best_epoch=17`, expected study count.
- Review `fallback_studies`, `partial_studies`, and `skipped_series`. Investigate any unexpected nonzero value using `series_audit.csv` and `study_audit.csv`; an all-fallback CSV is not evidence that the model worked.
- Notebook validates exact sample row order/IDs/columns and finite probabilities in [0,1] after saving/reloading the CSV.
- Select this successful saved notebook version and its `submission.csv` for scoring. Confirm the submitted version uses the intended model inputs.

If a run fails, inspect `failure_traceback.txt` and `run_manifest.json` if produced. Failures before `run_submission` begins (imports, decoder setup, CPU preflight) appear only in notebook output.
