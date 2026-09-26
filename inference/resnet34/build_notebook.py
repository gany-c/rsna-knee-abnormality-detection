from pathlib import Path
import nbformat as nb
BASE=Path(__file__).parent
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# RSNA knee MRI — ResNet34 inference and submission

Attach the **competition dataset** and your custom model **gany24558/gc-rsna-knee-resnet34 → PyTorch → study-mil**, selecting the uploaded version. Also attach the **rsna-dicom-decoders** offline wheel dataset. Enable a **GPU** and **disable Internet** before saving/submitting this notebook. No original ImageNet model, training labels, reports or normalized-training dataset is needed.

This loads the trained best-epoch CNN, reproduces the exported validation preparation, processes the scoring studies, and writes `/kaggle/working/submission.csv`. It does not retrain or submit automatically. The current model's local validation AUROC was about 0.6966; hidden-test performance and full runtime are not yet established.

Scoring replaces the three example studies with a different-sized test set. All IDs/order come from the supplied CSVs. The CNN does not use Fluid_Sensitive or Fat_Suppression; neither field is inferred from the other. No report or patient-sex field is required. No prevalence adjustment, thresholding or test-derived calibration is performed.
''')
code('''from pathlib import Path
INPUT_ROOT = Path('/kaggle/input')
MODEL_PACKAGE = None  # Set to the folder containing manifest.json only if discovery is ambiguous
COMPETITION_ROOT = None
OUTPUT_ROOT = Path('/kaggle/working')
REQUIRE_GPU = True
REQUIRE_STANDARD_DECODERS = True # Fail early if documented scoring formats are unsupported
MICROBATCH = 8
MAX_HOURS = 8.5
INFER_MISSING_PLANE = True # Only when missing/unknown; use unambiguous DICOM orientation
FALLBACK_SCORES = [0.5] * 12 # Explicit emergency scores when no usable series remain
WHEELHOUSE = None # Auto-discover attached rsna-dicom-decoders wheel
''')
md('## Offline decoder installation\nAttach the prepared rsna-dicom-decoders dataset containing the Python 3.12 Linux python-gdcm 3.2.6 wheel. This cell discovers it and installs locally with --no-index --no-deps before importing pydicom. It does not download anything or replace NumPy. Start a fresh session when adding the dependency to avoid cached decoder availability. WHEELHOUSE can select an explicit directory if multiple wheels are attached.')
code((BASE/"install_decoders.py").read_text())
md('''## Self-contained inference runtime
Checks model hashes before executing its saved modules. Loads weights with pretrained=False, so no download occurs. A synthetic saved-weight check and a multi-series comparison verify streaming pooling against the exported model's forward method.

Each series uses the original normalization and native geometry checks, then the exported deterministic adjacent-slice triplets and full-FOV resize. All usable series are evaluated, as during validation. Series are processed one at a time; CUDA image microbatches shrink on allocation failure. Training augmentation is disabled.
''')
code((BASE/'resnet_inference.py').read_text())
md('''## Locate inputs and predict
Missing/invalid patient-series data is logged and skipped. Missing anatomical-plane metadata can be recovered only from an unambiguous DICOM orientation. Unusable whole studies receive the declared fallback; all fallbacks and partial studies are counted. This policy preserves coverage but does not establish accuracy for partial studies. Dependency/model failures remain fatal.
''')
code('''MODEL_PACKAGE, COMPETITION_ROOT = resolve_inputs(INPUT_ROOT, MODEL_PACKAGE, COMPETITION_ROOT)
print('Custom model:', MODEL_HANDLE)
print('Attached package:', MODEL_PACKAGE)
print('Competition data:', COMPETITION_ROOT)
SUBMISSION, RECEIPT = run_submission(
    MODEL_PACKAGE, COMPETITION_ROOT, OUTPUT_ROOT,
    require_gpu=REQUIRE_GPU, microbatch=MICROBATCH, max_hours=MAX_HOURS,
    fallback_scores=FALLBACK_SCORES, infer_missing_plane=INFER_MISSING_PLANE,
    require_standard_decoders=REQUIRE_STANDARD_DECODERS,
)
display(SUBMISSION.head())
print({k: RECEIPT[k] for k in ['status','studies','usable_series','skipped_series','partial_studies','fallback_studies','seconds']})
''')
md('''## Submit the saved notebook version
1. Confirm `status: complete` and inspect the audit counts.
2. Save a version with **Internet disabled** and run all cells.
3. Select that completed version's **submission.csv** for submission. Kaggle reruns it on the hidden test studies.

Outputs:
- `submission.csv`: exact sample columns/order, one row per requested study, twelve finite probabilities.
- `series_audit.csv`: decoded/skipped series, warnings, transfer syntaxes and timings.
- `study_audit.csv`: complete/partial/fallback study statuses.
- `decoder_report.json`: available standard DICOM decoders.
- `run_manifest.json`: model fingerprint, runtime, completion/failure status and summary counts.
- `failure_traceback.txt`: created if an exception occurs; hidden-run output access may be restricted by Kaggle.

Keep output private. These diagnostics are for visible data and accessible runs, not hidden-data probing. Successful execution does not guarantee good predictive accuracy or zero fallbacks.

References: competition description/rules supplied in the project; [pydicom decoder documentation](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.get_decoder.html).
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'kaggle':{'isInternetEnabled':False,'accelerator':'gpu'}})
nb.validate(book)
for c in cells:
    if c.cell_type=='code':compile(c.source,'cnn_submission_cell','exec')
nb.write(book,BASE/'submit-knee-resnet34.ipynb')
print('Built self-contained CNN submission notebook:',len(cells),'cells')
