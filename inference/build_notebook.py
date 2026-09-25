from pathlib import Path
import nbformat as nb
BASE=Path(__file__).parent
cells=[]
def md(text): cells.append(nb.v4.new_markdown_cell(text))
def code(text): cells.append(nb.v4.new_code_cell(text))
md('''# RSNA Knee MRI — inference and submission

Uses **gany24558/gc-rsna-knee-dinov2 → PyTorch → five-fold-attention**.

Before running on Kaggle:
1. Add the **RSNA Knee Abnormalities Detection competition data**.
2. Add your custom model through **Add Input → Models**, choosing a specific version. The attached version stays fixed for the saved notebook run.
3. Enable a GPU and **disable Internet**. No downloads, API tokens, training labels, or separate Meta model are needed.
4. Run all cells. Save a completed notebook version and select `submission.csv` for competition submission.

The notebook creates `/kaggle/working/submission.csv`, `processing_audit.csv`, and `run_manifest.json`. Only `submission.csv` is the competition prediction artifact. The notebook does not automatically submit to Kaggle or upload a model. Keep outputs private.

The current model achieved validation macro AUROC 0.5389. This notebook supplies the submission workflow; it does not improve those weights. Full hidden-test runtime and compressed-DICOM support must be verified on Kaggle. The saved competition requirements specify a nine-hour limit and internet disabled.
''')
code('''from pathlib import Path
# Usually discovered automatically. Set explicit paths if multiple versions are attached.
MODEL_PACKAGE = None
COMPETITION_ROOT = None
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working')
REQUIRE_GPU = True
IMAGE_BATCH = 8
MAX_HOURS = 8.5
# Explicit emergency policy, independent of hidden labels. Not calibrated probabilities.
FALLBACK_SCORES = [0.5] * 12
''')
md('''## Offline runtime
The package manifest is checked before importing the exported model and preprocessing code. The encoder is shared across folds. Series are processed one at a time to bound memory; only the deterministic validation slice centers are encoded. Float16 feature rounding and fold-specific spacing scalers match training.
''')
code((BASE/'inference_runtime.py').read_text())
md('''## Locate attached inputs
Both model mounting layouts used by Kaggle are supported. No online `model_download` call is used. A missing or ambiguous model input produces a clear error rather than choosing unrelated weights.
''')
code('''if MODEL_PACKAGE is None:
    candidates = [p.parent for p in discover(INPUT_ROOT, 'manifest.json')
                  if 'gc-rsna-knee-dinov2' in p.parts
                  and 'five-fold-attention' in p.parts]
    MODEL_PACKAGE = unique_path(candidates, 'MODEL_PACKAGE')
if COMPETITION_ROOT is None:
    candidates = [p.parent for p in discover(INPUT_ROOT, 'test_series.csv')
                  if (p.parent/'test.csv').is_file()
                  and (p.parent/'sample_submission.csv').is_file()]
    COMPETITION_ROOT = unique_path(candidates, 'COMPETITION_ROOT')
print('Custom model:', MODEL_HANDLE)
print('Attached package:', MODEL_PACKAGE)
print('Competition data:', COMPETITION_ROOT)
''')
md('''## Predict and write submission
This cell checks the exported synthetic head reference, processes every requested test study, averages the five sigmoid predictions, and validates the CSV after writing it. Series-level geometry/decoding failures are logged and the remaining usable series are used. If none are usable, explicit fixed fallback scores (0.5 per target by default) preserve coverage. These are emergency scores, not calibrated estimates. Every skipped series and fallback study is counted. Missing decoder dependencies and model errors still stop the run. Partial-study accuracy and fallback frequency need validation on real scans. Runtime estimates from the small visible sample are approximate.
''')
code('''SUBMISSION, RUN_RECEIPT = run_submission(
    MODEL_PACKAGE, COMPETITION_ROOT, OUTPUT_ROOT,
    require_gpu=REQUIRE_GPU, image_batch=IMAGE_BATCH, max_hours=MAX_HOURS, fallback=FALLBACK_SCORES,
)
display(SUBMISSION.head())
print('Status:', RUN_RECEIPT['status'])
print('Studies:', RUN_RECEIPT['studies'])
print('Skipped series:', RUN_RECEIPT['skipped_series'])
print('Fallback studies:', RUN_RECEIPT['fallback_count'])
print('Elapsed minutes:', round(RUN_RECEIPT['seconds']/60, 2))
''')
md('''## Outputs and submission
- **submission.csv:** study IDs and twelve probabilities, in the sample submission order.
- **processing_audit.csv:** series counts, timings, and geometry warnings.
- **run_manifest.json:** model identity, file fingerprints, runtime, and completion status.

Use Kaggle's **Save Version / Save & Run All** workflow with Internet disabled. After successful completion, submit that version's `submission.csv`. This notebook never trains, changes model visibility, or submits on your behalf.

### Terms
| Term | Meaning |
|---|---|
| Inference | Applying saved model weights to new scans. |
| Fold head | One of five prediction models trained on different data splits. |
| Encoder | DINOv2, which converts image slices into numerical features. |
| Sigmoid | Converts a model output into a score between zero and one. |
| Ensemble | Averaging predictions from the five heads. |
| Manifest | A record of the model configuration and saved file fingerprints. |
| Artifact | An output file, such as `submission.csv`. |
| Offline | Runs using attached files with internet disabled. |
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'kaggle':{'isInternetEnabled':False,'accelerator':'gpu'}})
nb.validate(book)
for cell in book.cells:
    if cell.cell_type=='code': compile(cell.source,'notebook_cell','exec')
nb.write(book,BASE/'submit-knee-dinov2-attention.ipynb')
print('Notebook generated and validated:',len(cells),'cells')
