from pathlib import Path
import hashlib
import nbformat as nb
BASE=Path(__file__).parent
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# RSNA knee MRI — weighted ResNet34 + DINOv2 submission

Default: **80% ResNet34 + 20% DINOv2**, applied to probabilities for each study/finding. These are experimental weights, not validation-optimized weights; improvement is not guaranteed.

Attach these four inputs on Kaggle:
1. **RSNA Knee Abnormality Detection** competition data.
2. **gany24558/gc-rsna-knee-resnet34 → PyTorch → study-mil** — choose the model version used by your successful ResNet submission.
3. **gany24558/gc-rsna-knee-dinov2 → PyTorch → five-fold-attention** — choose the model version used by your successful DINOv2 submission.
4. **gany24558/rsna-dicom-decoders** — the Python 3.12 Linux GDCM wheel dataset.

Enable a **GPU**, disable **Internet**, start a **fresh session**, and run all cells. No training datasets, reports, API tokens, or original pretrained models are needed. All code is embedded; upload only this notebook.

Outputs: `/kaggle/working/submission.csv` plus component predictions/audits and `hybrid_run_manifest.json`. Submit the top-level `submission.csv` from the completed saved notebook version. Nothing is submitted automatically.
''')
code('''from pathlib import Path
import time
SESSION_STARTED = time.monotonic()
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working')
RESNET_PACKAGE = None # Explicit manifest.json parent if multiple versions are attached
DINOV2_PACKAGE = None
COMPETITION_ROOT = None
WHEELHOUSE = None
WEIGHTS = {'resnet34': 0.80, 'dinov2': 0.20} # Must sum to 1; same weights for all 12 findings
BATCHES = {'resnet34': 8, 'dinov2': 8}
REQUIRE_GPU = True
MAX_HOURS = 8.5 # Combined budget, including elapsed notebook setup; not per model
FALLBACK_SCORES = [0.5] * 12
''')
md('''## Install the offline DICOM decoder
This runs before either inference runtime imports pydicom. It installs only the attached GDCM wheel, with no network access. If you already imported pydicom, restart the session and Run All.''')
code((BASE/'install_decoders.py').read_text())
md('''## Extract embedded runtimes
The original ResNet34 and DINOv2 runtime sources are retained unchanged. Each model uses its own exported preprocessing, manifest checks and saved-weight tests. They run in separate processes, sequentially, so their Python modules cannot collide and GPU memory is released between models. The installed decoder path is explicitly passed to both processes.''')
sources={name:(BASE/name).read_text() for name in ['resnet_inference.py','dinov2_inference.py','hybrid_runtime.py']}
code("import sys\nCODE_ROOT = OUTPUT_ROOT / '_hybrid_code'\nCODE_ROOT.mkdir(parents=True, exist_ok=True)\nSOURCES = "+repr(sources)+"\nfor name, source in SOURCES.items():\n    (CODE_ROOT / name).write_text(source)\nif str(CODE_ROOT) not in sys.path:\n    sys.path.insert(0, str(CODE_ROOT))\n# Reload the wrapper when rerunning a cell after changing embedded source.\nsys.modules.pop('hybrid_runtime', None)\nfrom hybrid_runtime import run_hybrid\nprint('Embedded inference code ready.')")
md('''## Predict, align and blend
Both models process all requested test studies. Predictions are matched by **StudyInstanceUID and finding name**, never blended by unverified row position. Scores remain probabilities; no thresholds, additional sigmoid, prevalence adjustment, or hidden-test calibration are used.

The original per-model policy for invalid series is retained. If a component has no usable series, its declared 0.5 fallback still receives its configured blend weight; inspect its audit counts. Missing decoder/model dependencies remain fatal. If either positive-weight model fails or the combined time budget expires, no current top-level submission is produced. A zero weight skips that model.

Full hidden-test runtime will exceed that of either standalone model and has not been measured. The three example studies do not establish the total scoring time.''')
code('''SUBMISSION, RECEIPT = run_hybrid(
    CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT,
    packages={'resnet34': RESNET_PACKAGE, 'dinov2': DINOV2_PACKAGE},
    competition=COMPETITION_ROOT, weights=WEIGHTS, batches=BATCHES,
    require_gpu=REQUIRE_GPU, max_hours=MAX_HOURS, started=SESSION_STARTED,
    fallback=FALLBACK_SCORES,
)
display(SUBMISSION.head())
print({k: RECEIPT[k] for k in ['status', 'studies', 'weights', 'seconds']})
for name, report in RECEIPT['components'].items():
    print(name, 'fallback studies:', report.get('fallback_studies', report.get('fallback_count', 0)),
          'skipped series:', report['skipped_series'])
''')
md('''## Save and submit
Confirm the hybrid receipt says `complete`, inspect both component audit counts, then **Save Version → Save & Run All** with GPU enabled and Internet disabled. Submit `/kaggle/working/submission.csv` (not either component's CSV).

- `resnet34/`: original CNN predictions, decoder report and study/series audits.
- `dinov2/`: original transformer predictions and processing audit.
- `hybrid_run_manifest.json`: weights, mounted model paths, component receipts and output checksum.

Keep outputs private. Choose weights using held-out officially labeled studies unseen during either model's training. Do not assume that a blend improves the successful ResNet baseline.

**Terms:** probability blending is a weighted average of final model scores; this is not a newly trained hybrid architecture. Sequential inference means evaluating one model after the other. A fallback is an explicit emergency score for unusable data, not a calibrated prediction.''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'kaggle':{'isInternetEnabled':False,'accelerator':'gpu'}})
nb.validate(book)
for c in cells:
    if c.cell_type=='code':compile(c.source,'hybrid_cell','exec')
nb.write(book,BASE/'submit-knee-hybrid.ipynb')
print('Built validated hybrid notebook:',len(cells),'cells')
