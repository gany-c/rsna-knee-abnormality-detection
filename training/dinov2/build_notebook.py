from pathlib import Path
import ast,json,hashlib
import nbformat as nb
BASE=Path(__file__).parent
REPO=BASE.parents[1]
source=(BASE/'knee_training.py').read_text()
tree=ast.parse(source); lines=source.splitlines(keepends=True)
functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
header=''.join(lines[:functions[0].lineno-1])
# Embed the audited native transforms for future inference; do not execute them while training cached features.
prep=json.loads((REPO/'preprocessing/normalize-mri-for-training.ipynb').read_text())
prep_source='\n\n'.join(''.join(c['source']) for c in prep['cells'] if c['cell_type']=='code')
pt=ast.parse(prep_source);pl=prep_source.splitlines(keepends=True)
want={'validate_and_extract_geometry','native_grid','decode_native_slice','series_intensity_limits',
      'normalize_native_slice','select_native_indices','process_series','build_native_cache_arrays'}
prep_header='''from pathlib import Path
import hashlib, json, time, os, shutil, itertools
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
import pydicom
from pydicom.pixels import apply_modality_lut
ID='StudyInstanceUID'
SID='SeriesInstanceUID'
AXES = {
    'Sagittal': np.array([[1,0,0], [0,0,1], [0,-1,0]], float),
    'Coronal': np.array([[0,0,1], [1,0,0], [0,-1,0]], float),
    'Axial': np.array([[0,0,1], [0,1,0], [1,0,0]], float),
}
'''
prep_runtime=prep_header+'\n\n'.join(''.join(pl[n.lineno-1:n.end_lineno]) for n in pt.body if isinstance(n,ast.FunctionDef) and n.name in want)
compile(prep_runtime,'native_preprocessing.py','exec')
# Stored as a compressed constant to keep the training notebook readable; expands to plain source on export.
import zlib,base64
prep_blob=base64.b64encode(zlib.compress(prep_runtime.encode())).decode()
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# RSNA Knee MRI — frozen DINOv2 + finding-specific attention training

A self-contained **training** notebook for Kaggle. It reads our normalized MRI dataset directly, caches frozen DINOv2 Small features, trains fold-specific attention heads and exports a model package. It does not submit predictions, publish datasets, generate new report labels, or fine-tune the image encoder.

## Attach these inputs

1. **`gany24558/rsna-knee-normalized-training-data`**, latest complete version. All active shards must be processed. Existing labels/masks/weights/folds are inside these archives; no separate Qwen dataset is needed.
2. The competition dataset, for `train.csv` verification. Training images come from the normalized dataset, not the original DICOMs.
3. A **generic public DINOv2 Small, without registers** checkpoint. Supported: a Hugging Face `facebook/dinov2-small` directory with `config.json` and local weights, or Meta's official `dinov2_vits14_pretrain.pth` from an attached Kaggle model/dataset. No task-fine-tuned fold encoder. An arbitrary `.pth` architecture is not interchangeable.
4. Optional: previous **private** notebook outputs for feature/checkpoint resumption.

Select a GPU (one T4/P100 is sufficient for the design; runtime must be measured), leave **Internet off**, and use **Save Version → Run All**. Kaggle usually provides the required packages. There are no online `pip install`, model download, API credential or upload calls. If a package is missing, attach an offline wheel dataset or select a compatible Kaggle image.

## Rules translated into implementation

Reviewed project snapshots: `docs/rules.md`, `docs/data.md`, `docs/overview.md` (2026-09-24). External publicly accessible pretrained models are allowed; training does not read test labels or manually label test/validation records. The overview's **9-hour / Internet-off / submission.csv** requirements apply to the later submission notebook. This training run uses a conservative 7.5-hour budget and resumes across sessions.

Competition data redistribution is restricted. Keep this notebook and its complete output **private**: cached features, study IDs, training labels and OOF predictions are derived competition data. Only the separate `model_package` is prepared without those records; review the source rules and model license before distributing it. No automatic publishing occurs.

This is an implementable baseline, not a reproduced leaderboard score. The small verified set has already informed label development, so cross-validation is development evidence rather than an untouched final evaluation.''')
md('## 1. Configuration\nThe default runs feature extraction followed by all five folds. Use `features` or `train` to separate sessions. Hyperparameter changes create a separate training run; compatible frozen feature caches can still be reused.')
code('''from pathlib import Path
DATASET_ROOT = None      # Example: Path('/kaggle/input/datasets/gany24558/rsna-knee-normalized-training-data')
COMPETITION_ROOT = None  # Directory containing the original train.csv and train_series.csv
ENCODER_PATH = None      # HF folder OR path to dinov2_vits14_pretrain.pth; auto-discovery fails on ambiguity
ENCODER_BACKEND = 'auto' # 'auto', 'hf', or 'timm'
RESUME_ROOTS = []        # Previous run folders containing run_identity.json; otherwise discovered in /kaggle/input
OUTPUT_ROOT = Path('/kaggle/working/rsna-knee-training')
MODE = 'all'            # 'all', 'features', 'train'
FOLDS = [0, 1, 2, 3, 4] # For a pilot, use [0]; export is then clearly marked partial OOF
MAX_HOURS = 7.5
REQUIRE_GPU = True
CFG = dict(seed=42, hidden=256, lr=7e-4, weight_decay=2e-3,
           batch_size=64, epochs=32, patience=5, centers=16,
           verified_fraction=0.25, series_dropout=0.10, ema_decay=0.995,
           rank_lambda=0.0,     # Controlled experiment: 0.05 adds verified-only ranking loss
           refine_epochs=0,    # Controlled experiment: up to 5 verified-only epochs
           refine_lr=1e-4, image_batch=8, representation='single', # 'single' or 'triplet'
           min_free_gb=2.0, max_feature_ram_gb=5.0)
''')
md('## 2. Runtime imports and reproducibility\nDefinitions below are embedded in this notebook; no repository checkout is required. The exported runtime is assembled from these exact executed definitions.')
code(header)
# Split the real module into readable sections without duplicating code.
sections=[('Utilities and session budget',0,7),('Published-dataset loader and label audit',7,13),
          ('Normalized-series validation and offline encoder',13,19),('Resumable feature extraction',19,24),
          ('Study batches and attention architecture',24,27),('Loss, metrics and validation',27,33),
          ('Fold training and atomic checkpoints',33,37),('Model-package export',37,len(functions))]
# Derive grouping by explicit boundary names so edits cannot accidentally omit a function.
boundaries=[('Utilities and session budget','fingerprint'),('Published-dataset loader and label audit','discover_bundle'),
('Normalized-series validation and offline encoder','read_native'),('Resumable feature extraction','make_slice_inputs'),
('Study batches and attention architecture','sample_centers'),('Loss, metrics and validation','masked_bce'),
('Fold training and atomic checkpoints','make_epoch_pool'),('Model-package export','export_package')]
for j,(title,name) in enumerate(boundaries):
 start=next(i for i,n in enumerate(functions) if n.name==name)
 end=next(i for i,n in enumerate(functions) if n.name==boundaries[j+1][1]) if j+1<len(boundaries) else len(functions)
 md('### '+title)
 snippets=[]
 for n in functions[start:end]:
  first=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
  snippets.append(''.join(lines[first-1:n.end_lineno]))
 code('\n\n'.join(snippets))
md('## 3. Preflight and dataset audit\nAll active shards are required; legacy and partial shards are rejected. The original verified labels are cross-checked against `train.csv`. Patient-group separation is enforced where provided; a study-ID fallback is reported honestly.')
code('''import inspect, importlib.metadata, platform
if MODE not in {'all','features','train'}: raise ValueError('Invalid MODE')
if not FOLDS or len(set(FOLDS)) != len(FOLDS): raise ValueError('Choose unique folds')
if not (0 < CFG['verified_fraction'] < 1): raise ValueError('verified_fraction must lie between 0 and 1')
if CFG['hidden'] % 4 or min(CFG['epochs'], CFG['batch_size'], CFG['centers'], CFG['image_batch']) < 1:
    raise ValueError('Invalid model/training dimensions')
if not 0 <= CFG['series_dropout'] < 1 or CFG['refine_epochs'] < 0 or CFG['rank_lambda'] < 0:
    raise ValueError('Invalid regularization/refinement configuration')
if not 0 < MAX_HOURS <= 8.5: raise ValueError('Use a bounded session budget of at most 8.5 hours')
if REQUIRE_GPU and not torch.cuda.is_available(): raise RuntimeError('Enable a Kaggle GPU accelerator')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BUDGET = SessionBudget(MAX_HOURS)
seed_all(CFG['seed'])
ROOT = discover_bundle(DATASET_ROOT)
SERIES, LABELS, ALL_LABELS, INDEX = load_bundle(ROOT)
EXPECTED_PREPROCESSING_IMPLEMENTATION = 'ef1a2effbba938df32d512f010e3786904efc0cda3d04c95b074e29265018b5d'
if INDEX['preprocessing'].get('implementation') != EXPECTED_PREPROCESSING_IMPLEMENTATION:
    raise ValueError('Preprocessing implementation changed; update the embedded inference transforms before training')
if COMPETITION_ROOT is None:
    candidates = [p.parent for p in find_input_files('/kaggle/input','train.csv')
                  if (p.parent/'train_series.csv').is_file()]
    COMPETITION_ROOT = select_one(candidates, 'COMPETITION_ROOT')
OFFICIAL = pd.read_csv(Path(COMPETITION_ROOT)/'train.csv', dtype={ID:str})
if not OFFICIAL[ID].is_unique or set(OFFICIAL[ID]) != set(ALL_LABELS[ID]):
    raise ValueError('Official training IDs do not match the custom dataset')
original = OFFICIAL.set_index(ID).loc[ALL_LABELS[ID]]
for target in TARGETS:
    gold = original[target].notna().to_numpy()
    if not np.array_equal(gold, ALL_LABELS[target+'__gold'].to_numpy(bool)):
        raise ValueError('Verified-label mask drift: '+target)
    if not np.array_equal(original[target].to_numpy()[gold], ALL_LABELS[target].to_numpy()[gold]):
        raise ValueError('Verified-label value drift: '+target)
available_folds = sorted(LABELS.loc[LABELS.has_gold.eq(1),'fold'].unique().tolist())
if not set(FOLDS).issubset(available_folds): raise ValueError(f'Eligible verified folds are {available_folds}')
print('Device:', DEVICE)
print('Usable studies:', len(LABELS), '| series:', len(SERIES), '| excluded studies:', len(ALL_LABELS)-len(LABELS))
print('Grouping:', 'STUDY ONLY — no patient-separation guarantee' if
      ALL_LABELS.patient_group.eq(ALL_LABELS[ID]).all() else 'provided groups; verify patient provenance')
display(audit_labels(LABELS))
ENCODER_PATH, BACKEND = discover_encoder(ENCODER_PATH, ENCODER_BACKEND)
ENCODER_HASH = encoder_source_hash(ENCODER_PATH)
print('Offline encoder:', ENCODER_PATH, '| backend:', BACKEND)
ENCODER = FrozenEncoder(ENCODER_PATH, BACKEND).to(DEVICE).eval()
''')
md('## 4. Identity and restart discovery\nResume folders are private outputs from this notebook. Feature reuse checks encoder, preprocessing, code, representation and numerical-library identity. Head checkpoints additionally require the same complete training configuration.')
func_names=[n.name for n in functions]
code('''# Capture executed definitions so the exported code is exactly the code used here.
RUNTIME_HEADER = '''+repr(header)+'''
RUNTIME_NAMES = '''+repr(func_names)+'''
def definition_source(name):
    obj = globals()[name]
    if not inspect.isclass(obj):
        return inspect.getsource(obj)
    # inspect.getsource(Class) fails in Jupyter's __main__; recover its defining cell.
    import ast, linecache
    text = ''.join(linecache.getlines(obj.__init__.__code__.co_filename))
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
    return ast.get_source_segment(text, node) + chr(10)
RUNTIME_SOURCE = RUNTIME_HEADER + '\\n\\n'.join(definition_source(name) for name in RUNTIME_NAMES)
import base64, zlib
PREPROCESSING_SOURCE = zlib.decompress(base64.b64decode('''+repr(prep_blob)+''')).decode()
FEATURE_IDENTITY = dict(runtime=RUNTIME_VERSION, encoder_sha256=ENCODER_HASH, backend=BACKEND,
    dataset_identity=INDEX['identity'], representation=CFG['representation'], input_size=322,
    feature_pool='cls_support_weighted_patch_mean', candidates='all_acquired',
    torch_version=torch.__version__, numpy_version=np.__version__,
    encoder_library=importlib.metadata.version('transformers' if BACKEND=='hf' else 'timm'),
    code_sha256=hashlib.sha256((''.join(definition_source(name) for name in
        ['read_native','FrozenEncoder','make_slice_inputs','protocol_codes','extract_features'])).encode()).hexdigest())
TRAINING_ID = fingerprint(dict(cfg=CFG, feature=FEATURE_IDENTITY,
    runtime_sha256=hashlib.sha256(RUNTIME_SOURCE.encode()).hexdigest()))
WORK = OUTPUT_ROOT/TRAINING_ID[:16]
WORK.mkdir(parents=True, exist_ok=True)
atomic_json(WORK/'run_identity.json', dict(training_id=TRAINING_ID, feature_id=fingerprint(FEATURE_IDENTITY)))
atomic_json(WORK/'configuration.json', CFG)
atomic_json(WORK/'preprocessing_identity.json', INDEX)
audit_labels(LABELS).to_csv(WORK/'label_audit_private.csv', index=False)
if not RESUME_ROOTS:
    RESUME_ROOTS = [p.parent for p in find_input_files('/kaggle/input','run_identity.json')
                    if json.loads(p.read_text()).get('feature_id') == fingerprint(FEATURE_IDENTITY)]
RESUME_ROOTS = [Path(p) for p in RESUME_ROOTS]
for p in RESUME_ROOTS:
    if not (p/'run_identity.json').is_file(): raise FileNotFoundError('Resume folder needs run_identity.json: '+str(p))
HEAD_RESUME_ROOTS = [p for p in RESUME_ROOTS if json.loads((p/'run_identity.json').read_text()).get('training_id') == TRAINING_ID]
print('Private run folder:', WORK)
print('Compatible feature sources:', len(RESUME_ROOTS), '| head sources:', len(HEAD_RESUME_ROOTS))
'''.replace("'\\n\\n'","'\\n\\n'"))
md('## 5. Frozen feature extraction\nFeatures are cached per series, atomically. All valid slice centers are encoded once; each training batch samples up to 16. This can require multiple sessions. Save partial output privately and reattach it; do not bypass missing-shard checks. No claim is made that the entire dataset fits one GPU session.')
code('''if MODE == 'train':
    fid = fingerprint(FEATURE_IDENTITY)
    sources = [WORK/'features'/fid] + [p/'features'/fid for p in RESUME_ROOTS]
    paths = []
    for r in SERIES.to_dict('records'):
        path = next((p/feature_name(r) for p in sources if (p/feature_name(r)).is_file()), None)
        if path is None: raise FileNotFoundError('Feature cache incomplete; run MODE=features or all')
        check_features(path,r,fid)
        target = WORK/'features'/fid/feature_name(r)
        target.parent.mkdir(parents=True,exist_ok=True)
        if path.resolve() != target.resolve():
            if shutil.disk_usage(WORK).free < CFG['min_free_gb']*1e9 + path.stat().st_size:
                raise OSError('Insufficient space to retain cumulative feature output')
            tmp = target.with_suffix('.tmp'); shutil.copy2(path,tmp); tmp.replace(target)
        paths.append(str(target))
    FEATURE_SERIES = SERIES.copy(); FEATURE_SERIES['feature_path'] = paths
else:
    FEATURE_SERIES = extract_features(SERIES,ENCODER,INDEX['identity'],FEATURE_IDENTITY,WORK,
                                      RESUME_ROOTS,CFG,BUDGET)
READY = FEATURE_SERIES is not None
# Keep encoder on CPU while the small heads train; release GPU allocator reservations.
ENCODER = ENCODER.cpu()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if READY:
    atomic_json(WORK/'status.json',dict(status='FEATURES_COMPLETE',training_id=TRAINING_ID))
    print('Feature cache complete. Head training can now run without image decoding.')
''')
md('## 6. Train and validate fold heads\nUnknown labels contribute no loss. Metadata scalers use training folds only. The primary objective averages weighted BCE equally across supervised targets. Current and EMA weights compete on verified validation loss; best checkpoints are retained. Ranking loss and verified-only refinement are opt-in. Training checkpoints preserve the last complete epoch; an interruption restarts an unfinished epoch.')
code('''RESULTS = []
if READY and MODE != 'features':
    DATA = FeatureStudies(FEATURE_SERIES,LABELS,CFG['max_feature_ram_gb'])
    for fold in FOLDS:
        result = train_fold(DATA,fold,CFG,WORK,HEAD_RESUME_ROOTS,TRAINING_ID,BUDGET,DEVICE)
        if result[0] is None:
            atomic_json(WORK/'status.json',dict(status='TRAINING_PARTIAL',training_id=TRAINING_ID,
                        completed_folds=[r[0]['fold'] for r in RESULTS]))
            print('Training paused at session budget. Reattach this private output to resume.')
            break
        RESULTS.append(result)
else:
    print('Head training skipped: feature-only mode or incomplete features.')
''')
md('## 7. Export and verify\nOnly a completed requested fold set is exported. A one-fold pilot is marked as incomplete OOF coverage. The package includes a shared encoder, heads, scalers, manifests, exact runtime, native preprocessing functions/configuration, environment versions, and a synthetic head reload test. Private features and OOF rows stay outside the package. No submission file is generated by training.')
code('''if READY and MODE != 'features' and len(RESULTS) == len(FOLDS):
    PACKAGE, METRICS = export_package(ENCODER,DATA,RESULTS,CFG,INDEX['identity'],FEATURE_IDENTITY,
        TRAINING_ID,WORK,RUNTIME_SOURCE,PREPROCESSING_SOURCE,INDEX['preprocessing'])
    # Test strict head reload against deterministic synthetic inputs.
    reference = torch.load(PACKAGE/'synthetic_head_smoke.pt',map_location='cpu',weights_only=True)
    saved = torch.load(PACKAGE/f"head_fold_{RESULTS[0][0]['fold']}.pt",map_location='cpu',weights_only=True)
    reloaded = FindingAttention(saved['hidden']).eval(); reloaded.load_state_dict(saved['state_dict'],strict=True)
    with torch.inference_mode():
        actual = reloaded(reference['x'],reference['protocol'],reference['spacing'],reference['mask'])
    torch.testing.assert_close(actual,reference['expected'],rtol=1e-5,atol=1e-6)
    manifest = json.loads((PACKAGE/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if file_hash(PACKAGE/name) != digest: raise ValueError('Export checksum mismatch: '+name)
    atomic_json(WORK/'status.json',dict(status='COMPLETE_REQUESTED_FOLDS',training_id=TRAINING_ID,
                complete_oof=METRICS['complete_oof'],folds=FOLDS))
    display(pd.DataFrame(METRICS['pooled']['targets']))
    print('OOF macro AUROC:',METRICS['pooled']['macro_auroc'])
    print('Complete eligible OOF coverage:',METRICS['complete_oof'])
    print('Model package:',PACKAGE)
    print('Keep the full notebook output private. This run has not submitted or published anything.')
''')
md('''## Appendix: terms and limitations

| Term | Meaning |
|---|---|
| Study / series / slice | Examination / acquisition / one 2D image. |
| Frozen encoder | Image model whose parameters remain unchanged. |
| CLS / patch features | Whole-input summary / representations of small image regions. |
| Pooling | Combining slice features into a series representation. |
| Finding query | Learned attention request for one abnormality. |
| Label mask / support mask | Known diagnosis indicator / acquired-pixel indicator. |
| BCE / logit | Binary prediction loss / raw prediction before sigmoid. |
| EMA | Exponential moving average of model weights. |
| OOF | Predictions on each model's held-out fold. |
| AUROC / average precision | Ranking discrimination / a precision–recall summary. |
| Provenance | Record of model, data and preprocessing origins. |

The model uses study-level supervision, not lesion segmentation. Attention weights are not verified localization. Patch support weighting does not prevent padding from affecting the encoder internally. Undefined single-class AUROCs are recorded as unavailable. Report per-target coverage and results; our accepted generated labels do not provide equal support for all targets. There is no encoder fine-tuning or bootstrap uncertainty analysis in this initial notebook; both are later experiments.

References: local `training-algorithm.md`; Roman Tamrazov and evgendvorkin notebooks in `resources/reference_notebooks/`; [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md); [Transformers DINOv2 interface](https://huggingface.co/docs/transformers/model_doc/dinov2); [attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html).
''')
md('## Optional: save to Kaggle Models\nEnable Internet for this upload only. Set UPLOAD_TO_KAGGLE=True and run the final cell after export. New models default to private; existing models retain visibility. Keep inference offline.')
code((BASE/'upload_kaggle_model.py').read_text())
n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
 language_info=dict(name='python',version='3.11'),kaggle=dict(accelerator='gpu',isInternetEnabled=False)))
nb.validate(n)
for i,c in enumerate(n.cells):
 if c.cell_type=='code':compile(c.source,f'cell_{i}','exec')
p=BASE/'train-knee-dinov2-attention.ipynb';nb.write(n,p)
(BASE/'native_preprocessing.py').write_text(prep_runtime)
print(p, len(n.cells),'cells')
