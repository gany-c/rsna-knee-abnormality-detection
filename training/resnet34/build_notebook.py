from pathlib import Path
import ast,json
import nbformat as nb
BASE=Path(__file__).parent
WORKSPACE=Path('/Users/ganapathychidambaram/Documents/RSNA Knee MRI')
source=(WORKSPACE/'training_build/knee_training.py').read_text();tree=ast.parse(source);lines=source.splitlines(keepends=True)
names={'fingerprint','file_hash','atomic_json','atomic_torch','select_one','find_input_files','discover_bundle','safe_member','load_bundle','validate_labels','audit_labels','read_native','evaluate_metrics'}
header=source[:source.index('def fingerprint')]
support=header+'\n'.join(''.join(lines[n.lineno-1:n.end_lineno])+'\n' for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
runtime=(BASE/'cnn_runtime.py').read_text();prep=(WORKSPACE/'training_build/native_preprocessing.py').read_text()
for name,text in [('knee_data.py',support),('cnn_runtime.py',runtime),('native_preprocessing.py',prep)]:compile(text,name,'exec')
(BASE/'knee_data.py').write_text(support)
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# RSNA Knee MRI — ResNet34 CNN training

A separate end-to-end CNN experiment. It borrows learning-rate groups, best-checkpoint saving and scheduling from the BirdCLEF notebooks; MRI preparation and label handling are specific to this competition.

Attach **gany24558/rsna-knee-normalized-training-data** (all completed shards) and the original competition data. Generated labels are already bundled. Enable a **GPU** and **Internet** for ImageNet weights and final Kaggle Model upload. If using attached ResNet34 ImageNet weights, set PRETRAINED_PATH. Do not attach DINO feature caches: CNN weights change during training.

The split holds out 20% of patient groups (or study groups if patient identities are unavailable). Only verified labels in held-out groups determine validation loss and AUROC. All labels in those groups are excluded from training. Early stopping: **three consecutive epochs without strictly lower validation loss**. The best epoch is exported, not the final epoch.

Upload destination: **gany24558/gc-rsna-knee-resnet34 / PyTorch / study-mil**. The final cell automatically uploads the real trained package when run. This notebook does not launch training until you run it on Kaggle. Keep the notebook/model private. The existing DINO submission notebook cannot load CNN weights; a CNN-specific inference adapter is needed later.
''')
code('''from pathlib import Path
DATASET_ROOT = None
COMPETITION_ROOT = None
PRETRAINED_PATH = None # Optional attached official torchvision resnet34 state dictionary
RESUME_CHECKPOINT = None # Your own trusted last.pt from an earlier identical run
OUTPUT_ROOT = Path('/kaggle/working/rsna-knee-resnet34-training')
REQUIRE_GPU = True
UPLOAD_MODEL = True
MODEL_HANDLE = 'gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil'
CFG = dict(seed=42,val_fraction=.2,image_size=224,slices_per_series=4,max_train_series=4,
           microbatch=8,accumulate=4,dropout=.2,backbone_lr=1e-5,head_lr=1e-4,
           epochs=35,patience=3,max_hours=7.5)
''')
md('## Runtime modules\nThese are embedded; no repository checkout or separate Python files are required on Kaggle.')
code('import sys, json, hashlib, random\nSOURCE_FILES = '+repr({'knee_data.py':support,'cnn_runtime.py':runtime,'native_preprocessing.py':prep})+'''\nCODE_ROOT=Path('/kaggle/working/_resnet34_code')
CODE_ROOT.mkdir(parents=True,exist_ok=True)
for name,text in SOURCE_FILES.items():(CODE_ROOT/name).write_text(text)
sys.path.insert(0,str(CODE_ROOT))
import knee_data as data
import cnn_runtime as cnn
import numpy as np
import pandas as pd
import torch
''')
md('## Validate data and split whole groups\nUnknown labels never become negative labels. Verified labels must agree exactly with the competition train.csv. The split table and label counts are private diagnostics.')
code('''if REQUIRE_GPU and not torch.cuda.is_available():raise RuntimeError('Enable a GPU')
if CFG['patience'] != 3:raise ValueError('This experiment requires patience=3')
if min(CFG[k] for k in ['microbatch','accumulate','slices_per_series','max_train_series','epochs'])<1:
    raise ValueError('Training sizes must be positive')
torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed']);random.seed(CFG['seed'])
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT=data.discover_bundle(DATASET_ROOT)
SERIES,LABELS,ALL_LABELS,INDEX=data.load_bundle(ROOT)
if INDEX['preprocessing'].get('implementation')!='ef1a2effbba938df32d512f010e3786904efc0cda3d04c95b074e29265018b5d':
    raise ValueError('Unexpected preprocessing implementation; audit before changing the contract')
if COMPETITION_ROOT is None:
    COMPETITION_ROOT=data.select_one([p.parent for p in data.find_input_files('/kaggle/input','train.csv')
        if (p.parent/'train_series.csv').is_file()],'COMPETITION_ROOT')
official=pd.read_csv(Path(COMPETITION_ROOT)/'train.csv',dtype={data.ID:str})
if not official[data.ID].is_unique or set(official[data.ID])!=set(ALL_LABELS[data.ID]):
    raise ValueError('Official training studies differ from generated-label dataset')
original=official.set_index(data.ID).loc[ALL_LABELS[data.ID]]
for target in data.TARGETS:
    gold=original[target].notna().to_numpy()
    if not np.array_equal(gold,ALL_LABELS[target+'__gold'].to_numpy(bool)):
        raise ValueError('Verified mask drift: '+target)
    if not np.array_equal(original[target].to_numpy()[gold],ALL_LABELS[target].to_numpy()[gold]):
        raise ValueError('Verified label drift: '+target)
TRAIN,VAL,HELD_GROUPS=cnn.split_studies(LABELS,CFG['val_fraction'],CFG['seed'])
IDENTITY=dict(configuration=CFG,dataset=INDEX['identity'],
              split_hash=data.fingerprint({'train':TRAIN[data.ID].tolist(),'held':sorted(HELD_GROUPS)}),
              source_hash=data.fingerprint(SOURCE_FILES),initialization='ImageNet ResNet34' if PRETRAINED_PATH is None else data.file_hash(PRETRAINED_PATH))
WORK=OUTPUT_ROOT/data.fingerprint(IDENTITY)[:16];WORK.mkdir(parents=True,exist_ok=True)
split=LABELS[[data.ID,'patient_group']].copy()
split['role']=np.where(split.patient_group.isin(HELD_GROUPS),'held_out','training_group')
split['used_for_validation']=split[data.ID].isin(VAL[data.ID])
split['used_for_training']=split[data.ID].isin(TRAIN[data.ID])
split.to_csv(WORK/'split_private.csv',index=False)
data.atomic_json(WORK/'run_identity.json',IDENTITY)
print('Training studies:',len(TRAIN),'Verified validation studies:',len(VAL))
print('All held-out studies:',split.role.eq('held_out').sum())
print('Grouping:', 'study only; patient separation unavailable' if LABELS.patient_group.eq(LABELS[data.ID]).all() else 'provided patient groups')
display(data.audit_labels(VAL))
MODEL=cnn.ResNetKnee(pretrained=RESUME_CHECKPOINT is None,weights_path=PRETRAINED_PATH,dropout=CFG['dropout']).to(DEVICE)
''')
md('## Train the CNN and classifier\nAdam uses 1e-5 for the pretrained CNN and 1e-4 for the head. Validation uses deterministic slices and all usable series. Each epoch scans all training studies. Checkpointing and accumulation bound GPU memory; BatchNorm running statistics stay fixed. If the session budget is reached, resume from last.pt in a new session. Only completed validated epochs are eligible for export.')
code('''BEST,HISTORY,STOP_REASON=cnn.train_model(MODEL,TRAIN,VAL,SERIES,INDEX['identity']['run_id'],CFG,WORK,IDENTITY,RESUME_CHECKPOINT)
print('Stop reason:',STOP_REASON,'Best epoch:',BEST['epoch'],'Validation loss:',BEST['val_loss'])
display(pd.DataFrame(HISTORY))
''')
md('## Export and check the best model\nOnly weights, runtime, transforms, configuration and aggregate metrics enter model_package. Split IDs, labels, validation predictions and optimizer checkpoints remain outside it.')
code('''PACKAGE=cnn.export_model(MODEL,BEST,CFG,INDEX,IDENTITY,WORK,SOURCE_FILES)
print('Exported and reload-tested:',PACKAGE)
''')
md('## Upload to the separate CNN Kaggle Model\nRequires Internet and Kaggle authentication. New models default to private; existing models retain visibility. Re-running this cell creates a new version. If upload fails, the local package remains saved; retry only this cell, without retraining.')
code('''if UPLOAD_MODEL:
    import kagglehub
    manifest=json.loads((PACKAGE/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if data.file_hash(PACKAGE/name)!=digest:raise ValueError('Export changed: '+name)
    kagglehub.model_upload(MODEL_HANDLE,str(PACKAGE),version_notes=(
        f"ResNet34; grouped holdout; best epoch {BEST['epoch']}; verified BCE {BEST['val_loss']:.6f}; patience 3"))
    print('Upload accepted: https://www.kaggle.com/models/'+MODEL_HANDLE)
else:
    print('Upload disabled; package remains at',PACKAGE)
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'kaggle':{'isInternetEnabled':True,'accelerator':'gpu'}})
nb.validate(book)
for c in cells:
    if c.cell_type=='code':compile(c.source,'cell','exec')
nb.write(book,BASE/'train-knee-resnet34.ipynb')
print('Built',len(cells),'cells')
