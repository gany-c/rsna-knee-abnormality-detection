from pathlib import Path
import nbformat as nb
base=Path(__file__).parent
cells=[nb.v4.new_markdown_cell('''# Audit real training scans before resubmission
Attach the competition data and `gany24558/gc-rsna-knee-dinov2/pyTorch/five-fold-attention/1`. Internet can remain disabled; no GPU is required. This notebook does not submit predictions or use hidden data.

Run this on Kaggle to measure failures that cannot be tested with the local synthetic scans. Default: up to 250 series selected deterministically across imaging planes. Set MAX_SERIES=0 to scan all training series. This can take hours. The exact saved preprocessing code is used without relaxing its geometry checks.

Outputs: `training_scan_audit.csv` and `training_scan_summary.json`. Keep these private. This measures preprocessing failures; it does not establish prediction accuracy after dropping series or using fallback scores.'''),nb.v4.new_code_cell('''from pathlib import Path
MODEL_PACKAGE = None
COMPETITION_ROOT = None
MAX_SERIES = 250
OUTPUT_ROOT = Path('/kaggle/working')
'''), nb.v4.new_code_cell((base/'inference_runtime.py').read_text()),nb.v4.new_code_cell('''if MODEL_PACKAGE is None:
    MODEL_PACKAGE = unique_path([p.parent for p in discover('/kaggle/input','manifest.json')
        if 'gc-rsna-knee-dinov2' in p.parts and 'five-fold-attention' in p.parts], 'MODEL_PACKAGE')
if COMPETITION_ROOT is None:
    COMPETITION_ROOT = unique_path([p.parent for p in discover('/kaggle/input','train_series.csv')
        if (p.parent/'train_series').is_dir()], 'COMPETITION_ROOT')
package = Path(MODEL_PACKAGE)
manifest = json.loads((package/'manifest.json').read_text())
for name in ['native_preprocessing.py','preprocessing_config.json']:
    if sha256(package/name) != manifest['files'][name]:
        raise ValueError('Preprocessing checksum mismatch: '+name)
prep = load_module('audit_preprocessing',package/'native_preprocessing.py')
config = json.loads((package/'preprocessing_config.json').read_text())['config']
frame = pd.read_csv(Path(COMPETITION_ROOT)/'train_series.csv',dtype={ID:str,SID:str})
# Round-robin across planes after seeded shuffling; no label-based sampling.
frame = frame.sample(frac=1,random_state=42)
frame['_rank'] = frame.groupby('Anatomical_Plane',dropna=False).cumcount()
frame = frame.sort_values('_rank',kind='stable')
if MAX_SERIES: frame=frame.head(MAX_SERIES)
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
rows=[]
for number, record in enumerate(frame.to_dict('records')):
    started=time.monotonic()
    row={ID:record[ID],SID:record[SID],'plane':record['Anatomical_Plane']}
    try:
        image,support,meta=prep.process_series(Path(COMPETITION_ROOT)/'train_series'/record[ID]/record[SID],
            record[ID],record[SID],record['Anatomical_Plane'],config)
        row.update(status='ok',warning=meta.get('review_warning',''),
                   source_slices=meta['input_shape'][0],transfer_syntaxes=';'.join(meta['transfer_syntaxes']))
        del image,support
    except (InvalidDicomError,ValueError,OSError,RuntimeError,AttributeError) as error:
        row.update(status='failed',error_type=type(error).__name__,error=str(error))
    row['seconds']=time.monotonic()-started;rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT/'training_scan_audit.csv',index=False)
    if (number+1)%10==0:print(f'Audited {number+1}/{len(frame)}',flush=True)
audit=pd.DataFrame(rows)
summary={'series':len(audit),'failed':int(audit.status.eq('failed').sum()),
         'complete_training_scan':MAX_SERIES==0,'model_package':str(package),
         'preprocessing_sha256':manifest['files']['native_preprocessing.py']}
json_write(OUTPUT_ROOT/'training_scan_summary.json',summary)
print(summary)
display(audit.loc[audit.status.eq('failed')].head(30))
''')]
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'kaggle':{'isInternetEnabled':False}})
nb.validate(book)
for cell in cells:
    if cell.cell_type=='code':compile(cell.source,'audit_cell','exec')
nb.write(book,base/'audit-knee-training-scans.ipynb')
