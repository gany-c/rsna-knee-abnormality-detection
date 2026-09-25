"""Offline inference using the training package's exact adapters and transforms."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import importlib.util, importlib.metadata, hashlib, json, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pydicom.errors import InvalidDicomError

ID = 'StudyInstanceUID'
SID = 'SeriesInstanceUID'
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
           'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
MODEL_HANDLE = 'gany24558/gc-rsna-knee-dinov2/pyTorch/five-fold-attention'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    temp.replace(path)

def discover(root, filename):
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {'train_series','test_series','.cache'}]
        if filename in files:
            found.append(Path(folder) / filename)
    return found

def unique_path(candidates, description):
    if len(candidates) != 1:
        raise ValueError(f'Set the explicit {description} path. Candidates: {candidates}')
    return candidates[0]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Execute verified source without writing __pycache__ into the model package.
    exec(compile(Path(path).read_text(), str(path), 'exec'), module.__dict__)
    return module

def validate_submission(frame, sample):
    if list(frame.columns) != [ID, *TARGETS] or list(sample.columns) != [ID, *TARGETS]:
        raise ValueError('Submission columns do not match the twelve-target contract')
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError('Missing or duplicate submission IDs')
    if frame[ID].tolist() != sample[ID].tolist():
        raise ValueError('Submission IDs/order differ from sample_submission.csv')
    values = frame[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid prediction probabilities')

def load_package(package, device):
    package = Path(package)
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('runtime_version') != 'rsna-frozen-dino-v1':
        raise ValueError('Unsupported model package version')
    if manifest['targets'] != TARGETS or not manifest['complete_oof']:
        raise ValueError('Wrong targets or incomplete OOF package')
    if sorted(f['fold'] for f in manifest['folds']) != list(range(5)):
        raise ValueError('Expected exactly five distinct folds')
    required = {'knee_runtime.py','native_preprocessing.py','preprocessing_config.json',
                'synthetic_head_smoke.pt','encoder/adapter.json'}
    required.update(f['file'] for f in manifest['folds'])
    if not required.issubset(manifest['files']):
        raise ValueError('Missing required manifest entries')
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Unexpected or missing package files')
    for name, digest in manifest['files'].items():
        path = package/name
        if path.is_symlink() or not path.resolve().is_relative_to(package.resolve()):
            raise ValueError('Unsafe model package path')
        if sha256(path) != digest:
            raise ValueError('Model checksum mismatch: '+name)
    runtime = load_module('saved_knee_runtime', package/'knee_runtime.py')
    prep = load_module('saved_knee_preprocessing', package/'native_preprocessing.py')
    prep_identity = json.loads((package/'preprocessing_config.json').read_text())
    config = prep_identity['config']
    if config['depth'] != 64 or config['size'] != 320 or not config.get('fit_full_fov'):
        raise ValueError('Unexpected preprocessing image contract')
    cfg = manifest['configuration']
    if cfg['representation'] not in {'single','triplet'} or not 1 <= cfg['centers'] <= 64:
        raise ValueError('Invalid slice sampling contract')
    heads = []
    for fold in manifest['folds']:
        saved = torch.load(package/fold['file'], map_location='cpu', weights_only=True)
        if saved['fold'] != fold['fold'] or sha256(package/fold['file']) != fold['sha256']:
            raise ValueError('Fold identity mismatch')
        model = runtime.FindingAttention(saved['hidden']).eval()
        model.load_state_dict(saved['state_dict'], strict=True)
        scaler = saved['scaler']
        mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
        if scaler['transform'] != 'log_mm' or mean.shape != (3,) or std.shape != (3,):
            raise ValueError('Unsupported spacing scaler')
        if not np.isfinite([mean,std]).all() or (std <= 0).any():
            raise ValueError('Invalid spacing scaler')
        heads.append((model, scaler))
    reference = torch.load(package/'synthetic_head_smoke.pt', map_location='cpu', weights_only=True)
    with torch.inference_mode():
        actual = heads[0][0](reference['x'],reference['protocol'],reference['spacing'],reference['mask'])
    torch.testing.assert_close(actual, reference['expected'], rtol=1e-5, atol=1e-6)
    heads = [(model.to(device), scaler) for model,scaler in heads]
    encoder = runtime.FrozenEncoder.from_export(package/'encoder').to(device).eval()
    return manifest, runtime, prep, config, encoder, heads

def predict_study(records, dicom_root, bundle, device, image_batch, fallback=None):
    manifest, runtime, prep, config, encoder, heads = bundle
    cfg = manifest['configuration']
    features, protocols, spacings, audit = [], [], [], []
    for record in records:
        started = time.monotonic()
        try:
            image, support, meta = prep.process_series(
                dicom_root/record[ID]/record[SID], record[ID], record[SID],
                record['Anatomical_Plane'], config)
        except (InvalidDicomError, ValueError, OSError, RuntimeError, AttributeError) as error:
            # Only data preparation failures are recoverable; encoder/head errors propagate.
            message = str(error)
            if isinstance(error, RuntimeError) and any(term in message.lower() for term in
                    ['missing dependencies', 'no available plugins', 'all plugins are missing']):
                raise RuntimeError('A DICOM decoder dependency is missing; attach offline decoder wheels. ' + message) from error
            audit.append({ID:record[ID], SID:record[SID], 'status':'skipped',
                          'error_type':type(error).__name__, 'error':message,
                          'seconds':time.monotonic()-started})
            continue
        centers = runtime.sample_centers(int(meta['selected_count']), cfg['centers'])
        chunks = []
        for offset in range(0,len(centers),image_batch):
            x, v = runtime.make_slice_inputs(image, support, centers[offset:offset+image_batch], cfg['representation'])
            context = torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
            with torch.inference_mode(), context:
                f = encoder(x.to(device), v.to(device)).float().cpu().numpy()
            # Training caches slice features as float16 before float32 aggregation.
            chunks.append(f.astype(np.float16).astype(np.float32))
        f = np.concatenate(chunks)
        if not np.isfinite(f).all():
            raise ValueError('Nonfinite encoder features')
        features.append(np.concatenate([f.mean(0), f.max(0)]))
        protocols.append(runtime.protocol_codes(record))
        spacings.append(meta['spacing_summary_drc_mm'])
        audit.append({ID:record[ID], SID:record[SID], 'status':'ok',
                      'source_slices':meta['input_shape'][0], 'encoded_slices':len(centers),
                      'seconds':time.monotonic()-started, 'warning':meta.get('review_warning','')})
    if not features:
        if fallback is None:
            raise ValueError('Study has no usable series and no fallback is configured')
        audit.append({ID:records[0][ID] if records else '', SID:'',
                      'status':'fallback', 'error':'No usable series; fixed fallback scores used'})
        return np.asarray(fallback,dtype=np.float32).copy(), audit
    spacing = np.asarray(spacings, np.float32)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid physical spacing')
    x = torch.as_tensor(np.asarray(features)[None], device=device)
    p = torch.as_tensor(np.asarray(protocols)[None], device=device)
    mask = torch.ones((1,len(features)),dtype=torch.bool,device=device)
    predictions = []
    with torch.inference_mode():
        for model, scaler in heads:
            scaled = ((np.log(spacing)-np.asarray(scaler['mean']))/np.asarray(scaler['std'])).astype(np.float32)
            s = torch.as_tensor(scaled[None], device=device)
            predictions.append(model(x,p,s,mask).sigmoid().float().cpu().numpy()[0])
    prediction = np.mean(predictions,axis=0)
    if prediction.shape != (12,) or not np.isfinite(prediction).all():
        raise ValueError('Invalid study prediction')
    return prediction, audit

def run_submission(package, competition, output, require_gpu=True, image_batch=8, max_hours=8.5, fallback=None):
    if fallback is None:
        fallback = np.full(12, 0.5, dtype=np.float32)
    fallback = np.asarray(fallback, dtype=np.float32)
    if fallback.shape != (12,) or not np.isfinite(fallback).all() or ((fallback<0)|(fallback>1)).any():
        raise ValueError('Fallback must contain twelve finite scores between zero and one')
    started = time.monotonic()
    package, competition, output = map(Path, (package,competition,output))
    output.mkdir(parents=True, exist_ok=True)
    # Archive a previous output so a failed new run cannot leave a stale submission.csv.
    if (output/'submission.csv').exists():
        (output/'submission.csv').replace(output/f'submission.previous-{time.time_ns()}.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if require_gpu and device.type != 'cuda':
        raise RuntimeError('Enable a Kaggle GPU accelerator before running inference')
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    test = pd.read_csv(competition/'test.csv',dtype={ID:str})
    sample = pd.read_csv(competition/'sample_submission.csv',dtype={ID:str})
    series = pd.read_csv(competition/'test_series.csv',dtype={ID:str,SID:str})
    validate_submission(sample,sample)
    if test[ID].isna().any() or test[ID].duplicated().any() or set(test[ID]) != set(sample[ID]):
        raise ValueError('Test/sample study coverage mismatch')
    required = {ID,SID,'Anatomical_Plane','Fluid_Sensitive','Fat_Suppression'}
    if not required.issubset(series.columns) or series[[ID,SID]].isna().any().any():
        raise ValueError('Missing series metadata')
    if series.duplicated([ID,SID]).any() or not set(series[ID]).issubset(set(test[ID])):
        raise ValueError('Duplicate series or incomplete/unexpected study coverage')
    bundle = load_package(package,device)
    manifest = bundle[0]
    print('Loaded five heads and shared encoder. Exported head smoke test passed.', flush=True)
    predictions, audits = [], []
    groups = {uid: g.sort_values(SID).to_dict('records') for uid,g in series.groupby(ID)}
    try:
        for index, uid in enumerate(sample[ID]):
            if time.monotonic()-started > max_hours*3600:
                raise TimeoutError('Inference exceeded its runtime budget')
            pred, audit = predict_study(groups.get(uid, []),competition/'test_series',bundle,device,image_batch,fallback)
            for row in audit:
                row[ID] = uid
            predictions.append(pred); audits.extend(audit)
            pd.DataFrame(audits).to_csv(output/'processing_audit.csv',index=False)
            elapsed = time.monotonic()-started
            print(f'Studies {index+1}/{len(sample)}; elapsed {elapsed/60:.1f} min; '
                  f'rough total estimate {elapsed/(index+1)*len(sample)/60:.1f} min',flush=True)
    except Exception as error:
        json_write(output/'run_manifest.json',dict(status='failed',model_handle=MODEL_HANDLE,
                   study=uid,completed_studies=len(predictions),error=str(error),seconds=time.monotonic()-started))
        raise
    result = pd.DataFrame(np.stack(predictions),columns=TARGETS)
    result.insert(0,ID,sample[ID].tolist())
    validate_submission(result,sample)
    temp = output/'submission.csv.tmp'
    result.to_csv(temp,index=False)
    validate_submission(pd.read_csv(temp,dtype={ID:str}),sample)
    temp.replace(output/'submission.csv')
    receipt = dict(status='complete',model_handle=MODEL_HANDLE,mounted_package=str(package),
                   package_manifest_sha256=sha256(package/'manifest.json'),training_id=manifest['training_id'],
                   model_folds=[f['fold'] for f in manifest['folds']],studies=len(sample),series=len(series),
                   seconds=time.monotonic()-started,device=str(device),image_batch=image_batch,
                   encoder_precision='float16 autocast' if device.type=='cuda' else 'float32',
                   peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                   submission_sha256=sha256(output/'submission.csv'),
                   fallback_count=sum(a['status']=='fallback' for a in audits),
                   skipped_series=sum(a['status']=='skipped' for a in audits),
                   fallback_scores=fallback.tolist(),
                   partial_study_count=len({a[ID] for a in audits if a['status']=='skipped'} -
                                           {a[ID] for a in audits if a['status']=='fallback'}),
                   environment={name:importlib.metadata.version(name) for name in
                                ['torch','numpy','pandas','pydicom','scipy']})
    json_write(output/'run_manifest.json',receipt)
    print('Validated submission saved:',output/'submission.csv')
    return result, receipt
