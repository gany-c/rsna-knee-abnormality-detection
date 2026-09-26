"""Offline ResNet34 submission using the exact exported validation transforms."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys, types, json, hashlib, time, csv, traceback, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.pixels import get_decoder

ID='StudyInstanceUID';SID='SeriesInstanceUID'
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
MODEL_HANDLE='gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil'
STANDARD_SYNTAXES=['1.2.840.10008.1.2','1.2.840.10008.1.2.1','1.2.840.10008.1.2.4.57',
                   '1.2.840.10008.1.2.4.70','1.2.840.10008.1.2.4.90','1.2.840.10008.1.2.4.91']

class DecoderUnavailable(Exception):pass
class SeriesDataError(Exception):pass

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str));tmp.replace(path)

def find_files(root,name):
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series','cache','features','.git','__pycache__'}]
        if name in files:yield Path(folder)/name

def resolve_inputs(input_root,package=None,competition=None):
    if package is None:
        matches=[p.parent for p in find_files(input_root,'manifest.json')
                 if 'gc-rsna-knee-resnet34' in p.parts and 'study-mil' in p.parts]
        if len(matches)!=1:raise ValueError(f'Attach one CNN model version or set MODEL_PACKAGE. Found: {matches}')
        package=matches[0]
    if competition is None:
        matches=[p.parent for p in find_files(input_root,'test.csv') if (p.parent/'sample_submission.csv').is_file()]
        if len(matches)!=1:raise ValueError(f'Set COMPETITION_ROOT. Found: {matches}')
        competition=matches[0]
    return Path(package),Path(competition)

def source_module(name,path):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(Path(path).read_text(),str(path),'exec'),module.__dict__)
    return module

def load_package(package,device):
    package=Path(package).resolve();manifest=json.loads((package/'manifest.json').read_text())
    if manifest.get('schema_version')!=1 or manifest.get('model_family')!='resnet34-study-mean-max-v1':
        raise ValueError('Attach the ResNet34 study-mil package, not the DINO model')
    if manifest.get('targets')!=TARGETS:raise ValueError('Target order differs from training')
    required={'model.pt','cnn_runtime.py','knee_data.py','native_preprocessing.py','preprocessing_config.json','synthetic_smoke.pt'}
    if not required.issubset(manifest['files']):raise ValueError('Incomplete exported package')
    for name,digest in manifest['files'].items():
        path=package/name
        if not path.resolve().is_relative_to(package) or not path.is_file():raise ValueError('Invalid package member: '+name)
        if sha256(path)!=digest:raise ValueError('Package checksum mismatch: '+name)
    helpers=source_module('resnet_saved_data',package/'knee_data.py')
    # cnn_runtime imports knee_data. Bind only during import to avoid stale notebook modules.
    previous=sys.modules.get('knee_data');sys.modules['knee_data']=helpers
    try:runtime=source_module('resnet_saved_runtime',package/'cnn_runtime.py')
    finally:
        if previous is None:sys.modules.pop('knee_data',None)
        else:sys.modules['knee_data']=previous
    prep=source_module('resnet_saved_preprocessing',package/'native_preprocessing.py')
    cfg=manifest['configuration'];contract=manifest['inference']
    expected={'representation':'adjacent-slice-triplet','resize':'bilinear antialias full-FOV',
              'sampling':'rounded linspace','series_selection':'all usable, sorted SeriesInstanceUID',
              'slice_pool':'mean+max','series_pool':'mean','outputs':'12 logits; sigmoid once'}
    if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('Unsupported CNN input contract')
    if cfg['image_size']!=contract['image_size'] or cfg['slices_per_series']!=contract['slices_per_series']:
        raise ValueError('Configuration and inference contract disagree')
    if not 1<=int(cfg['slices_per_series'])<=64 or not 32<=int(cfg['image_size'])<=1024:
        raise ValueError('Invalid image/slice dimensions')
    prep_identity=json.loads((package/'preprocessing_config.json').read_text());prep_cfg=prep_identity['config']
    if (prep_cfg['depth'],prep_cfg['size'])!=(64,320) or not prep_cfg.get('fit_full_fov'):
        raise ValueError('Unexpected native preprocessing geometry')
    saved=torch.load(package/'model.pt',map_location='cpu',weights_only=True)
    if saved['architecture']!='resnet34' or saved['dropout']!=cfg['dropout']:raise ValueError('Checkpoint architecture mismatch')
    model=runtime.ResNetKnee(pretrained=False,dropout=saved['dropout']).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    np.testing.assert_allclose(model.mean.flatten().numpy(),contract['normalization_mean'],atol=1e-7)
    np.testing.assert_allclose(model.std.flatten().numpy(),contract['normalization_std'],atol=1e-7)
    reference=torch.load(package/'synthetic_smoke.pt',map_location='cpu',weights_only=True)
    with torch.inference_mode():actual=model(reference['images']).float().cpu()
    torch.testing.assert_close(actual,reference['expected'],rtol=1e-4,atol=1e-5)
    model=model.to(device).eval()
    # Check the streaming aggregation against the exported forward method on multiple series.
    images=reference['images']+[reference['images'][0].flip(0)*.9]
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        direct=model(images).float()
        vectors=[series_vector(model,x,2,device) for x in images]
        streamed=model.head(torch.stack(vectors).mean(0,keepdim=True)).float()
    torch.testing.assert_close(streamed,direct,rtol=2e-3,atol=2e-4)
    return manifest,runtime,prep,prep_cfg,model

def decoder_inventory():
    rows=[]
    for uid in STANDARD_SYNTAXES:
        try:
            decoder=get_decoder(uid)
            rows.append(dict(uid=uid,available=bool(decoder.is_available),plugins=list(decoder.available_plugins),
                             missing=list(decoder.missing_dependencies)))
        except NotImplementedError as error:rows.append(dict(uid=uid,available=False,error=str(error)))
    return rows

def safe_uid(value):
    value=str(value)
    if not value or value in {'.','..'} or '/' in value or '\\' in value or value.strip()!=value:
        raise ValueError('Invalid identifier/path component')
    return value

def sample_schema(frame):
    if len(frame)==0 or len(frame.columns)!=13 or set(frame.columns)!={ID,*TARGETS}:raise ValueError('Invalid sample submission schema')
    if frame[ID].isna().any() or frame[ID].duplicated().any():raise ValueError('Missing/duplicate sample IDs')
    for uid in frame[ID]:safe_uid(uid)

def load_inputs(competition):
    sample=pd.read_csv(competition/'sample_submission.csv',dtype={ID:str});sample_schema(sample)
    test=pd.read_csv(competition/'test.csv',dtype={ID:str})
    if test[ID].isna().any() or test[ID].duplicated().any() or set(test[ID])!=set(sample[ID]):
        raise ValueError('test.csv and sample_submission.csv coverage mismatch')
    path=competition/'test_series.csv';notes={}
    if path.is_file():
        frame=pd.read_csv(path,dtype={ID:str,SID:str})
        if not {ID,SID}.issubset(frame.columns):raise ValueError('test_series.csv requires study and series IDs')
        notes['rows_outside_requested_studies']=int((~frame[ID].isin(sample[ID])).sum())
        frame=frame.loc[frame[ID].isin(sample[ID])].copy()
        if frame[[ID,SID]].isna().any().any():raise ValueError('Missing series identifiers')
        for uid in frame[SID]:safe_uid(uid)
        # Exact duplicate rows add no information; conflicting duplicates are audited per series.
        notes['exact_duplicate_rows']=int(frame.duplicated().sum());frame=frame.drop_duplicates()
        records=[]
        for _,g in frame.groupby([ID,SID],sort=True):
            row=g.iloc[0].to_dict()
            if len(g)>1:row['_metadata_error']='Conflicting duplicate series metadata'
            records.append(row)
    else:
        notes['series_table_missing']=True;records=[]
        for uid in sample[ID]:
            folder=competition/'test_series'/uid
            if folder.is_dir():
                records.extend({ID:uid,SID:safe_uid(p.name)} for p in sorted(folder.iterdir()) if p.is_dir())
    groups={uid:[] for uid in sample[ID]}
    for record in records:groups[record[ID]].append(record)
    for uid in groups:groups[uid].sort(key=lambda r:r[SID])
    return sample,groups,notes

def series_headers(folder,record,infer_plane=True):
    if record.get('_metadata_error'):raise SeriesDataError(record['_metadata_error'])
    paths=sorted(folder.glob('*.dcm'))
    if not paths:raise SeriesDataError('No DICOM files in series folder')
    first=None;syntaxes=set()
    for path in paths:
        header=pydicom.dcmread(path,stop_before_pixels=True)
        if first is None:first=header
        uid=str(header.file_meta.TransferSyntaxUID);syntaxes.add(uid)
    for uid in syntaxes:
        try:decoder=get_decoder(uid)
        except NotImplementedError as error:raise SeriesDataError('Unsupported transfer syntax: '+uid) from error
        if not decoder.is_available:
            raise DecoderUnavailable('Missing decoder for '+uid+': '+'; '.join(decoder.missing_dependencies))
    raw=record.get('Anatomical_Plane');plane=str(raw).strip().capitalize() if pd.notna(raw) else ''
    note=''
    if plane not in {'Sagittal','Coronal','Axial'}:
        if not infer_plane:raise SeriesDataError('Missing/unknown anatomical plane')
        iop=np.asarray(first.ImageOrientationPatient,float)
        if iop.shape!=(6,) or not np.isfinite(iop).all():raise SeriesDataError('Cannot infer plane from orientation')
        normal=np.cross(iop[:3],iop[3:]);norm=np.linalg.norm(normal)
        if norm<1e-6:raise SeriesDataError('Degenerate slice orientation')
        alignment=np.abs(normal/norm);rank=np.argsort(alignment)
        if alignment[rank[-1]]-alignment[rank[-2]]<.1:raise SeriesDataError('Ambiguous anatomical plane')
        plane=['Sagittal','Coronal','Axial'][int(rank[-1])]
        note='Plane inferred from DICOM orientation because metadata was missing/unknown'
    elif str(raw)!=plane:note='Plane spelling/case normalized'
    return plane,note,sorted(syntaxes)

@torch.inference_mode()
def series_vector(model,images,microbatch,device):
    features=[];offset=0;batch=int(microbatch)
    while offset<len(images):
        # Retry only CUDA allocation failures, reducing this image microbatch.
        try:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                x=images[offset:offset+batch].to(device)
                feature=model.backbone((x-model.mean)/model.std).float()
            features.append(feature);offset+=len(feature)
        except torch.cuda.OutOfMemoryError:
            if device.type!='cuda' or batch<=1:raise
            if 'x' in locals():del x
            torch.cuda.empty_cache();batch=max(1,batch//2)
    f=torch.cat(features)
    if not torch.isfinite(f).all():raise ValueError('Nonfinite CNN image features')
    return torch.cat([f.mean(0),f.max(0).values])

@torch.inference_mode()
def predict_study(uid,records,competition,bundle,device,microbatch,fallback,infer_plane,deadline,emit):
    manifest,runtime,prep,prep_cfg,model=bundle;vectors=[];failed=0
    for record in records:
        if time.monotonic()>deadline:raise TimeoutError('Inference session time budget reached')
        started=time.monotonic();row={ID:uid,SID:record[SID]}
        folder=competition/'test_series'/uid/record[SID]
        try:
            plane,note,syntaxes=series_headers(folder,record,infer_plane)
            image,support,meta=prep.process_series(folder,uid,record[SID],plane,prep_cfg)
        except DecoderUnavailable:raise
        except (SeriesDataError,InvalidDicomError,ValueError,OSError,AttributeError,KeyError,RuntimeError) as error:
            if isinstance(error,RuntimeError) and any(t in str(error).lower() for t in ['missing dependencies','all plugins are missing','no available plugins']):
                raise DecoderUnavailable(str(error)) from error
            failed+=1;row.update(status='skipped',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
            emit(row);continue
        images=runtime.prepare_series(image,support,meta,manifest['configuration'],rng=None)
        del image,support
        vectors.append(series_vector(model,images,microbatch,device));del images
        row.update(status='ok',seconds=time.monotonic()-started,source_slices=meta['input_shape'][0],
                   plane=plane,warning='; '.join(x for x in [note,meta.get('review_warning','')] if x),
                   transfer_syntaxes=';'.join(syntaxes))
        emit(row)
    if not vectors:return fallback.copy(),dict(status='fallback',usable_series=0,skipped_series=failed)
    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        logits=model.head(torch.stack(vectors).mean(0,keepdim=True))
    pred=logits.float().sigmoid().cpu().numpy()[0]
    if pred.shape!=(12,) or not np.isfinite(pred).all():raise ValueError('Nonfinite CNN prediction')
    return pred,dict(status='partial' if failed else 'ok',usable_series=len(vectors),skipped_series=failed)

def validate_submission(frame,sample):
    sample_schema(sample)
    if list(frame.columns)!=list(sample.columns) or frame[ID].tolist()!=sample[ID].tolist():
        raise ValueError('Submission columns, identifiers or ordering differ from sample')
    values=frame[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid submission probabilities')

def run_submission(package,competition,output,require_gpu=True,microbatch=8,max_hours=8.5,
                   fallback_scores=None,infer_missing_plane=True,require_standard_decoders=True):
    package,competition,output=map(Path,(package,competition,output));output.mkdir(parents=True,exist_ok=True)
    if (output/'submission.csv').exists():
        (output/'submission.csv').replace(output/f'submission.previous-{time.time_ns()}.csv')
    started=time.monotonic();stage='preflight';current=None;studies=[]
    receipt=dict(status='running',model_handle=MODEL_HANDLE,mounted_package=str(package))
    write_json(output/'run_manifest.json',receipt)
    try:
        if microbatch<1 or not 0<max_hours<=8.5:raise ValueError('Invalid microbatch/time budget')
        fallback=np.array([.5]*12 if fallback_scores is None else fallback_scores,dtype=np.float32)
        if fallback.shape!=(12,) or not np.isfinite(fallback).all() or ((fallback<0)|(fallback>1)).any():raise ValueError('Invalid fallback scores')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if require_gpu and device.type!='cuda':raise RuntimeError('Enable a Kaggle GPU accelerator')
        torch.manual_seed(42)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        sample,groups,notes=load_inputs(competition)
        decoders=decoder_inventory();write_json(output/'decoder_report.json',decoders)
        missing=[r['uid'] for r in decoders if not r['available']]
        print('Unavailable standard DICOM decoders:',missing or 'none',flush=True)
        if missing:
            print('Attach compatible offline decoder wheels. A missing decoder is not treated as a bad patient scan.',flush=True)
            if require_standard_decoders:
                raise DecoderUnavailable('Required scoring-data decoders unavailable: '+', '.join(missing)+'. See decoder_report.json and the offline wheel setup cell.')
        stage='model_loading';bundle=load_package(package,device);manifest=bundle[0]
        print(f"Loaded best CNN epoch {manifest['best_epoch']}; saved-weight and streaming checks passed.",flush=True)
        result=sample.copy();result[TARGETS]=np.nan;stage='prediction'
        fields=[ID,SID,'status','error_type','error','seconds','source_slices','plane','warning','transfer_syntaxes']
        with (output/'series_audit.csv').open('w',newline='') as log:
            writer=csv.DictWriter(log,fieldnames=fields);writer.writeheader()
            def emit(row):writer.writerow(row);log.flush()
            for index,uid in enumerate(sample[ID]):
                current=uid;check=time.monotonic()
                if check>started+max_hours*3600:raise TimeoutError('Inference session time budget reached')
                pred,status=predict_study(uid,groups[uid],competition,bundle,device,microbatch,fallback,infer_missing_plane,started+max_hours*3600,emit)
                result.loc[index,TARGETS]=pred
                studies.append({ID:uid,**status,'seconds':time.monotonic()-check})
                pd.DataFrame(studies).to_csv(output/'study_audit.csv',index=False)
                elapsed=time.monotonic()-started
                print(f'Studies {index+1}/{len(sample)}; {status["status"]}; elapsed {elapsed/60:.1f} min; rough total {elapsed/(index+1)*len(sample)/60:.1f} min',flush=True)
        stage='export';validate_submission(result,sample)
        temp=output/'submission.csv.tmp';result.to_csv(temp,index=False)
        validate_submission(pd.read_csv(temp,dtype={ID:str}),sample);temp.replace(output/'submission.csv')
        receipt.update(status='complete',studies=len(sample),usable_series=sum(r['usable_series'] for r in studies),
            skipped_series=sum(r['skipped_series'] for r in studies),fallback_studies=sum(r['status']=='fallback' for r in studies),
            partial_studies=sum(r['status']=='partial' for r in studies),fallback_scores=fallback.tolist(),
            input_notes=notes,package_manifest_sha256=sha256(package/'manifest.json'),best_epoch=manifest['best_epoch'],
            submission_sha256=sha256(output/'submission.csv'),device=str(device),microbatch=microbatch,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            infer_missing_plane=infer_missing_plane,seconds=time.monotonic()-started,
            environment={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','pandas','pydicom','scipy']})
        write_json(output/'run_manifest.json',receipt)
        print('Validated submission saved:',output/'submission.csv',flush=True)
        return result,receipt
    except Exception as error:
        receipt.update(status='failed',stage=stage,study=current,completed_studies=len(studies),
                       error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        write_json(output/'run_manifest.json',receipt)
        (output/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
