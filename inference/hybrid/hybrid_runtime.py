"""Sequential offline inference and probability blending; no training or calibration."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ID = 'StudyInstanceUID'
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
           'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
HANDLES = {'resnet34':'gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil',
           'dinov2':'gany24558/gc-rsna-knee-dinov2/pyTorch/five-fold-attention'}

def json_write(path, obj):
    path=Path(path); tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str));tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def check_weights(weights):
    if set(weights)!=set(HANDLES):raise ValueError('Weights must specify resnet34 and dinov2')
    values=[float(weights[k]) for k in HANDLES]
    if any(not math.isfinite(x) or x<0 for x in values) or not math.isclose(sum(values),1.,abs_tol=1e-8,rel_tol=0):
        raise ValueError('Weights must be finite, nonnegative, and sum to one')
    return dict(zip(HANDLES,values))

def align_predictions(frame,sample):
    if list(sample.columns)!=[ID,*TARGETS] or sample.empty or sample[ID].isna().any() or sample[ID].duplicated().any():
        raise ValueError('Invalid sample submission schema or study IDs')
    if len(frame.columns)!=13 or set(frame.columns)!=set(sample.columns):
        raise ValueError('Prediction label columns differ from sample')
    if frame[ID].isna().any() or frame[ID].duplicated().any() or set(frame[ID])!=set(sample[ID]):
        raise ValueError('Prediction studies differ from sample or contain duplicates')
    aligned=frame.set_index(ID).loc[sample[ID],TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(aligned).all() or ((aligned<0)|(aligned>1)).any():
        raise ValueError('Predictions must be finite probabilities in [0,1]')
    return aligned

def blend_predictions(frames,sample,weights):
    weights=check_weights(weights)
    total=np.zeros((len(sample),12),dtype=np.float64)
    for name,weight in weights.items():
        if weight: total+=weight*align_predictions(frames[name],sample)
    result=pd.DataFrame(total,columns=TARGETS)
    result.insert(0,ID,sample[ID].tolist())
    align_predictions(result,sample)
    return result

def locate(root,filename):
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series','cache','features','.git','__pycache__'}]
        if filename in files:yield Path(folder)/filename

def resolve_inputs(root,packages,competition,weights):
    resolved={}
    for name,weight in check_weights(weights).items():
        if not weight:continue
        value=packages.get(name)
        if value is None:
            _,slug,_,variant=HANDLES[name].split('/')
            candidates=[p.parent for p in locate(root,'manifest.json') if slug in p.parts and variant in p.parts]
            if len(candidates)!=1:raise ValueError(f'Attach exactly one {name} model version or set its package path. Found {candidates}')
            value=candidates[0]
        value=Path(value)
        if not (value/'manifest.json').is_file():raise FileNotFoundError(f'Missing {name} manifest: {value}')
        resolved[name]=value
    if competition is None:
        candidates=[p.parent for p in locate(root,'test.csv') if (p.parent/'test_series.csv').is_file() and (p.parent/'sample_submission.csv').is_file()]
        if len(candidates)!=1:raise ValueError(f'Set COMPETITION_ROOT; candidates: {candidates}')
        competition=candidates[0]
    return resolved,Path(competition)

CHILD = '''import json, sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())
if cfg['name']=='resnet34':
    from resnet_inference import run_submission
    run_submission(cfg['package'],cfg['competition'],cfg['output'],
        require_gpu=cfg['require_gpu'],microbatch=cfg['batch'],max_hours=cfg['hours'],
        fallback_scores=cfg['fallback'],infer_missing_plane=True,require_standard_decoders=True)
else:
    # The same standard decoder preflight also applies to DINO-only runs.
    from resnet_inference import decoder_inventory, DecoderUnavailable
    missing=[r['uid'] for r in decoder_inventory() if not r['available']]
    if missing:raise DecoderUnavailable('Missing standard DICOM decoders: '+', '.join(missing))
    from dinov2_inference import run_submission
    run_submission(cfg['package'],cfg['competition'],cfg['output'],
        require_gpu=cfg['require_gpu'],image_batch=cfg['batch'],max_hours=cfg['hours'],fallback=cfg['fallback'])
'''

def run_hybrid(code_root,input_root,output_root,packages=None,competition=None,
               weights=None,batches=None,require_gpu=True,max_hours=8.5,started=None,
               fallback=None,decoder_dir='/kaggle/working/_dicom_deps'):
    started=time.monotonic() if started is None else started
    output=Path(output_root);output.mkdir(parents=True,exist_ok=True)
    final=output/'submission.csv'
    if final.exists():final.replace(output/f'submission.previous-{time.time_ns()}.csv')
    receipt={'status':'running','stage':'setup'}
    json_write(output/'hybrid_run_manifest.json',receipt)
    try:
        weights=check_weights(weights or {'resnet34':.8,'dinov2':.2})
        if not math.isfinite(max_hours) or max_hours<=0:raise ValueError('MAX_HOURS must be positive')
        batches=batches or {'resnet34':8,'dinov2':8}
        if any(type(batches.get(k)) is not int or batches[k]<1 for k,w in weights.items() if w):
            raise ValueError('Image batch sizes must be positive integers')
        fallback=[.5]*12 if fallback is None else fallback
        if np.asarray(fallback).shape!=(12,) or not np.isfinite(fallback).all() or np.any(np.asarray(fallback)<0) or np.any(np.asarray(fallback)>1):
            raise ValueError('Fallback must contain twelve probabilities')
        packages,competition=resolve_inputs(input_root,packages or {},competition,weights)
        sample=pd.read_csv(competition/'sample_submission.csv',dtype={ID:str})
        align_predictions(sample,sample)
        test=pd.read_csv(competition/'test.csv',dtype={ID:str})
        if test[ID].isna().any() or test[ID].duplicated().any() or set(test[ID])!=set(sample[ID]):
            raise ValueError('Test/sample study coverage differs')
        receipt.update(weights=weights,model_handles=HANDLES,packages={k:str(v) for k,v in packages.items()},components={})
        env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
        env['PYTHONPATH']=os.pathsep.join([str(decoder_dir),str(code_root),*sys.path])
        frames={}
        for name,weight in weights.items():
            if not weight:continue
            remaining=max_hours*3600-(time.monotonic()-started)
            if remaining<=30:raise TimeoutError('Combined runtime budget exhausted before '+name)
            receipt['stage']=name;json_write(output/'hybrid_run_manifest.json',receipt)
            component=output/name;component.mkdir(exist_ok=True)
            config={'name':name,'package':str(packages[name]),'competition':str(competition),
                    'output':str(component),'require_gpu':require_gpu,'batch':batches[name],
                    'hours':(remaining-15)/3600,'fallback':fallback}
            config_path=component/'invocation.json';json_write(config_path,config)
            print(f'Running {name}; weight={weight:.2f}; combined budget remaining {remaining/3600:.2f} h',flush=True)
            subprocess.run([sys.executable,'-u','-c',CHILD,str(config_path)],env=env,check=True,timeout=remaining-15)
            # Child exit releases the full model/GPU allocation before the other model loads.
            report=json.loads((component/'run_manifest.json').read_text())
            if report.get('status')!='complete':raise RuntimeError(name+' did not finish')
            frame=pd.read_csv(component/'submission.csv',dtype={ID:str})
            align_predictions(frame,sample);frames[name]=frame;receipt['components'][name]=report
            json_write(output/'hybrid_run_manifest.json',receipt)
        receipt['stage']='blend'
        result=blend_predictions(frames,sample,weights)
        if time.monotonic()-started>=max_hours*3600:raise TimeoutError('Combined runtime budget exhausted')
        tmp=output/'submission.csv.tmp';result.to_csv(tmp,index=False)
        align_predictions(pd.read_csv(tmp,dtype={ID:str}),sample);tmp.replace(final)
        receipt.update(status='complete',stage='complete',studies=len(result),seconds=time.monotonic()-started,
                       submission_sha256=digest(final),fallback_policy='Fixed weighted blend, including component fallback scores')
        json_write(output/'hybrid_run_manifest.json',receipt)
        print('Hybrid submission saved:',final,flush=True)
        return result,receipt
    except Exception as error:
        # A failed attempt must not leave a current submission artifact.
        if final.exists():final.replace(output/f'submission.failed-{time.time_ns()}.csv')
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        json_write(output/'hybrid_run_manifest.json',receipt)
        raise
