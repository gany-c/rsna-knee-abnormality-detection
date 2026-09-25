"""End-to-end ResNet34 study-level MRI training."""
from pathlib import Path
import copy, json, time, hashlib, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet34, ResNet34_Weights
import knee_data as data

class ResNetKnee(nn.Module):
    def __init__(self, pretrained=False, weights_path=None, dropout=.2):
        super().__init__()
        self.backbone=resnet34(weights=ResNet34_Weights.DEFAULT if pretrained and weights_path is None else None)
        if weights_path:
            self.backbone.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True)
        self.backbone.fc=nn.Identity()
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(1024,12))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode)
        # Small study batches: retain pretrained BN running statistics; affine weights train.
        for module in self.backbone.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self

    def forward(self,series_images,microbatch=8):
        if not series_images:raise ValueError('Empty study')
        device=self.mean.device; vectors=[]
        for images in series_images:
            features=[]
            for start in range(0,len(images),microbatch):
                x=images[start:start+microbatch].to(device)
                x=(x-self.mean)/self.std
                f=checkpoint(self.backbone,x,use_reentrant=False) if self.training else self.backbone(x)
                features.append(f.float())
            f=torch.cat(features)
            vectors.append(torch.cat([f.mean(0),f.max(0).values]))
        return self.head(torch.stack(vectors).mean(0,keepdim=True))

class EarlyStop:
    def __init__(self,patience=3):
        self.patience=patience;self.best=float('inf');self.bad_epochs=0
    def update(self,loss):
        if not np.isfinite(loss):raise ValueError('Nonfinite validation loss')
        improved=loss<self.best
        if improved:self.best=float(loss);self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=self.patience

def split_studies(labels,val_fraction=.2,seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    data.validate_labels(labels)
    splitter=GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed)
    train_ix,val_ix=next(splitter.split(labels,groups=labels.patient_group))
    held=set(labels.iloc[val_ix].patient_group)
    known=labels[[t+'__mask' for t in data.TARGETS]].to_numpy(bool)
    weights=labels[[t+'__weight' for t in data.TARGETS]].to_numpy(float)
    active=(known & (weights>0)).any(1)
    train=labels.iloc[train_ix].loc[active[train_ix]].copy()
    val=labels.iloc[val_ix].loc[labels.iloc[val_ix].has_gold.eq(1)].copy()
    if train.empty or val.empty:raise ValueError('Need nonempty training and verified validation studies; inspect split')
    if set(train.patient_group)&held:raise ValueError('Group leakage')
    return train.reset_index(drop=True),val.reset_index(drop=True),held

def prepare_series(image,support,meta,cfg,rng=None):
    n=int(meta['selected_count']);count=cfg['slices_per_series']
    if n<=count:centers=np.arange(n)
    elif rng is None:centers=np.linspace(0,n-1,count).round().astype(int)
    else:centers=np.array([rng.choice(chunk) for chunk in np.array_split(np.arange(n),count)])
    neighbors=np.clip(centers[:,None]+np.array([-1,0,1]),0,n-1)
    x=torch.from_numpy(image[neighbors].astype(np.float32))
    x=F.interpolate(x,size=(cfg['image_size'],cfg['image_size']),mode='bilinear',align_corners=False,antialias=True)
    # Full field of view, no flips or crops that might change anatomical interpretation.
    if rng is not None:
        x=(x*float(rng.uniform(.9,1.1))).clamp(0,1)
    return x

def study_images(uid,by_study,run_id,cfg,rng=None):
    records=by_study[uid]
    if rng is not None and len(records)>cfg['max_train_series']:
        records=[records[i] for i in sorted(rng.choice(len(records),cfg['max_train_series'],replace=False))]
    images=[]
    for record in records:
        image,support,meta=data.read_native(record,run_id)
        images.append(prepare_series(image,support,meta,cfg,rng))
    return images

def row_targets(row,verified=False):
    y=torch.tensor([[0 if pd.isna(row[t]) else float(row[t]) for t in data.TARGETS]],dtype=torch.float32)
    mask=torch.tensor([[bool(row[t+('__gold' if verified else '__mask')]) for t in data.TARGETS]])
    w=torch.ones_like(y) if verified else torch.tensor([[float(row[t+'__weight']) for t in data.TARGETS]])
    return y,mask,w

def masked_loss(logits,y,mask,weights):
    w=mask.to(logits.dtype)*weights
    if not (w>0).any():raise ValueError('Study has no supervised targets')
    return (F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()/w.sum()

class BudgetReached(Exception):pass

def check_budget(deadline):
    if time.monotonic()>deadline:raise BudgetReached('Session time budget reached')

@torch.inference_mode()
def validate(model,val,by_study,run_id,cfg,device,deadline):
    model.eval();zs=[];ys=[];masks=[]
    for row in val.to_dict('records'):
        check_budget(deadline)
        images=study_images(row[data.ID],by_study,run_id,cfg)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            z=model(images,cfg['microbatch'])
        zs.append(z.float().cpu());y,m,_=row_targets(row,True);ys.append(y);masks.append(m)
    z,y,m=map(torch.cat,(zs,ys,masks))
    if not torch.isfinite(z).all():raise ValueError('Nonfinite validation predictions')
    # Compute one loss over the full verified set, not a mean of unequal batch means.
    loss=masked_loss(z,y,m,torch.ones_like(y)).item()
    metrics=data.evaluate_metrics(y.numpy(),z.sigmoid().numpy(),m.numpy())
    return loss,metrics,z.sigmoid().numpy()

def train_model(model,train,val,series,run_id,cfg,work,identity,resume=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    by={uid:g.sort_values(data.SID).to_dict('records') for uid,g in series.groupby(data.ID)}
    optimizer=torch.optim.Adam([{'params':model.backbone.parameters(),'lr':cfg['backbone_lr']},
                                {'params':model.head.parameters(),'lr':cfg['head_lr']}])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    stop=EarlyStop(3);rng=np.random.default_rng(cfg['seed']);history=[];start=0;best=None
    if resume:
        saved=torch.load(resume,map_location=device,weights_only=False) # only your own trusted checkpoint
        if saved['identity']!=identity:raise ValueError('Resume identity/config/data mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        history=saved['history'];start=saved['epoch'];best=saved['best']
        stop.best=best['val_loss'];stop.bad_epochs=saved['bad_epochs']
    deadline=time.monotonic()+cfg['max_hours']*3600;reason='max_epochs'
    try:
        if stop.bad_epochs>=3:reason='early_stopping'
        else:
            for epoch in range(start,cfg['epochs']):
                model.train();records=train.to_dict('records');order=rng.permutation(len(records));losses=[]
                optimizer.zero_grad(set_to_none=True)
                for step,idx in enumerate(order):
                    check_budget(deadline)
                    row=records[idx];images=study_images(row[data.ID],by,run_id,cfg,rng)
                    y,m,w=[v.to(device) for v in row_targets(row)]
                    window_start=(step//cfg['accumulate'])*cfg['accumulate']
                    window_size=min(cfg['accumulate'],len(order)-window_start)
                    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        logits=model(images,cfg['microbatch']);loss=masked_loss(logits,y,m,w)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    scaler.scale(loss/window_size).backward();losses.append(float(loss.detach()))
                    if (step+1)%cfg['accumulate']==0 or step+1==len(order):
                        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    if (step+1)%100==0:print(f'Epoch {epoch+1}: {step+1}/{len(order)} studies',flush=True)
                val_loss,metrics,pred=validate(model,val,by,run_id,cfg,device,deadline)
                improved,done=stop.update(val_loss)
                row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=val_loss,
                         macro_auroc=metrics['macro_auroc'],bad_epochs=stop.bad_epochs)
                history.append(row);print(row,flush=True)
                if improved:
                    best=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                              epoch=epoch+1,val_loss=val_loss,metrics=metrics)
                    data.atomic_torch(work/'best.pt',best)
                    frame=pd.DataFrame(pred,columns=data.TARGETS);frame.insert(0,data.ID,val[data.ID].tolist())
                    frame.to_csv(work/'validation_predictions_private.csv',index=False)
                scheduler.step(epoch+1)
                pd.DataFrame(history).to_csv(work/'history.csv',index=False)
                data.atomic_torch(work/'last.pt',dict(identity=identity,state=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,best=best,
                    bad_epochs=stop.bad_epochs,history=history,rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
                if done:reason='early_stopping';break
    except BudgetReached:
        reason='time_budget';print('Time budget reached; resume from last.pt (last fully validated epoch).',flush=True)
    if best is None:raise RuntimeError('No completed validated epoch; no model will be exported or uploaded')
    model.load_state_dict(best['state_dict']);model.eval()
    data.atomic_json(work/'training_status.json',dict(reason=reason,best_epoch=best['epoch'],best_val_loss=best['val_loss']))
    return best,history,reason

def export_model(model,best,cfg,index,identity,work,source_files):
    package=Path(work)/'model_package';package.mkdir(exist_ok=True)
    data.atomic_torch(package/'model.pt',dict(state_dict=best['state_dict'],architecture='resnet34',dropout=cfg['dropout']))
    for name,text in source_files.items():(package/name).write_text(text)
    data.atomic_json(package/'preprocessing_config.json',index['preprocessing'])
    # Validate saved weights on synthetic image bags without training records.
    loaded=ResNetKnee(pretrained=False,dropout=cfg['dropout']).eval()
    loaded.load_state_dict(torch.load(package/'model.pt',weights_only=True,map_location='cpu')['state_dict'],strict=True)
    device=next(model.parameters()).device
    gen=torch.Generator().manual_seed(71);images=[torch.rand(2,3,cfg['image_size'],cfg['image_size'],generator=gen)]
    with torch.inference_mode():
        expected=loaded(images).cpu();actual=model(images).float().cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-4)
    data.atomic_torch(package/'synthetic_smoke.pt',dict(images=images,expected=expected))
    import importlib.metadata
    versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','pandas','scipy','pydicom','scikit-learn']}
    (package/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    (package/'README.md').write_text('ResNet34 CNN knee MRI model. Load model.pt with cnn_runtime.ResNetKnee(pretrained=False).\n'
        'See manifest.json for native preprocessing, triplet slices, resize, normalization and aggregation.\n'
        'Not compatible with the DINOv2 submission notebook. Keep private. No labels or patient rows included.\n')
    manifest=dict(schema_version=1,model_family='resnet34-study-mean-max-v1',targets=data.TARGETS,configuration=cfg,
        training_identity=identity,best_epoch=best['epoch'],best_validation_loss=best['val_loss'],
        validation_metrics=best['metrics'],environment=versions,
        inference=dict(representation='adjacent-slice-triplet',resize='bilinear antialias full-FOV',
        image_size=cfg['image_size'],slices_per_series=cfg['slices_per_series'],sampling='rounded linspace',
        series_selection='all usable, sorted SeriesInstanceUID',slice_pool='mean+max',series_pool='mean',
        outputs='12 logits; sigmoid once',normalization_mean=[.485,.456,.406],normalization_std=[.229,.224,.225]),
        files={str(p.relative_to(package)):data.file_hash(p) for p in package.rglob('*') if p.is_file() and p.name!='manifest.json'})
    data.atomic_json(package/'manifest.json',manifest)
    return package
