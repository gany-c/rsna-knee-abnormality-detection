"""Offline RSNA frozen-DINO training runtime. Embedded verbatim in the notebook."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import copy, gc, hashlib, io, json, math, random, shutil, tarfile, time
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ID, SID = 'StudyInstanceUID', 'SeriesInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RUNTIME_VERSION = 'rsna-frozen-dino-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str))
    tmp.replace(path)


def atomic_torch(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(data, tmp); tmp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class SessionBudget:
    def __init__(self, hours):
        self.start = time.monotonic(); self.seconds = hours * 3600
    def expired(self, reserve=90):
        return time.monotonic() - self.start > self.seconds - reserve


def select_one(candidates, description):
    candidates = sorted(set(Path(p).resolve() for p in candidates))
    if len(candidates) != 1:
        raise ValueError(f'Set {description} explicitly: found {len(candidates)} candidates: {candidates}')
    return candidates[0]


def find_input_files(root, filename):
    """Discover lightweight manifests without traversing hundreds of thousands of DICOMs."""
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {'train_series','test_series','cache','features','.git','__pycache__'}]
        if filename in files:
            yield Path(current)/filename


def discover_bundle(explicit=None, input_root='/kaggle/input'):
    if explicit:
        root = Path(explicit)
    else:
        candidates = []
        for p in find_input_files(input_root,'dataset_index.json'):
            try:
                index = json.loads(p.read_text())
                if index.get('dataset_handle') == 'gany24558/rsna-knee-normalized-training-data':
                    candidates.append(p.parent)
            except (ValueError, OSError): pass
        root = select_one(candidates, 'DATASET_ROOT')
    if not (root / 'dataset_index.json').is_file():
        raise FileNotFoundError('Attach gany24558/rsna-knee-normalized-training-data, with dataset_index.json')
    return root


def safe_member(archive, name):
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Unsafe archive member: {name}')
    member = archive.getmember(name)
    if not member.isfile(): raise ValueError(f'Not a regular member: {name}')
    return member


def load_bundle(root):
    """Use only active, processed shards; never legacy archives or partial snapshots."""
    root = Path(root); index = json.loads((root / 'dataset_index.json').read_text())
    identity = index['identity']; expected = {f'{i:03d}' for i in range(int(identity['num_shards']))}
    if set(index['shards']) != expected:
        raise ValueError(f'Preprocessing is incomplete. Missing shards: {sorted(expected-set(index["shards"]))}')
    frames, raw_reference, labels, prep_reference = [], None, None, None
    for slot, entry in sorted(index['shards'].items()):
        if entry['status'] != 'processed': raise ValueError(f'Shard {slot} is partial; resume preprocessing first')
        if Path(entry['file']).name != entry['file']: raise ValueError('Invalid package filename')
        package = root / entry['file']
        with tarfile.open(package, 'r:') as arc:
            info = json.load(arc.extractfile(safe_member(arc, 'publication_manifest.json')))
            if info['status'] != 'processed' or {k: info[k] for k in identity} != identity:
                raise ValueError('Publication identity/status mismatch')
            prep = json.load(arc.extractfile(safe_member(arc, 'preprocessing_config.json')))
            if prep.get('run_id') != identity['run_id']: raise ValueError('Preprocessing config run mismatch')
            if prep_reference is not None and prep != prep_reference: raise ValueError('Preprocessing configs differ')
            prep_reference = prep
            raw = arc.extractfile(safe_member(arc, 'study_labels_and_folds.csv')).read()
            if hashlib.sha256(raw).hexdigest() != identity['label_sha256']:
                raise ValueError('Label checksum mismatch')
            if raw_reference is not None and raw != raw_reference: raise ValueError('Labels differ across shards')
            raw_reference = raw
            if labels is None:
                labels = pd.read_csv(io.BytesIO(raw), dtype={ID: str, 'patient_group': str})
            frame = pd.read_csv(arc.extractfile(safe_member(arc, f'training_series_shard_{int(slot):03d}.csv')),
                                dtype={ID: str, SID: str})
            if len(frame) and not frame.status.eq('ok').all(): raise ValueError('Training series contain failures')
            # Direct seek offsets avoid repeatedly scanning tar archives during extraction.
            offsets, sizes = [], []
            for member in frame.cache_path:
                item = safe_member(arc, member); offsets.append(item.offset_data); sizes.append(item.size)
            frame['archive_path'] = str(package); frame['offset'] = offsets; frame['nbytes'] = sizes
            frames.append(frame)
    series = pd.concat(frames, ignore_index=True).sort_values([ID, SID]).reset_index(drop=True)
    if series.empty or series.duplicated([ID, SID]).any(): raise ValueError('Empty or duplicated training series')
    if not labels[ID].is_unique: raise ValueError('Duplicate study labels')
    if not set(series[ID]).issubset(set(labels[ID])): raise ValueError('Series with no label record')
    validate_labels(labels)
    eligible = labels.loc[labels[ID].isin(series[ID])].copy().sort_values(ID).reset_index(drop=True)
    index = dict(index, preprocessing=prep_reference)
    return series, eligible, labels, index


def validate_labels(labels):
    required = {ID, 'fold', 'patient_group', 'has_gold'}
    required |= {t+s for t in TARGETS for s in ['', '__mask', '__weight', '__gold']}
    if required - set(labels): raise ValueError(f'Missing label columns: {sorted(required-set(labels))}')
    if labels.patient_group.isna().any() or labels.patient_group.str.strip().eq('').any():
        raise ValueError('Missing grouping identity')
    folds = pd.to_numeric(labels.fold, errors='raise').to_numpy(float)
    if not np.isfinite(folds).all() or not np.equal(folds, folds.astype(int)).all() or (folds < -1).any():
        raise ValueError('Invalid fold values')
    if labels.groupby('patient_group').fold.nunique().gt(1).any(): raise ValueError('A group crosses folds')
    for t in TARGETS:
        for s in ['__mask', '__gold']:
            if not labels[t+s].isin([0, 1]).all(): raise ValueError('Invalid label mask')
        m = labels[t+'__mask'].eq(1); g = labels[t+'__gold'].eq(1)
        w = labels[t+'__weight'].to_numpy(float)
        if not np.isfinite(w).all() or (w < 0).any(): raise ValueError('Invalid reliability weights')
        if not labels.loc[m, t].isin([0, 1]).all(): raise ValueError('Known target is not binary')
        if (g & ~m).any() or (g & labels[t+'__weight'].ne(1)).any(): raise ValueError('Invalid verified override')
    any_gold = labels[[t+'__gold' for t in TARGETS]].any(axis=1)
    if not labels.has_gold.eq(any_gold.astype(int)).all(): raise ValueError('has_gold does not match per-target masks')
    if labels.loc[any_gold, 'fold'].lt(0).any(): raise ValueError('Verified studies need validation folds')


def audit_labels(labels):
    rows = []
    for fold, block in labels.groupby('fold'):
        for t in TARGETS:
            m = block[t+'__mask'].eq(1) & block[t+'__weight'].gt(0)
            g = block[t+'__gold'].eq(1)
            rows.append(dict(fold=int(fold), target=t, studies=len(block), positive=int((m & block[t].eq(1)).sum()),
                             negative=int((m & block[t].eq(0)).sum()), unknown=int((~m).sum()),
                             verified_positive=int((g & block[t].eq(1)).sum()),
                             verified_negative=int((g & block[t].eq(0)).sum())))
    return pd.DataFrame(rows)


def read_native(record, expected_run):
    with open(record['archive_path'], 'rb') as f:
        f.seek(int(record['offset'])); payload = f.read(int(record['nbytes']))
    if len(payload) != int(record['nbytes']): raise ValueError('Truncated archive')
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        image = z['image']; meta = json.loads(str(z['meta'].item()))
        support = np.unpackbits(z['valid_bits'], count=image.size).reshape(image.shape).astype(bool)
    if image.dtype != np.float16 or image.shape != (64, 320, 320) or list(image.shape) != meta['shape']:
        raise ValueError('Expected preprocessing contract float16 [64,320,320]')
    if meta.get('representation') != 'native-plane-full-fov-v3' or meta.get('run_id') != expected_run:
        raise ValueError('Wrong normalization identity/representation')
    if meta.get(ID) != record[ID] or meta.get(SID) != record[SID]: raise ValueError('Cache ID mismatch')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1: raise ValueError('Invalid pixel range')
    n = int(meta['selected_count'])
    if n <= 0 or n > len(image) or support[n:].any(): raise ValueError('Invalid acquired-slice mask')
    valid = np.flatnonzero(support[:n].reshape(n, -1).any(axis=1))
    positions = np.asarray(meta['slice_positions_mm'], dtype=np.float32)
    spacing = np.asarray(meta['spacing_summary_drc_mm'], dtype=np.float32)
    if len(valid) != n or len(positions) != n or not np.isfinite(positions).all(): raise ValueError('Invalid slice positions')
    if n > 1 and not (np.diff(positions) > 0).all(): raise ValueError('Unordered slice positions')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any(): raise ValueError('Invalid spacing')
    return image, support, meta


def discover_encoder(explicit=None, backend='auto', input_root='/kaggle/input'):
    if explicit:
        p = Path(explicit)
        chosen = ('hf' if p.is_dir() else 'timm') if backend == 'auto' else backend
        return p, chosen
    candidates = []
    for p in find_input_files(input_root,'config.json'):
        try:
            c = json.loads(p.read_text())
            if c.get('model_type') == 'dinov2' and c.get('hidden_size') == 384:
                if any(p.parent.glob('*.safetensors')) or (p.parent/'pytorch_model.bin').exists():
                    candidates.append((p.parent, 'hf'))
        except (ValueError, OSError): pass
    for p in find_input_files(input_root,'dinov2_vits14_pretrain.pth'):
        candidates.append((p, 'timm'))
    if backend != 'auto': candidates = [(p,b) for p,b in candidates if b == backend]
    if len(candidates) != 1: raise ValueError(f'Set ENCODER_PATH/BACKEND explicitly. Found: {candidates}')
    return candidates[0]


def encoder_source_hash(path):
    path = Path(path)
    if path.is_file(): return file_hash(path)
    selected = sorted(p for p in path.iterdir() if p.is_file() and
                      (p.suffix in {'.json', '.safetensors'} or p.name == 'pytorch_model.bin'))
    if not selected: raise ValueError('Empty encoder folder')
    return fingerprint({p.name: file_hash(p) for p in selected})


class FrozenEncoder(nn.Module):
    def __init__(self, path, backend, size=322):
        super().__init__(); self.backend = backend; self.size = size
        if backend == 'hf':
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            c = self.encoder.config
            if c.model_type != 'dinov2' or c.hidden_size != 384 or c.patch_size != 14 or c.num_hidden_layers != 12:
                raise ValueError('Attach the generic DINOv2 Small without registers or a classifier')
        elif backend == 'timm':
            import timm
            from timm.models.vision_transformer import checkpoint_filter_fn
            self.encoder = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                                              num_classes=0, img_size=size)
            state = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint_filter_fn(state, self.encoder)
            self.encoder.load_state_dict(state, strict=True)
        else: raise ValueError('Unknown encoder backend')
        self.requires_grad_(False); self.eval()
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self, images, support):
        # Deliberately no crop/rescale: preserve 320px cache, add a one-pixel border.
        if images.shape[1:] != (3,320,320): raise ValueError('Unexpected encoder input')
        images = F.pad(images, (1,1,1,1)); support = F.pad(support, (1,1,1,1))
        x = (images-self.mean)/self.std
        tokens = self.encoder(pixel_values=x).last_hidden_state if self.backend=='hf' else self.encoder.forward_features(x)
        if tokens.shape[1:] != (530,384): raise ValueError('Unexpected DINO token layout')
        weights = F.avg_pool2d(support.float(),14,14).flatten(1)
        pooled = (tokens[:,1:].float()*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-6)
        return torch.cat([tokens[:,0].float(),pooled],1)

    @classmethod
    def from_export(cls, folder):
        folder = Path(folder); adapter = json.loads((folder/'adapter.json').read_text())
        if adapter['backend'] == 'hf': return cls(folder,'hf',adapter['size'])
        # Export is already converted/resized: use the exact architecture, no checkpoint filtering.
        import timm
        model = cls.__new__(cls); nn.Module.__init__(model)
        model.backend = 'timm'; model.size = adapter['size']
        model.encoder = timm.create_model(adapter['architecture'],pretrained=False,num_classes=0,img_size=model.size)
        model.encoder.load_state_dict(torch.load(folder/'encoder.pt',map_location='cpu',weights_only=True),strict=True)
        model.register_buffer('mean',torch.tensor(adapter['mean']).view(1,3,1,1))
        model.register_buffer('std',torch.tensor(adapter['std']).view(1,3,1,1))
        model.requires_grad_(False);model.eval()
        return model

    def export(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        if self.backend == 'hf': self.encoder.save_pretrained(folder, safe_serialization=True)
        else: atomic_torch(folder/'encoder.pt', self.encoder.state_dict())
        atomic_json(folder/'adapter.json',dict(backend=self.backend,size=self.size,
                    architecture='vit_small_patch14_dinov2.lvd142m',converted_state=True,
                    mean=[.485,.456,.406],std=[.229,.224,.225],patch_size=14))


def make_slice_inputs(image, support, centers, representation='single'):
    n = int(support.reshape(len(support),-1).any(1).sum())
    if representation == 'single': indices = np.repeat(np.asarray(centers)[:,None],3,1)
    elif representation == 'triplet': indices = np.clip(np.asarray(centers)[:,None]+np.array([-1,0,1]),0,n-1)
    else: raise ValueError('Choose single or triplet')
    pixels = torch.from_numpy(image[indices].astype(np.float32))
    # Average support across the input channels before patch weighting.
    valid = torch.from_numpy(support[indices].mean(1,keepdims=True).astype(np.float32))
    return pixels, valid


def protocol_codes(record):
    plane = {'Sagittal':0,'Coronal':1,'Axial':2}.get(str(record.get('Anatomical_Plane')),3)
    def binary(v):
        if pd.isna(v): return 2
        return int(float(v)) if str(v) in ['0','1','0.0','1.0'] else 2
    return np.array([plane,binary(record.get('Fluid_Sensitive')),binary(record.get('Fat_Suppression'))],np.int64)


def feature_name(record):
    return fingerprint([record[ID],record[SID]]) + '.npz'


def check_features(path, record, feature_id):
    with np.load(path, allow_pickle=False) as z:
        if str(z['feature_id'].item()) != feature_id or str(z['study'].item()) != record[ID] or str(z['series'].item()) != record[SID]:
            raise ValueError('Feature cache identity mismatch')
        f = z['features']; positions = z['positions']; protocol = z['protocol']; spacing = z['spacing']
        if f.ndim != 2 or f.shape[1] != 768 or not 1 <= len(f) <= 64 or not np.isfinite(f).all():
            raise ValueError('Invalid cached features')
        if positions.shape != (len(f),) or not np.isfinite(positions).all(): raise ValueError('Bad feature positions')
        if protocol.shape != (3,) or not np.array_equal(protocol,protocol_codes(record)): raise ValueError('Protocol mismatch')
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing<=0).any(): raise ValueError('Bad spacing')
    return path


def extract_features(series, encoder, identity, feature_identity, work, resume_roots, cfg, budget):
    feature_id = fingerprint(feature_identity)
    folder = Path(work)/'features'/feature_id; folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'identity.json',feature_identity)
    sources = [folder] + [Path(p)/'features'/feature_id for p in resume_roots]
    device = next(encoder.parameters()).device
    records, completed = series.to_dict('records'), []
    for i, r in enumerate(records):
        name = feature_name(r)
        existing = next((p/name for p in sources if (p/name).is_file()),None)
        if existing is not None:
            check_features(existing,r,feature_id)
            # Carry prior features into this output so the next resume needs only this cumulative run.
            target = folder/name
            if existing.resolve() != target.resolve():
                if shutil.disk_usage(work).free < cfg['min_free_gb']*1e9 + existing.stat().st_size:
                    raise OSError('Insufficient disk to retain resumed features')
                temp = target.with_suffix('.tmp'); shutil.copy2(existing,temp); temp.replace(target)
            completed.append(str(target)); continue
        if budget.expired():
            atomic_json(Path(work)/'status.json',dict(status='FEATURES_PARTIAL',completed=len(completed),total=len(records),feature_id=feature_id))
            print('Session budget reached. Save this PRIVATE output and attach it on the next run.'); return None
        if shutil.disk_usage(work).free < cfg['min_free_gb']*1e9: raise OSError('Insufficient free disk for features')
        image,support,meta = read_native(r,identity['run_id'])
        centers = np.arange(int(meta['selected_count']))  # all real candidates; training samples later
        chunks=[]
        for j in range(0,len(centers),cfg['image_batch']):
            x,v = make_slice_inputs(image,support,centers[j:j+cfg['image_batch']],cfg['representation'])
            ctx = torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
            with torch.inference_mode(),ctx:
                f = encoder(x.to(device),v.to(device)).float().cpu().numpy()
            if not np.isfinite(f).all(): raise ValueError('Encoder produced nonfinite features')
            chunks.append(f.astype(np.float16))
        target=folder/name; tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as out:
            np.savez_compressed(out,feature_id=np.array(feature_id),study=np.array(r[ID]),series=np.array(r[SID]),
                features=np.concatenate(chunks),positions=np.asarray(meta['slice_positions_mm'],np.float32),
                protocol=protocol_codes(r),spacing=np.asarray(meta['spacing_summary_drc_mm'],np.float32))
        tmp.replace(target); check_features(target,r,feature_id); completed.append(str(target))
        if i%25==0 or i+1==len(records): print(f'Features {i+1}/{len(records)}; elapsed {(time.monotonic()-budget.start)/60:.1f} min',flush=True)
    result=series.copy(); result['feature_path']=completed
    result.to_csv(Path(work)/'feature_series.csv',index=False)
    return result


def sample_centers(n,count,rng=None):
    if n<=count: return np.arange(n)
    if rng is None: return np.linspace(0,n-1,count).round().astype(int)
    bins=np.array_split(np.arange(n),count)
    return np.asarray([rng.choice(b) for b in bins])


class FeatureStudies:
    """Compact arrays in RAM; raw MRI volumes are never stacked across studies."""
    def __init__(self,series,labels,max_ram_gb=5):
        self.labels=labels.set_index(ID).loc[sorted(labels[ID])].reset_index()
        self.ids=self.labels[ID].tolist(); self.items=[]; estimated=0
        by={uid:g for uid,g in series.groupby(ID)}
        for uid in self.ids:
            features=[];protocol=[];spacing=[]
            for _,r in by[uid].sort_values(SID).iterrows():
                with np.load(r.feature_path,allow_pickle=False) as z:
                    f=z['features'].copy();p=z['protocol'].copy();s=z['spacing'].copy()
                estimated+=f.nbytes
                if estimated>max_ram_gb*1e9: raise MemoryError('Feature RAM budget exceeded; increase max_feature_ram_gb if available')
                features.append(f);protocol.append(p);spacing.append(s)
            if not features: raise ValueError('Empty study')
            self.items.append((features,np.stack(protocol),np.stack(spacing)))
        self.y=np.nan_to_num(self.labels[TARGETS].to_numpy(np.float32),nan=0)
        self.mask=self.labels[[t+'__mask' for t in TARGETS]].to_numpy(bool)
        self.gold=self.labels[[t+'__gold' for t in TARGETS]].to_numpy(bool)
        self.weights=self.labels[[t+'__weight' for t in TARGETS]].to_numpy(np.float32)
        self.groups=self.labels.patient_group.to_numpy(str);self.folds=self.labels.fold.to_numpy(int)
        print(f'Loaded {len(self.ids)} studies; feature payload {estimated/1e9:.2f} GB')

    def split(self,fold):
        val=np.flatnonzero((self.folds==fold)&self.gold.any(1))
        held=set(self.groups[self.folds==fold])
        train=np.flatnonzero(~np.isin(self.groups,list(held)) & (self.mask & (self.weights>0)).any(1))
        if not len(val) or not len(train): raise ValueError(f'Empty train or verified validation for fold {fold}')
        if set(self.groups[train])&set(self.groups[val]): raise ValueError('Group leakage')
        return train,val

    def scaler(self,indices):
        x=np.concatenate([np.log(self.items[i][2]) for i in indices])
        return dict(mean=x.mean(0).tolist(),std=np.maximum(x.std(0),1e-3).tolist(),transform='log_mm')

    def batch(self,indices,scaler,centers,device,rng=None):
        count=max(len(self.items[i][0]) for i in indices);b=len(indices)
        x=np.zeros((b,count,1536),np.float32);p=np.zeros((b,count,3),np.int64)
        s=np.zeros((b,count,3),np.float32);m=np.zeros((b,count),bool)
        for j,i in enumerate(indices):
            fs,protocol,spacing=self.items[i];n=len(fs)
            for k,f in enumerate(fs):
                selected=f[sample_centers(len(f),centers,rng)].astype(np.float32)
                x[j,k]=np.concatenate([selected.mean(0),selected.max(0)])
            p[j,:n]=protocol;s[j,:n]=(np.log(spacing)-np.asarray(scaler['mean']))/np.asarray(scaler['std']);m[j,:n]=True
        return tuple(torch.as_tensor(v,device=device) for v in [x,p,s,m])


class FindingAttention(nn.Module):
    def __init__(self,hidden=256,attention_dropout=.1,dropout=.15):
        super().__init__();self.hidden=hidden
        self.proj=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,hidden),nn.GELU())
        self.plane=nn.Embedding(4,hidden);self.fluid=nn.Embedding(3,hidden);self.fat=nn.Embedding(3,hidden)
        self.spacing=nn.Linear(3,hidden)
        self.query=nn.Parameter(torch.randn(12,hidden)*.02)
        self.attn=nn.MultiheadAttention(hidden,4,dropout=attention_dropout,batch_first=True)
        self.fuse=nn.Sequential(nn.LayerNorm(5*hidden),nn.Linear(5*hidden,hidden),nn.GELU(),nn.Dropout(dropout))
        self.weight=nn.Parameter(torch.randn(12,hidden)*.03);self.bias=nn.Parameter(torch.zeros(12))

    def forward(self,x,protocol,spacing,mask,series_dropout=0.):
        mask=mask.bool()
        if not mask.any(1).all(): raise ValueError('Cannot predict an all-empty study')
        if self.training and series_dropout:
            keep=(torch.rand_like(mask.float())>=series_dropout)&mask
            empty=~keep.any(1); first=mask.float().argmax(1)
            keep[torch.arange(len(mask),device=mask.device)[empty],first[empty]]=True;mask=keep
        h=self.proj(x)+self.plane(protocol[:,:,0])+self.fluid(protocol[:,:,1])+self.fat(protocol[:,:,2])+self.spacing(spacing)
        h=h.masked_fill(~mask[:,:,None],0)
        mean=h.sum(1,keepdim=True)/mask.sum(1)[:,None,None]
        q=self.query[None].expand(len(h),-1,-1)
        a,_=self.attn(q,h,h,key_padding_mask=~mask,need_weights=False)
        g=mean.expand_as(a)
        f=self.fuse(torch.cat([a,q,g,(a-g).abs(),a*g],-1))
        return (f*self.weight[None]).sum(-1)+self.bias


def masked_bce(logits,targets,mask,weights):
    use=mask.bool() & (weights>0)
    clean=torch.where(use,targets,torch.zeros_like(targets))
    w=torch.where(use,weights,torch.zeros_like(weights))
    losses=F.binary_cross_entropy_with_logits(logits,clean,reduction='none')
    denom=w.sum(0);active=denom>0
    if not active.any(): return logits.sum()*0
    return ((losses*w).sum(0)[active]/denom[active]).mean()


def ranking_loss(logits,y,gold):
    result=[]
    for t in range(12):
        pos=logits[gold[:,t] & (y[:,t]==1),t];neg=logits[gold[:,t] & (y[:,t]==0),t]
        if len(pos) and len(neg): result.append(F.softplus(-(pos[:,None]-neg[None,:])).mean())
    return torch.stack(result).mean() if result else logits.sum()*0


def ema_update(ema,model,decay):
    with torch.no_grad():
        for ep,p in zip(ema.parameters(),model.parameters()): ep.mul_(decay).add_(p,alpha=1-decay)
        for eb,b in zip(ema.buffers(),model.buffers()): eb.copy_(b)


def evaluate_metrics(y,p,gold):
    rows=[]
    for t,name in enumerate(TARGETS):
        v=gold[:,t];truth=y[v,t];score=p[v,t];positive=int((truth==1).sum());negative=int((truth==0).sum())
        auc=float(roc_auc_score(truth,score)) if positive and negative else None
        ap=float(average_precision_score(truth,score)) if positive and negative else None
        rows.append(dict(target=name,n=int(v.sum()),positive=positive,negative=negative,auroc=auc,average_precision=ap))
    aucs=[r['auroc'] for r in rows if r['auroc'] is not None]
    return dict(macro_auroc=float(np.mean(aucs)) if aucs else None,targets=rows,
                note='Average precision uses sklearn; it is not trapezoidal PR-AUC.')


@torch.inference_mode()
def predict_head(model,data,indices,scaler,cfg,device):
    model.eval();logits=[]
    for start in range(0,len(indices),cfg['batch_size']):
        ix=indices[start:start+cfg['batch_size']]
        logits.append(model(*data.batch(ix,scaler,cfg['centers'],device)).cpu())
    z=torch.cat(logits)
    if not torch.isfinite(z).all(): raise ValueError('Nonfinite validation logits')
    loss=masked_bce(z,torch.from_numpy(data.y[indices]),torch.from_numpy(data.gold[indices]),torch.ones_like(z))
    return float(loss),z.sigmoid().numpy()


def make_epoch_pool(data,train,cfg,rng,refine=False):
    gold=train[data.gold[train].any(1)]
    weak=train[~data.gold[train].any(1)]
    if refine or not len(weak):
        return rng.permutation(gold) if len(gold) else rng.permutation(train)
    rng.shuffle(weak)
    if not len(gold): return weak
    nw=max(1,round(cfg['batch_size']*(1-cfg['verified_fraction'])))
    chunks=[]
    for start in range(0,len(weak),nw):
        w=weak[start:start+nw]
        ng=max(1,round(len(w)*cfg['verified_fraction']/(1-cfg['verified_fraction'])))
        chunk=np.concatenate([w,rng.choice(gold,ng,replace=True)]);rng.shuffle(chunk);chunks.append(chunk)
    return np.concatenate(chunks)


def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train_fold(data,fold,cfg,work,resume_roots,training_id,budget,device):
    train,val=data.split(fold);scaler=data.scaler(train)
    seed_all(cfg['seed']+fold)
    model=FindingAttention(cfg['hidden']).to(device);ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    folder=Path(work)/'folds';folder.mkdir(exist_ok=True,parents=True)
    name=f'fold_{fold}.pt';target=folder/name
    candidates=[target]+[Path(r)/'folds'/name for r in resume_roots]
    previous=next((p for p in candidates if p.exists()),None)
    history=[];best_loss=float('inf');best_state=None;best_kind=None;best_epoch=None;bad=0;start_epoch=0;phase='mixed'
    if previous:
        saved=torch.load(previous,map_location='cpu',weights_only=True)
        if saved['training_id']!=training_id: raise ValueError(f'Checkpoint configuration mismatch: {previous}')
        if saved['complete']:
            model.load_state_dict(saved['best']);loss,pred=predict_head(model,data,val,saved['scaler'],cfg,device)
            if previous!=target: shutil.copy2(previous,target)
            return saved,pred,val
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best_loss=saved['best_loss'];best_state=saved['best'];best_kind=saved['best_kind'];best_epoch=saved['best_epoch']
        bad=saved['bad'];start_epoch=saved['next_epoch'];phase=saved['phase']
    stages=['mixed']+(['refine'] if cfg['refine_epochs'] else [])
    start_stage=stages.index(phase)
    saved=None
    for stage in stages[start_stage:]:
        if stage!=phase:
            model.load_state_dict(best_state);ema.load_state_dict(best_state)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['refine_lr'],weight_decay=cfg['weight_decay'])
            start_epoch=0;bad=0;phase=stage
        epochs=cfg['epochs'] if stage=='mixed' else cfg['refine_epochs']
        for epoch in range(start_epoch,epochs):
            if bad >= cfg['patience']: break
            if budget.expired(): return None,None,None
            seed_all(cfg['seed']+fold*10000+epoch+(1000 if stage=='refine' else 0))
            rng=np.random.default_rng(cfg['seed']+fold*10000+epoch+(1000 if stage=='refine' else 0))
            pool=make_epoch_pool(data,train,cfg,rng,stage=='refine');model.train();total=0.;steps=0
            for start in range(0,len(pool),cfg['batch_size']):
                ix=pool[start:start+cfg['batch_size']]
                if budget.expired(45): return None,None,None  # Previous epoch checkpoint remains resumable.
                y=torch.as_tensor(data.y[ix],device=device);gold=torch.as_tensor(data.gold[ix],device=device)
                m=gold if stage=='refine' else torch.as_tensor(data.mask[ix],device=device)
                w=torch.ones_like(y) if stage=='refine' else torch.as_tensor(data.weights[ix],device=device)
                optimizer.zero_grad(set_to_none=True)
                z=model(*data.batch(ix,scaler,cfg['centers'],device,rng),series_dropout=cfg['series_dropout'])
                loss=masked_bce(z,y,m,w)+cfg['rank_lambda']*ranking_loss(z,y,gold)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();ema_update(ema,model,cfg['ema_decay'])
                total+=float(loss.detach());steps+=1
            current_loss,current_pred=predict_head(model,data,val,scaler,cfg,device)
            ema_loss,ema_pred=predict_head(ema,data,val,scaler,cfg,device)
            use_ema=ema_loss<current_loss;score=ema_loss if use_ema else current_loss
            pred=ema_pred if use_ema else current_pred
            if score<best_loss-1e-6:
                best_loss=score;best_state=cpu_state(ema if use_ema else model);best_kind='ema' if use_ema else 'current';best_epoch=f'{stage}:{epoch+1}';bad=0
            else: bad+=1
            metric=evaluate_metrics(data.y[val],pred,data.gold[val])
            row=dict(phase=stage,epoch=epoch+1,train_loss=total/max(steps,1),validation_loss=score,
                     macro_auroc=metric['macro_auroc'],best_loss=best_loss)
            history.append(row);print(f'Fold {fold}: {row}',flush=True)
            saved=dict(training_id=training_id,fold=int(fold),complete=False,model=cpu_state(model),ema=cpu_state(ema),
                       optimizer=optimizer.state_dict(),best=best_state,best_loss=best_loss,best_kind=best_kind,best_epoch=best_epoch,
                       history=history,bad=bad,next_epoch=epoch+1,phase=stage,scaler=scaler)
            atomic_torch(target,saved)
            if bad>=cfg['patience']: break
        start_epoch=0
    if saved is None and previous: saved=torch.load(previous,map_location='cpu',weights_only=True)
    if saved is None: raise ValueError('No training epoch completed')
    saved['complete']=True;atomic_torch(target,saved)
    model.load_state_dict(best_state);loss,pred=predict_head(model,data,val,scaler,cfg,device)
    return saved,pred,val


def export_package(encoder,data,results,cfg,identity,feature_identity,training_id,work,runtime_source,preprocessing_source,preprocessing_config):
    package=Path(work)/'model_package';package.mkdir(exist_ok=True)
    encoder.export(package/'encoder')
    folds=[];oof=np.full_like(data.y,np.nan);rows=[]
    for saved,pred,val in results:
        fold=saved['fold'];name=f'head_fold_{fold}.pt'
        atomic_torch(package/name,dict(state_dict=saved['best'],scaler=saved['scaler'],fold=fold,hidden=cfg['hidden']))
        oof[val]=pred;folds.append(dict(fold=fold,file=name,sha256=file_hash(package/name),best_epoch=saved['best_epoch'],
                                      weight_type=saved['best_kind'],validation_loss=saved['best_loss']))
        rows.append(dict(fold=fold,**evaluate_metrics(data.y[val],pred,data.gold[val])))
    gold_rows=data.gold.any(1);covered=np.isfinite(oof).all(1)&gold_rows
    frame=pd.DataFrame(oof,columns=TARGETS);frame.insert(0,ID,data.ids);frame['fold']=data.folds
    frame.to_csv(Path(work)/'oof_private.csv',index=False)
    metrics=dict(per_fold=rows,pooled=evaluate_metrics(data.y[covered],oof[covered],data.gold[covered]),
                 covered_verified=int(covered.sum()),eligible_verified=int(gold_rows.sum()),
                 complete_oof=bool(covered[gold_rows].all()))
    atomic_json(Path(work)/'validation_private.json',metrics)
    # Synthetic inputs only: exported smoke reference contains no patient image/features/IDs.
    seed_all(991);test=dict(x=torch.randn(2,3,1536),protocol=torch.zeros(2,3,3,dtype=torch.long),
                          spacing=torch.zeros(2,3,3),mask=torch.tensor([[True,True,False],[True,False,False]]))
    check=FindingAttention(cfg['hidden']).eval();check.load_state_dict(results[0][0]['best'])
    with torch.no_grad():test['expected']=check(test['x'],test['protocol'],test['spacing'],test['mask'])
    atomic_torch(package/'synthetic_head_smoke.pt',test)
    (package/'knee_runtime.py').write_text(runtime_source)
    (package/'native_preprocessing.py').write_text(preprocessing_source)
    atomic_json(package/'preprocessing_config.json',preprocessing_config)
    import importlib.metadata
    versions={}
    for name in ['torch','numpy','pandas','scikit-learn','transformers','timm','safetensors','scipy','pydicom']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    (package/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    manifest=dict(schema_version=1,runtime_version=RUNTIME_VERSION,training_id=training_id,targets=TARGETS,
                  dataset_identity=identity,feature_identity=feature_identity,configuration=cfg,folds=folds,
                  environment=versions,grouping='study-separated' if np.array_equal(data.groups,np.asarray(data.ids)) else 'provided group-separated; patient identity must be audited',
                  complete_oof=metrics['complete_oof'],source='generic public DINOv2; no task-adapted encoder',
                  model_scope='frozen encoder plus attention head; encoder fine-tuning not implemented',
                  inference=dict(centers=cfg['centers'],sampling='rounded linspace over valid slices',ensemble='mean sigmoid',
                                 input_shape=[64,320,320],padding=[1,1,1,1],feature_pool='CLS + support-weighted patch mean; series mean+max',
                                 missing_study_policy='error; no silently substituted predictions'),
                  privacy='Package excludes source scans, reports, cached features, labels, IDs and OOF rows. Do not publish the whole training output.',
                  validation_limitations='Small verified set already informed label development; no independent test claim.')
    (package/'README.md').write_text('Frozen DINOv2 Small knee MRI model package. See manifest.json for the input contract.\n'
        'Heads use fold-specific log-spacing scalers. Generic encoder is shared. Average sigmoid outputs.\n'
        'Reference preparation code is native_preprocessing.py; no test-label or report input.\n'
        'This package is not a submission notebook and contains no submission.csv.\n'
        'Review source competition and encoder licenses before distribution; no automatic upload occurs.\n')
    manifest['files']={str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file() and p.name!='manifest.json'}
    atomic_json(package/'manifest.json',manifest)
    return package,metrics
