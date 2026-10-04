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


def discover_bundle(explicit=None, input_root='/kaggle/input', dataset_handle='gany24558/rsna-knee-normalized-all-training-data'):
    if explicit:
        root = Path(explicit)
    else:
        candidates = []
        for p in find_input_files(input_root,'dataset_index.json'):
            try:
                index = json.loads(p.read_text())
                if index.get('dataset_handle') == dataset_handle:
                    candidates.append(p.parent)
            except (ValueError, OSError): pass
        root = select_one(candidates, 'DATASET_ROOT')
    if not (root / 'dataset_index.json').is_file():
        raise FileNotFoundError(f'Attach {dataset_handle}, with dataset_index.json')
    index = json.loads((root / 'dataset_index.json').read_text())
    if index.get('dataset_handle') != dataset_handle:
        raise ValueError(f'Expected dataset {dataset_handle}; found {index.get("dataset_handle")}')
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
        values = labels.loc[m, t].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Known target must be a finite probability in [0, 1]')
        if not labels.loc[g, t].isin([0, 1]).all(): raise ValueError('Official target must be binary')
        if (g & ~m).any() or (g & labels[t+'__weight'].ne(1)).any(): raise ValueError('Invalid verified override')
    any_gold = labels[[t+'__gold' for t in TARGETS]].any(axis=1)
    if not labels.has_gold.eq(any_gold.astype(int)).all(): raise ValueError('has_gold does not match per-target masks')
    if labels.loc[any_gold, 'fold'].lt(0).any(): raise ValueError('Verified studies need validation folds')


def smooth_soft_targets(target, epsilon=0.05):
    """Smooth probabilities once; missing values remain missing (not negatives)."""
    if not 0 <= epsilon < 0.5:
        raise ValueError('epsilon must be in [0, 0.5)')
    return target * (1 - 2 * epsilon) + epsilon


def apply_community_labels(labels, path, epsilon=0.05, weight=0.25, column_map=None):
    """Replace generated labels only. Keep official labels, folds and groups unchanged.

    column_map maps our target names to parquet column names. No positional guesses.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Attach rsna-knee-hpo-assets and set COMMUNITY_LABELS_PATH: {path}')
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Community label weight must be positive and finite')
    frame = pd.read_parquet(path)
    if ID not in frame and frame.index.name == ID:
        frame = frame.reset_index()
    mapping = column_map or {t: t for t in TARGETS}
    if set(mapping) != set(TARGETS) or len(set(mapping.values())) != len(TARGETS):
        raise ValueError('Provide one distinct community column per target')
    missing = {ID, *mapping.values()} - set(frame.columns)
    if missing:
        raise ValueError(f'Missing parquet columns {sorted(missing)}; available: {list(frame.columns)}. Set COMMUNITY_COLUMN_MAP explicitly.')
    if frame[ID].isna().any(): raise ValueError('Missing community study ID')
    frame[ID] = frame[ID].astype(str)
    if frame[ID].duplicated().any(): raise ValueError('Duplicate community study IDs')
    extra = set(frame[ID]) - set(labels[ID])
    if extra: raise ValueError(f'Community labels include {len(extra)} IDs outside official training data')
    if not len(frame): raise ValueError('Empty community label table')
    frame = frame.set_index(ID)
    out = labels.copy()
    for t in TARGETS:
        values = pd.to_numeric(frame[mapping[t]], errors='raise').astype(float)
        present = values.notna()
        if not np.isfinite(values[present]).all() or not values[present].between(0, 1).all():
            raise ValueError(f'Invalid community probabilities: {t}')
        aligned = values.reindex(out[ID]).to_numpy(float)
        gold = out[t+'__gold'].eq(1).to_numpy()
        known = np.isfinite(aligned) & ~gold
        out[t] = out[t].astype(float)
        out.loc[~gold, t] = smooth_soft_targets(aligned[~gold], epsilon)
        out.loc[~gold, t+'__mask'] = known[~gold].astype(int)
        out.loc[~gold, t+'__weight'] = np.where(known[~gold], weight, 0.)
        out.loc[~gold, t+'__source'] = np.where(known[~gold], 'community_soft', 'none')
        out.loc[~gold, t+'__state'] = np.where(known[~gold], 'soft', 'unmentioned')
    validate_labels(out)
    if not out[[t+'__mask' for t in TARGETS]].to_numpy(bool)[~out.has_gold.astype(bool)].any():
        raise ValueError('No usable community targets for non-gold studies')
    provenance = dict(filename=path.name, sha256=file_hash(path), epsilon=epsilon,
                      weight=weight, column_map=mapping, rows=len(frame),
                      missing_studies=len(set(labels[ID])-set(frame.index)))
    return out, provenance


def audit_labels(labels):
    rows = []
    for fold, block in labels.groupby('fold'):
        for t in TARGETS:
            m = block[t+'__mask'].eq(1) & block[t+'__weight'].gt(0)
            g = block[t+'__gold'].eq(1)
            rows.append(dict(fold=int(fold), target=t, studies=len(block), positive=int((m & block[t].eq(1)).sum()),
                             negative=int((m & block[t].eq(0)).sum()), soft=int((m & block[t].gt(0) & block[t].lt(1)).sum()), unknown=int((~m).sum()),
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

