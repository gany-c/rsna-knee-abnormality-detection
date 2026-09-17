"""Paste this into a Kaggle notebook cell and edit DATASET_ID.
Add KAGGLE_USERNAME and KAGGLE_KEY in Add-ons > Secrets and enable them
for this notebook (legacy API credentials from Kaggle account settings).
Run after the export cells, or interrupt extraction first to back up partial
checkpoints. A busy notebook kernel cannot execute this cell concurrently.
Use UPLOAD_MODE='create' once, then 'version' for later snapshots.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

DATASET_ID = 'YOUR_KAGGLE_USERNAME/rsna-knee-report-labels'
DATASET_TITLE = 'RSNA Knee Report Labels'
UPLOAD_MODE = 'create'  # 'version' after the first successful creation
SOURCE_DIR = Path('/kaggle/working/report_labels')


def stage_label_outputs(source, destination, dataset_id, title):
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f'No label output directory: {source}')
    destination.mkdir(parents=True, exist_ok=True)
    # Explicit allowlist excludes model weights, input reports, and credentials.
    csv_names = ['qwen_training_labels.csv', 'qwen_extractions.csv',
                 'qwen_errors.csv', 'qwen_validation.csv']
    copied = []
    for name in csv_names:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
            copied.append(name)
    checkpoint_files = sorted((source / 'cache').rglob('*.json'))
    records = [p for p in checkpoint_files if p.name != 'config.json']
    if not copied and not records:
        raise ValueError('Nothing to upload: run extraction first.')
    if checkpoint_files:
        with zipfile.ZipFile(destination / 'checkpoints.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in checkpoint_files:
                # Ignore .tmp files; the generation notebook writes completed JSON atomically.
                archive.write(path, path.relative_to(source).as_posix())
    manifest = {'dataset': dataset_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                'csv_files': copied, 'checkpoint_records': len(records),
                'note': 'CSVs reflect the last export; checkpoints can contain newer partial progress.'}
    (destination / 'backup_manifest.json').write_text(json.dumps(manifest, indent=2))
    metadata = {'id': dataset_id, 'title': title,
                'licenses': [{'name': 'other'}],
                'description': 'Private backup of RSNA knee report-derived weak labels and resume checkpoints. '
                               'Source competition data and report excerpts remain subject to the competition terms. '
                               'No additional redistribution rights are granted.'}
    (destination / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
    return manifest


def upload_label_outputs():
    if 'YOUR_KAGGLE_USERNAME' in DATASET_ID or not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]{3,50}', DATASET_ID):
        raise ValueError('Set DATASET_ID to your actual username/dataset-slug.')
    if UPLOAD_MODE not in {'create', 'version'}:
        raise ValueError("UPLOAD_MODE must be 'create' or 'version'.")
    if shutil.which('kaggle') is None:
        raise RuntimeError('Install the CLI in a separate cell: %pip install kaggle')
    # Pass secrets only to the subprocess environment; never write or print credentials.
    env = os.environ.copy()
    if not env.get('KAGGLE_API_TOKEN') and not (env.get('KAGGLE_USERNAME') and env.get('KAGGLE_KEY')):
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        try:
            env['KAGGLE_API_TOKEN'] = secrets.get_secret('KAGGLE_API_TOKEN')
        except Exception:
            try:
                env['KAGGLE_USERNAME'] = secrets.get_secret('KAGGLE_USERNAME')
                env['KAGGLE_KEY'] = secrets.get_secret('KAGGLE_KEY')
            except Exception:
                raise RuntimeError('Enable a KAGGLE_API_TOKEN secret, or both KAGGLE_USERNAME and KAGGLE_KEY secrets.') from None
    with tempfile.TemporaryDirectory(prefix='label-dataset-upload-') as directory:
        manifest = stage_label_outputs(SOURCE_DIR, directory, DATASET_ID, DATASET_TITLE)
        print('Backup:', len(manifest['csv_files']), 'CSVs and', manifest['checkpoint_records'], 'checkpoint records')
        cmd = ['kaggle', 'datasets', UPLOAD_MODE, '-p', directory, '--keep-tabular']
        if UPLOAD_MODE == 'version':
            cmd += ['-m', 'Label checkpoint backup ' + manifest['created_at']]
        # Creation is private by default: deliberately omit --public.
        # Versions retain the dataset's existing visibility and preserve old versions.
        subprocess.run(cmd, env=env, check=True)
    print('Upload submitted: https://www.kaggle.com/datasets/' + DATASET_ID)
    print('Check processing status below; retain local files until the dataset is ready.')
    subprocess.run(['kaggle', 'datasets', 'status', DATASET_ID], env=env, check=True)


if __name__ == '__main__':
    upload_label_outputs()
