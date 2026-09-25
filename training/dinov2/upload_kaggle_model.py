# Optional final notebook cell: run after Export and verify.
# Enable Internet in this TRAINING notebook's settings before uploading.
# Set True and run only this cell; no training is repeated.
UPLOAD_TO_KAGGLE = False

from pathlib import Path
import hashlib
import json

if not UPLOAD_TO_KAGGLE:
    print('Upload skipped. Set UPLOAD_TO_KAGGLE = True to save the package to Kaggle Models.')
else:
    import kagglehub

    # PACKAGE is set by the export cell. The fallback supports this completed run.
    upload_package = Path(globals().get(
        'PACKAGE',
        '/kaggle/working/rsna-knee-training/f0121bad6f0c529e/model_package'
    )).resolve()
    upload_manifest = json.loads((upload_package / 'manifest.json').read_text())
    if not upload_manifest.get('complete_oof'):
        raise RuntimeError('This upload cell requires the completed five-fold package.')

    # Verify the export and refuse unexpected files in the upload directory.
    expected_files = set(upload_manifest['files']) | {'manifest.json'}
    actual_files = {str(p.relative_to(upload_package))
                    for p in upload_package.rglob('*') if p.is_file()}
    if actual_files != expected_files:
        raise RuntimeError('Package file list differs from its export manifest.')
    for relative_name, expected_hash in upload_manifest['files'].items():
        source_file = upload_package / relative_name
        if source_file.is_symlink() or not source_file.resolve().is_relative_to(upload_package):
            raise RuntimeError('Unsafe package path: ' + relative_name)
        digest = hashlib.sha256()
        with source_file.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError('Package checksum mismatch: ' + relative_name)

    # New models are private by default. Keep this model private in Kaggle.
    # Existing models retain their visibility; check it before uploading again.
    model_handle = (
        'gany24558/rsna-knee-dinov2-' + upload_manifest['training_id']
        + '/pyTorch/five-fold-attention'
    )
    print('Uploading model package to:', model_handle)
    # Kaggle notebooks normally authenticate automatically. Never paste tokens here.
    # Re-running a successful upload creates another version of this variation.
    kagglehub.model_upload(
        model_handle,
        str(upload_package),
        version_notes='Frozen DINOv2 Small and five attention heads; training run '
                      + upload_manifest['training_id'],
    )
    print('Upload accepted. Kaggle may need time to process the files.')
    print('Model page: https://www.kaggle.com/models/' + model_handle)
