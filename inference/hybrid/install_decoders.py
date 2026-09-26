# Run before the inference runtime imports pydicom. Requires a fresh session.
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

search_root = Path(WHEELHOUSE) if WHEELHOUSE is not None else INPUT_ROOT
wheels = []
for directory, subdirs, files in os.walk(search_root):
    subdirs[:] = [d for d in subdirs if d not in {'train_series', 'test_series', 'cache', 'features'}]
    wheels.extend(Path(directory)/name for name in files
                  if name.startswith('python_gdcm-3.2.6-') and name.endswith('.whl'))

if wheels:
    if len(wheels) != 1:
        raise ValueError('Multiple GDCM wheels found; set WHEELHOUSE to the intended wheel directory.')
    if 'pydicom.pixels.decoders.gdcm' in sys.modules:
        raise RuntimeError('Restart the Kaggle session, then Run All: pydicom has already cached decoder availability.')
    decoder_dir = Path('/kaggle/working/_dicom_deps')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index',
                           '--no-deps', '--upgrade', '--target', str(decoder_dir), str(wheels[0])])
    sys.path.insert(0, str(decoder_dir))
    import gdcm
    print('Offline GDCM installed:', gdcm.Version.GetVersion())
else:
    print('No attached GDCM wheel found. Existing decoders will be checked before inference.')
