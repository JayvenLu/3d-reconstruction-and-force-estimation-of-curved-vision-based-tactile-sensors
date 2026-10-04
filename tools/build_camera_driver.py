"""Optional read-only driver inspector; built locally, no environment installation."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
if os.name != 'nt':
    raise SystemExit('The DirectShow camera inspector is Windows-only.')
os.chdir(ROOT)
(ROOT / '.cache').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / '.cache')
from setuptools import setup, Extension
import pybind11
if len(sys.argv) == 1:
    sys.argv += ['build_ext', '--inplace', '--build-temp', str(ROOT / '.cache' / 'camera_build'),
                 '--build-lib', str(ROOT / '.cache' / 'camera_lib')]
setup(name='force-camera-inspector', version='1.0', ext_modules=[
    Extension('optimized._camera_driver', ['tools/camera_driver.cpp'],
              include_dirs=[pybind11.get_include()], libraries=['strmiids', 'ole32', 'oleaut32'],
              language='c++', extra_compile_args=['/O2', '/std:c++17', '/EHsc'])])
