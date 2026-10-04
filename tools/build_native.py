"""Build once during setup; does not install anything into the Python environment."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
from setuptools import setup, Extension
import pybind11

if len(sys.argv) == 1:
    sys.argv += ['build_ext', '--inplace', '--build-temp', str(ROOT / '.cache' / 'build')]
setup(name='force20hz-native', version='1.0',
      ext_modules=[Extension('optimized._matching', ['optimized/matching.cpp'],
                             include_dirs=[pybind11.get_include()], language='c++',
                             extra_compile_args=(['/O2', '/std:c++17', '/EHsc', '/fp:precise'] if os.name == 'nt' else ['-O3', '-std=c++17']))])
