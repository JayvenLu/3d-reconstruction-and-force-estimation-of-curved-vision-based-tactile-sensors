"""Resolve resources and caches relative to this source release."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
_DLL_HANDLES = []


def configure():
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    cache = ROOT / '.cache'
    for name in ('cupy', 'matplotlib', 'torch', 'tmp'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(cache / 'cupy')
    os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    os.environ['TEMP'] = os.environ['TMP'] = str(cache / 'tmp')
    if os.name == 'nt':
        cuda = Path(sys.prefix) / 'bin'
        if cuda.is_dir():
            os.environ.setdefault('CUDA_PATH', sys.prefix)
            os.environ['PATH'] = str(cuda) + os.pathsep + os.environ.get('PATH', '')
            _DLL_HANDLES.append(os.add_dll_directory(str(cuda)))


def load_reference():
    """Load the local numerical reference, using only this project's dependencies."""
    import importlib.util
    name = '_force_baseline_reference'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / 'baseline' / 'infer_force_realtime_2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
