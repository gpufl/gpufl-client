import os
import sys
if os.name == 'nt':
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            try:
                os.add_dll_directory(bin_path)
            except AttributeError:
                # Fallback for older Python versions (pre-3.8) or non-standard envs
                pass

from ._gpumon_client import Scope

try:
    from ._gpumon_client import init, shutdown
    __all__ = ["Scope", "init", "shutdown"]
except ImportError:
    __all__ = ["Scope"]