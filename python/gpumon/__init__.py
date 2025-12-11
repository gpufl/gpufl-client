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

# 2. Import C++ Core Bindings
try:
    # We now import log_kernel as well (added in bindings.cpp)
    from ._gpumon_client import Scope, init, shutdown, log_kernel
except ImportError:
    # Fallback if binary is missing or load failed (prevents immediate crash during IDE indexing)
    def init(*args, **kwargs): pass
    def shutdown(): pass
    def log_kernel(*args): pass
    class Scope:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# 3. Import Python Utilities (The Numba Wrapper)
try:
    from .utils import launch_kernel
except ImportError:
    # Should only fail if utils.py is missing or syntax error
    launch_kernel = None

# 4. Define Public API
__all__ = ["Scope", "init", "shutdown", "log_kernel", "launch_kernel"]