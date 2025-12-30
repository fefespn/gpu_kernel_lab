# SPDX-License-Identifier: Apache-2.0
"""
Kernel implementations for vector addition across multiple backends.
Uses lazy imports to avoid requiring all dependencies.
"""

__all__ = ['CublasAdd', 'TritonAdd', 'CutileAdd', 'PytorchAdd', 'get_backend', 'BACKENDS']


def _get_cublas():
    from .cublas_add import CublasAdd
    return CublasAdd


def _get_triton():
    from .triton_add import TritonAdd
    return TritonAdd


def _get_cutile():
    from .cutile_add import CutileAdd
    return CutileAdd


def _get_pytorch():
    from .pytorch_add import PytorchAdd
    return PytorchAdd


# Lazy registry - functions that return classes
_BACKEND_LOADERS = {
    'cublas': _get_cublas,
    'triton': _get_triton,
    'cutile': _get_cutile,
    'pytorch': _get_pytorch,
}

# Cache for loaded backends
_LOADED_BACKENDS = {}


def get_backend(name: str):
    """Get a kernel backend by name (lazy loading)."""
    if name not in _BACKEND_LOADERS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKEND_LOADERS.keys())}")
    
    if name not in _LOADED_BACKENDS:
        _LOADED_BACKENDS[name] = _BACKEND_LOADERS[name]()
    
    return _LOADED_BACKENDS[name]


# For backwards compatibility - these are properties that lazy-load
class _LazyBackends:
    """Lazy dict-like access to backends."""
    
    def __getitem__(self, name):
        return get_backend(name)
    
    def __contains__(self, name):
        return name in _BACKEND_LOADERS
    
    def keys(self):
        return _BACKEND_LOADERS.keys()
    
    def items(self):
        return ((k, get_backend(k)) for k in _BACKEND_LOADERS.keys())


BACKENDS = _LazyBackends()


# Lazy module-level imports for backwards compatibility
def __getattr__(name):
    """Lazy import for module-level class access."""
    if name == 'CublasAdd':
        return _get_cublas()
    elif name == 'TritonAdd':
        return _get_triton()
    elif name == 'CutileAdd':
        return _get_cutile()
    elif name == 'PytorchAdd':
        return _get_pytorch()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
