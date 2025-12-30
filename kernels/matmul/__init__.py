# SPDX-License-Identifier: Apache-2.0
"""
Kernel implementations for matrix multiplication across multiple backends.
Uses lazy imports to avoid requiring all dependencies.
"""

__all__ = ['TritonMatmul', 'CutileMatmul', 'PytorchMatmul', 'get_backend', 'BACKENDS']


def _get_triton():
    from .triton_matmul import TritonMatmul
    return TritonMatmul


def _get_cutile():
    from .cutile_matmul import CutileMatmul
    return CutileMatmul


def _get_pytorch():
    from .pytorch_matmul import PytorchMatmul
    return PytorchMatmul


# Lazy registry
_BACKEND_LOADERS = {
    'triton': _get_triton,
    'cutile': _get_cutile,
    'pytorch': _get_pytorch,
}

_LOADED_BACKENDS = {}


def get_backend(name: str):
    """Get a kernel backend by name (lazy loading)."""
    if name not in _BACKEND_LOADERS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKEND_LOADERS.keys())}")
    
    if name not in _LOADED_BACKENDS:
        _LOADED_BACKENDS[name] = _BACKEND_LOADERS[name]()
    
    return _LOADED_BACKENDS[name]


class _LazyBackends:
    """Lazy dict-like access to backends."""
    
    def __getitem__(self, name):
        return get_backend(name)
    
    def __contains__(self, name):
        return name in _BACKEND_LOADERS
    
    def keys(self):
        return _BACKEND_LOADERS.keys()


BACKENDS = _LazyBackends()


def __getattr__(name):
    """Lazy import for module-level class access."""
    if name == 'TritonMatmul':
        return _get_triton()
    elif name == 'CutileMatmul':
        return _get_cutile()
    elif name == 'PytorchMatmul':
        return _get_pytorch()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
