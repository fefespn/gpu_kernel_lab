# SPDX-License-Identifier: Apache-2.0
"""
Kernel implementations for vector addition across multiple backends.
"""

from .cublas_add import CublasAdd
from .triton_add import TritonAdd
from .cutile_add import CutileAdd
from .pytorch_add import PytorchAdd

__all__ = ['CublasAdd', 'TritonAdd', 'CutileAdd', 'PytorchAdd']

# Registry for easy lookup
BACKENDS = {
    'cublas': CublasAdd,
    'triton': TritonAdd,
    'cutile': CutileAdd,
    'pytorch': PytorchAdd,
}

def get_backend(name: str):
    """Get a kernel backend by name."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]
