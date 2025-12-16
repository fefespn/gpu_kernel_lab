# SPDX-License-Identifier: Apache-2.0
"""
cuBLAS vector addition using CuPy.
For simple vector addition, uses optimized CuPy operators (which use cuBLAS internally).
"""

import cupy as cp
import numpy as np
from typing import Optional, Dict, Any


class CublasAdd:
    """cuBLAS-based vector addition kernel."""
    
    name = "cublas"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cuBLAS kernel.
        
        Args:
            config: Configuration dict with hardware settings
        """
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
    
    def __call__(self, a: cp.ndarray, b: cp.ndarray, c: cp.ndarray) -> None:
        """
        Perform vector addition: c = a + b using CuPy.
        
        CuPy's addition uses optimized CUDA kernels internally.
        
        Args:
            a: First input array
            b: Second input array
            c: Output array (will be modified in-place)
        """
        # Use CuPy's optimized addition (uses cuBLAS/cuDNN internally)
        cp.add(a, b, out=c)
    
    def compile(self) -> Dict[str, Any]:
        """
        Compile/prepare the kernel. For cuBLAS, this is a no-op.
        
        Returns:
            Dict with compilation artifacts (empty for cuBLAS)
        """
        return {
            'backend': self.name,
            'status': 'ready',
            'artifacts': {}
        }
    
    @staticmethod
    def create_inputs(size: int, dtype=cp.float32):
        """
        Create input arrays for testing/benchmarking.
        
        Args:
            size: Number of elements
            dtype: Data type
            
        Returns:
            Tuple of (a, b, c) arrays
        """
        a = cp.ones(size, dtype=dtype)
        b = cp.ones(size, dtype=dtype)
        c = cp.zeros(size, dtype=dtype)
        return a, b, c
    
    @staticmethod
    def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NumPy reference implementation."""
        return a + b
