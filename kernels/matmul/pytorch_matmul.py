# SPDX-License-Identifier: Apache-2.0
"""
PyTorch matrix multiplication using torch.compile.
"""

import torch
from typing import Optional, Dict, Any


def _matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simple matmul function to be compiled."""
    return torch.matmul(a, b)


class PytorchMatmul:
    """PyTorch-based matrix multiplication kernel using torch.compile."""
    
    name = "pytorch"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyTorch kernel.
        
        Args:
            config: Configuration dict with hardware settings
        """
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        
        # Compile with inductor backend
        self._compiled_fn = torch.compile(_matmul_fn, backend='inductor')
    
    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        """
        Perform matrix multiplication: C = A @ B.
        
        Args:
            A: First input matrix (M x K)
            B: Second input matrix (K x N)
            C: Output matrix (M x N, will be modified in-place)
        """
        result = self._compiled_fn(A, B)
        C.copy_(result)
    
    def compile(self) -> Dict[str, Any]:
        """
        Trigger compilation by running a forward pass.
        
        Returns:
            Dict with compilation status
        """
        # Create small input to trigger compilation
        A = torch.ones(64, 64, device='cuda', dtype=torch.float32)
        B = torch.ones(64, 64, device='cuda', dtype=torch.float32)
        
        try:
            _ = self._compiled_fn(A, B)
            status = 'compiled'
        except Exception as e:
            status = f'compile_error: {e}'
        
        return {
            'backend': self.name,
            'status': status,
            'artifacts': {}
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=torch.float32, device='cuda'):
        """
        Create input matrices for testing/benchmarking.
        
        Args:
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Number of columns in A / rows in B
            dtype: Data type
            device: Device to create tensors on
            
        Returns:
            Tuple of (A, B, C) tensors
        """
        # Scale inputs to control accumulation magnitude
        scale = 1.0 / (k ** 0.5)
        A = torch.randn(m, k, dtype=dtype, device=device) * scale
        B = torch.randn(k, n, dtype=dtype, device=device) * scale
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference implementation."""
        return torch.matmul(A, B)
