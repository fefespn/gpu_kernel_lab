# SPDX-License-Identifier: Apache-2.0
"""
PyTorch vector addition using torch.compile with Triton backend.
"""

import torch
from typing import Optional, Dict, Any


def _add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simple addition function to be compiled."""
    return a + b


class PytorchAdd:
    """PyTorch-based vector addition kernel using torch.compile."""
    
    name = "pytorch"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyTorch kernel.
        
        Args:
            config: Configuration dict with hardware settings
        """
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        
        # Compile with Triton backend
        self._compiled_fn = torch.compile(_add_fn, backend='inductor')
    
    def __call__(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        """
        Perform vector addition: c = a + b.
        
        Args:
            a: First input tensor
            b: Second input tensor
            c: Output tensor (will be modified in-place)
        """
        if self.hardware_mode == 'compile_only':
            raise RuntimeError(
                "Cannot execute in compile_only mode. "
                "PyTorch compile requires native execution."
            )
        
        result = self._compiled_fn(a, b)
        c.copy_(result)
    
    def compile(self) -> Dict[str, Any]:
        """
        Trigger compilation by running a forward pass.
        
        Returns:
            Dict with compilation status
        """
        # Create small input to trigger compilation
        a = torch.ones(16, device='cuda', dtype=torch.float32)
        b = torch.ones(16, device='cuda', dtype=torch.float32)
        
        try:
            _ = self._compiled_fn(a, b)
            status = 'compiled'
        except Exception as e:
            status = f'compile_error: {e}'
        
        return {
            'backend': self.name,
            'status': status,
            'artifacts': {}  # PyTorch doesn't expose PTX/CUBIN directly
        }
    
    @staticmethod
    def create_inputs(size: int, dtype=torch.float32, device='cuda'):
        """
        Create input tensors for testing/benchmarking.
        
        Args:
            size: Number of elements
            dtype: Data type
            device: Device to create tensors on
            
        Returns:
            Tuple of (a, b, c) tensors
        """
        a = torch.ones(size, dtype=dtype, device=device)
        b = torch.ones(size, dtype=dtype, device=device)
        c = torch.zeros(size, dtype=dtype, device=device)
        return a, b, c
    
    @staticmethod
    def reference(a, b):
        """Reference implementation."""
        return a + b
