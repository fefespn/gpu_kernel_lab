# SPDX-License-Identifier: Apache-2.0
"""
Triton vector addition kernel with support for:
- Native mode: JIT compilation on Blackwell (sm_100)
- Compile-only mode: AOT compilation for sm_100 on any GPU
"""

import os
import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any


@triton.jit
def _vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for vector addition."""
    # Get block ID
    pid = tl.program_id(0)
    
    # Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offsets < n_elements
    
    # Load input tiles
    a_tile = tl.load(a_ptr + offsets, mask=mask)
    b_tile = tl.load(b_ptr + offsets, mask=mask)
    
    # Elementwise addition
    result = a_tile + b_tile
    
    # Store result
    tl.store(c_ptr + offsets, result, mask=mask)


class TritonAdd:
    """Triton-based vector addition kernel."""
    
    name = "triton"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Triton kernel.
        
        Args:
            config: Configuration dict with hardware settings
        """
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        self.target_sm = self.config.get('hardware', {}).get('target_sm', 100)
        self.tile_size = self.config.get('kernels', {}).get('add', {}).get('tile_size', 16)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        self._compiled = None
    
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
                "Use compile() to generate artifacts instead."
            )
        
        n_elements = a.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        _vector_add_kernel[grid](
            a, b, c,
            n_elements,
            BLOCK_SIZE=self.tile_size
        )
    
    def compile(self) -> Dict[str, Any]:
        """
        Compile the kernel for sm_100 (AOT compilation).
        Works on any GPU by using offline compilation.
        
        Returns:
            Dict with compilation artifacts (ptx, cubin, sass paths)
        """
        import triton.compiler as tc
        from triton.backends.compiler import GPUTarget
        
        signature = {
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "c_ptr": "*fp32",
            "n_elements": "i32",
        }
        
        src = tc.ASTSource(
            fn=_vector_add_kernel,
            constexprs={"BLOCK_SIZE": self.tile_size},
            signature=signature
        )
        
        target = GPUTarget("cuda", self.target_sm, 64)
        compiled_result = triton.compile(src, target=target)
        asm = compiled_result.asm
        
        self._compiled = asm
        
        artifacts = {}
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save PTX
        if 'ptx' in asm:
            ptx_path = os.path.join(self.output_dir, f"triton_add_sm{self.target_sm}.ptx")
            with open(ptx_path, 'w') as f:
                f.write(asm['ptx'])
            artifacts['ptx'] = ptx_path
        
        # Save CUBIN
        if 'cubin' in asm:
            cubin_path = os.path.join(self.output_dir, f"triton_add_sm{self.target_sm}.cubin")
            with open(cubin_path, 'wb') as f:
                f.write(asm['cubin'])
            artifacts['cubin'] = cubin_path
        
        return {
            'backend': self.name,
            'status': 'compiled',
            'target_sm': self.target_sm,
            'artifacts': artifacts
        }
    
    def get_sass(self) -> Optional[str]:
        """
        Extract SASS from compiled cubin.
        
        Returns:
            SASS code as string, or None if not compiled
        """
        if self._compiled is None:
            return None
        
        try:
            from triton.tools.disasm import get_sass
            return get_sass(self._compiled['cubin'])
        except (ImportError, KeyError):
            return None
    
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
