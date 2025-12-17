# SPDX-License-Identifier: Apache-2.0
"""
CuPy vector addition using ElementwiseKernel.
"""

import cupy as cp
import numpy as np
import tempfile
import subprocess
import os
from typing import Optional, Dict, Any


class CublasAdd:
    """CuPy ElementwiseKernel-based vector addition."""
    
    name = "cublas"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        self.output_dir = self.config.get('output_dir', 'outputs')
        self.add_kernel = None
    
    def __call__(self, a: cp.ndarray, b: cp.ndarray, c: cp.ndarray) -> None:
        if self.add_kernel is None:
            self.compile()
        self.add_kernel(a, b, c)
    
    def compile(self) -> Dict[str, Any]:
        """Compile the kernel and extract SASS."""
        # Define ElementwiseKernel
        self.add_kernel = cp.ElementwiseKernel(
            'T a, T b',
            'T c',
            'c = a + b',
            'vector_add_kernel'
        )
        
        # Trigger compilation
        dummy_a = cp.empty((1024,), dtype=cp.float32)
        dummy_b = cp.empty_like(dummy_a)
        dummy_c = cp.empty_like(dummy_a)
        self.add_kernel(dummy_a, dummy_b, dummy_c)
        
        # Get device info
        device = cp.cuda.Device()
        actual_sm = int(device.compute_capability)
        
        # Setup output directory
        backend_output_dir = os.path.join(self.output_dir, 'cublas')
        os.makedirs(backend_output_dir, exist_ok=True)
        
        cuobjdump_path = self.config.get('hardware', {}).get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
        
        # cached_code is C++ source, not binary - need to get binary from cache
        from pathlib import Path
        import shutil
        
        cache_dir = Path.home() / '.cupy' / 'kernel_cache'
        artifacts = {}
        
        if cache_dir.exists():
            all_cubins = sorted(cache_dir.glob('**/*.cubin'), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_cubins:
                cubin_src = all_cubins[0]  # Most recent
                cubin_dst = os.path.join(backend_output_dir, f'add_sm{actual_sm}.cubin')
                
                # CuPy cache files have a header before ELF - strip it
                with open(cubin_src, 'rb') as f:
                    content = f.read()
                    elf_start = content.find(b'\x7fELF')
                    if elf_start > 0:
                        kernel_bytes = content[elf_start:]
                    else:
                        kernel_bytes = content
                
                with open(cubin_dst, 'wb') as f:
                    f.write(kernel_bytes)
                
                artifacts['cubin'] = cubin_dst
                print(f"    CUBIN saved: {cubin_dst}")
                
                # Extract SASS
                sass_path = os.path.join(backend_output_dir, f'add_sm{actual_sm}.sass')
                sass_result = subprocess.run(
                    [cuobjdump_path, '-sass', cubin_dst],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if sass_result.returncode == 0 and sass_result.stdout:
                    with open(sass_path, 'w') as f:
                        f.write(sass_result.stdout)
                    artifacts['sass'] = sass_path
                    print(f"    SASS saved: {sass_path}")
                
                # Extract PTX
                ptx_path = os.path.join(backend_output_dir, f'add_sm{actual_sm}.ptx')
                ptx_result = subprocess.run(
                    [cuobjdump_path, '-ptx', cubin_dst],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if ptx_result.returncode == 0 and ptx_result.stdout:
                    with open(ptx_path, 'w') as f:
                        f.write(ptx_result.stdout)
                    artifacts['ptx'] = ptx_path
                    print(f"    PTX saved: {ptx_path}")
        
        return {
            'backend': self.name,
            'status': 'compiled',
            'actual_sm': actual_sm,
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(size: int, dtype=cp.float32):
        a = cp.ones(size, dtype=dtype)
        b = cp.ones(size, dtype=dtype)
        c = cp.zeros(size, dtype=dtype)
        return a, b, c
    
    @staticmethod
    def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
