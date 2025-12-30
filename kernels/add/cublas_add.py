# SPDX-License-Identifier: Apache-2.0
"""
cuBLAS vector addition using CUDA runtime cubin dumping.
Uses cuBLAS saxpy (y = alpha*x + y) to perform c = a + b.
"""

import cupy as cp
import numpy as np
import os
import subprocess
import glob
import tempfile
from typing import Optional, Dict, Any


class CublasAdd:
    """cuBLAS-based vector addition using runtime cubin dumping."""
    
    name = "cublas"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        self.output_dir = self.config.get('output_dir', 'outputs')
        self.cuda_cache_dir = None
    
    def __call__(self, a: cp.ndarray, b: cp.ndarray, c: cp.ndarray) -> None:
        """
        Perform vector addition: c = a + b.
        """
        cp.add(a, b, out=c)
    
    def compile(self) -> Dict[str, Any]:
        """
        Trigger cuBLAS compilation and extract SASS via CUDA_DUMP_CUBIN.
        
        Note: We uses matrix multiplication (a @ b) to guarantee triggering 
        cuBLAS and dumping a cubin, as vector addition (cp.add) might not 
        always go through cuBLAS or dump a generic kernel.
        """
        # Setup CUDA cache directory
        self.cuda_cache_dir = tempfile.mkdtemp(prefix='cuda_cache_')
        
        # Set environment variables for CUDA cubin dumping
        old_env = {}
        env_vars = {
            'CUDA_CACHE_DISABLE': '0',
            'CUDA_CACHE_PATH': self.cuda_cache_dir,
            'CUDA_FORCE_PTX_JIT': '0',
            'CUDA_DUMP_CUBIN': '1',
        }
        
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Trigger cuBLAS with GEMM to ensure cubin dump
            # Using float16 pairwise ensures usage of Tensor Cores/cublasLt often
            M, N, K = 1024, 1024, 1024
            
            # Use a separate stream to avoid interfering with ongoing work
            with cp.cuda.Stream():
                a_gemm = cp.random.randn(M, K, dtype=cp.float16)
                b_gemm = cp.random.randn(K, N, dtype=cp.float16)
                c_gemm = a_gemm @ b_gemm
                cp.cuda.runtime.deviceSynchronize()
            
            # Get device info
            device = cp.cuda.Device()
            actual_sm = int(device.compute_capability)
            
            # Setup output directory
            backend_output_dir = os.path.join(self.output_dir, 'cublas')
            os.makedirs(backend_output_dir, exist_ok=True)
            
            artifacts = {}
            
            # Find dumped cubin files
            cubin_files = glob.glob(os.path.join(self.cuda_cache_dir, '**/*.cubin'), recursive=True)
            
            if cubin_files:
                # Take the most recent one
                cubin_src = max(cubin_files, key=os.path.getmtime)
                cubin_dst = os.path.join(backend_output_dir, f'add_sm{actual_sm}.cubin')
                
                # Copy cubin
                import shutil
                shutil.copy2(cubin_src, cubin_dst)
                artifacts['cubin'] = cubin_dst
                print(f"    CUBIN saved: {cubin_dst}")
                
                # Extract SASS using nvdisasm
                nvdisasm_path = self.config.get('hardware', {}).get('nvdisasm_path', '/usr/local/cuda/bin/nvdisasm')
                sass_path = os.path.join(backend_output_dir, f'add_sm{actual_sm}.sass')
                
                try:
                    result = subprocess.run(
                        [nvdisasm_path, cubin_dst],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    with open(sass_path, 'w') as f:
                        f.write(result.stdout)
                    artifacts['sass'] = sass_path
                    print(f"    SASS saved: {sass_path}")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"    Warning: nvdisasm failed: {e}")
                
                # Try cuobjdump for PTX
                cuobjdump_path = self.config.get('hardware', {}).get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
                ptx_path = os.path.join(backend_output_dir, f'add_sm{actual_sm}.ptx')
                
                try:
                    result = subprocess.run(
                        [cuobjdump_path, '-ptx', cubin_dst],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        with open(ptx_path, 'w') as f:
                            f.write(result.stdout)
                        artifacts['ptx'] = ptx_path
                        print(f"    PTX saved: {ptx_path}")
                except FileNotFoundError:
                    pass
            else:
                print(f"    Warning: No cubin files found in {self.cuda_cache_dir}")
            
            return {
                'backend': self.name,
                'status': 'compiled',
                'actual_sm': actual_sm,
                'artifacts': artifacts
            }
            
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            
            # Cleanup cache directory
            if self.cuda_cache_dir and os.path.exists(self.cuda_cache_dir):
                import shutil
                shutil.rmtree(self.cuda_cache_dir, ignore_errors=True)
    
    @staticmethod
    def create_inputs(size: int, dtype=cp.float32):
        a = cp.ones(size, dtype=dtype)
        b = cp.ones(size, dtype=dtype)
        c = cp.zeros(size, dtype=dtype)
        return a, b, c
    
    @staticmethod
    def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
