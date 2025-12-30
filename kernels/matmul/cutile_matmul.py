# SPDX-License-Identifier: Apache-2.0
"""
cuTile matrix multiplication kernel with support for:
- Native mode: Normal execution on Blackwell (sm_100)
- Compile-only mode: Monkey-patch get_sm_arch() to compile for sm_100 on any GPU
"""

import os
import subprocess
import shutil
from math import ceil
from typing import Optional, Dict, Any

# Must set CUDA_TILE_TEMP_DIR before importing cuda.tile
_OUTPUT_DIR = None
_BACKEND_OUTPUT_DIR = None


def _setup_cutile_env(output_dir: str):
    """Setup environment for cuTile with backend-specific subfolder."""
    global _OUTPUT_DIR, _BACKEND_OUTPUT_DIR
    
    _BACKEND_OUTPUT_DIR = os.path.join(output_dir, 'cutile_matmul')
    os.makedirs(_BACKEND_OUTPUT_DIR, exist_ok=True)
    
    _OUTPUT_DIR = _BACKEND_OUTPUT_DIR
    os.environ['CUDA_TILE_TEMP_DIR'] = _BACKEND_OUTPUT_DIR


class CutileMatmul:
    """cuTile-based matrix multiplication kernel."""
    
    name = "cutile"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cuTile kernel.
        
        Args:
            config: Configuration dict with hardware settings
        """
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        self.target_sm = self.config.get('hardware', {}).get('target_sm', 100)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        self._kernel = None
        self._compiled = False
        self._ct = None
        self._torch = None
    
    def _import_and_patch(self):
        """Import cuTile and apply sm_100 monkey-patch if needed."""
        if self._ct is not None:
            return
        
        _setup_cutile_env(self.output_dir)
        
        import torch
        import cuda.tile as ct
        
        self._torch = torch
        self._ct = ct
        
        # Monkey-patch for compile-only mode
        if self.hardware_mode == 'compile_only':
            import cuda.tile._compile as ct_compile
            ct_compile.get_sm_arch = lambda: f'sm_{self.target_sm}'
    
    def _create_kernel(self):
        """Create the cuTile kernel function."""
        if self._kernel is not None:
            return
        
        self._import_and_patch()
        ct = self._ct
        
        # Helper functions for swizzling
        def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
            num_bid_m = ct.cdiv(M, tm)
            num_bid_n = ct.cdiv(N, tn)
            num_bid_in_group = GROUP_SIZE_M * num_bid_n
            group_id = bid // num_bid_in_group
            first_bid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
            bid_m = first_bid_m + (bid % group_size_m)
            bid_n = (bid % num_bid_in_group) // group_size_m
            return bid_m, bid_n
        
        def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
            bid = ct.bid(0)
            return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)
        
        @ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
        def matmul_kernel(A, B, C,
                          tm: ct.Constant[int],
                          tn: ct.Constant[int],
                          tk: ct.Constant[int]):
            """
            cuTile kernel for matrix multiplication C = A @ B.
            
            Args:
                A: Input matrix A (M x K)
                B: Input matrix B (K x N)
                C: Output matrix C (M x N)
                tm: Tile size along M dimension
                tn: Tile size along N dimension
                tk: Tile size along K dimension
            """
            GROUP_SIZE_M = 8
            M = A.shape[0]
            N = B.shape[1]
            bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
            
            num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
            
            # Initialize accumulator in fp32 for precision
            accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
            zero_pad = ct.PaddingMode.ZERO
            
            # Convert fp32 to tf32 to use tensorcore
            dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
            
            # K-dimension loop
            for k in range(num_tiles_k):
                a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
                b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
                accumulator = ct.mma(a, b, accumulator)
            
            # Convert to output dtype and store
            accumulator = ct.astype(accumulator, C.dtype)
            ct.store(C, index=(bidx, bidy), tile=accumulator)
        
        self._kernel = matmul_kernel
    
    def _get_tile_sizes(self, dtype):
        """Get optimal tile sizes based on dtype."""
        # Based on cuTile example heuristics
        if hasattr(dtype, 'itemsize'):
            itemsize = dtype.itemsize
        else:
            itemsize = 4 if dtype == self._torch.float32 else 2
        
        if itemsize == 2:  # fp16/bf16
            return 128, 256, 64
        else:  # fp32
            return 32, 32, 32
    
    def __call__(self, A, B, C) -> None:
        """
        Perform matrix multiplication: C = A @ B.
        
        Args:
            A: First input matrix (M x K) - torch.Tensor
            B: Second input matrix (K x N) - torch.Tensor
            C: Output matrix (M x N, will be modified in-place) - torch.Tensor
        """
        self._create_kernel()
        ct = self._ct
        torch = self._torch
        
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Incompatible dimensions: A is {A.shape}, B is {B.shape}"
        
        # Get optimal tile sizes
        tm, tn, tk = self._get_tile_sizes(A.dtype)
        
        # Calculate grid
        grid_x = ceil(M / tm)
        grid_y = ceil(N / tn)
        grid = (grid_x * grid_y, 1, 1)
        
        # Launch kernel
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            self._kernel,
            (A, B, C, tm, tn, tk)
        )
    
    def compile(self) -> Dict[str, Any]:
        """
        Compile the kernel for sm_100.
        
        Returns:
            Dict with compilation artifacts
        """
        self._create_kernel()
        ct = self._ct
        torch = self._torch
        
        # Create dummy inputs to trigger compilation
        m, n, k = 256, 256, 256
        A = torch.ones(m, k, dtype=torch.float32, device='cuda')
        B = torch.ones(k, n, dtype=torch.float32, device='cuda')
        C = torch.zeros(m, n, dtype=torch.float32, device='cuda')
        
        tm, tn, tk = self._get_tile_sizes(A.dtype)
        grid = (ceil(m / tm) * ceil(n / tn), 1, 1)
        
        artifacts = {}
        
        try:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                self._kernel,
                (A, B, C, tm, tn, tk)
            )
            self._compiled = True
            status = 'compiled_and_executed'
        except Exception as e:
            self._compiled = True
            status = f'compiled_only (exec error: {type(e).__name__})'
        
        # Backend-specific output directory
        backend_output_dir = os.path.join(self.output_dir, 'cutile_matmul')
        
        # Check for artifacts
        cubin_processed = False
        if os.path.exists(backend_output_dir):
            for f in os.listdir(backend_output_dir):
                old_path = os.path.join(backend_output_dir, f)
                
                if f.endswith('.cubin') and not cubin_processed:
                    new_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}.cubin")
                    if old_path != new_path:
                        shutil.copy2(old_path, new_path)
                    artifacts['cubin'] = new_path
                    cubin_processed = True
                    
                    # Extract SASS
                    sass_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}.sass")
                    cuobjdump_path = self.config.get('hardware', {}).get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
                    
                    try:
                        with open(sass_path, 'w') as sf:
                            subprocess.run(
                                [cuobjdump_path, '-sass', new_path],
                                stdout=sf,
                                stderr=subprocess.DEVNULL,
                                check=True
                            )
                        artifacts['sass'] = sass_path
                        print(f"    SASS extracted: {sass_path}")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        pass
                
                elif f.endswith('.ptx') and 'ptx' not in artifacts:
                    new_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}.ptx")
                    if old_path != new_path:
                        shutil.copy2(old_path, new_path)
                    artifacts['ptx'] = new_path
        
        return {
            'backend': self.name,
            'status': status,
            'target_sm': self.target_sm,
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=None, device='cuda'):
        """
        Create input matrices for testing/benchmarking.
        
        Args:
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Number of columns in A / rows in B
            dtype: Data type (default: float32)
            device: Device to create tensors on
            
        Returns:
            Tuple of (A, B, C) tensors
        """
        import torch
        if dtype is None:
            dtype = torch.float32
        
        # Scale inputs to control accumulation magnitude
        scale = 1.0 / (k ** 0.5)
        A = torch.randn(m, k, dtype=dtype, device=device) * scale
        B = torch.randn(k, n, dtype=dtype, device=device) * scale
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference implementation."""
        import torch
        return torch.matmul(A, B)
