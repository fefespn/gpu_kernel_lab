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
    
    # Class-level cache: {(tm, tn, tk, dtype_name): compiled_kernel}
    # This ensures we only compile once per unique kernel specialization
    _kernel_cache = {}
    _kernel_func = None  # The kernel function (shared across all instances)
    
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
        """Create the cuTile kernel function (class-level, created once)."""
        # Use class-level kernel function to avoid recreating it
        if CutileMatmul._kernel_func is not None:
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
        
        CutileMatmul._kernel_func = matmul_kernel
    
    def _get_tile_sizes_simple(self, dtype):
        """Get tile sizes based on dtype only (simple heuristic)."""
        # Based on cuTile example heuristics
        if hasattr(dtype, 'itemsize'):
            itemsize = dtype.itemsize
        else:
            itemsize = 4 if dtype == self._torch.float32 else 2
        
        if itemsize == 2:  # fp16/bf16
            return 128, 256, 64
        else:  # fp32
            return 32, 32, 32
    
    def _get_tile_sizes(self, dtype, M: int, N: int, K: int):
        """
        Get optimal tile sizes based on dtype and matrix dimensions.
        
        Uses same logic as Triton: adapts to matrix size, enforces power-of-2,
        and ensures minimum tile size of 16.
        """
        torch = self._torch
        
        # Base tile sizes by dtype
        if hasattr(dtype, 'itemsize'):
            itemsize = dtype.itemsize
        else:
            itemsize = 4 if dtype == torch.float32 else 2
        
        if itemsize == 2:  # fp16/bf16
            # Larger tiles for Tensor Core friendly types
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        else:  # fp32
            # Comparable to Triton's fp32 tile sizes
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        
        # Ensure tile sizes don't exceed matrix dimensions (for efficiency)
        # But don't go below minimum of 16
        BLOCK_M = min(BLOCK_M, max(16, M))
        BLOCK_N = min(BLOCK_N, max(16, N))
        BLOCK_K = min(BLOCK_K, max(16, K))
        
        # Round up to next power of 2 (better for GPU memory coalescing)
        def next_power_of_2(x):
            return 1 << (x - 1).bit_length()
        
        BLOCK_M = next_power_of_2(BLOCK_M)
        BLOCK_N = next_power_of_2(BLOCK_N)
        BLOCK_K = next_power_of_2(BLOCK_K)
        
        return BLOCK_M, BLOCK_N, BLOCK_K
    
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
        
        # Get optimal tile sizes (adaptive based on matrix dimensions)
        tm, tn, tk = self._get_tile_sizes(A.dtype, M, N, K)
        
        # Calculate grid
        grid_x = ceil(M / tm)
        grid_y = ceil(N / tn)
        grid = (grid_x * grid_y, 1, 1)
        
        # Launch kernel using class-level cached kernel
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            CutileMatmul._kernel_func,
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
        
        # Backend-specific output directory
        backend_output_dir = os.path.join(self.output_dir, 'cutile_matmul')
        os.makedirs(backend_output_dir, exist_ok=True)
        
        # Clean old compilation artifacts (keep our renamed ones)
        # This ensures we get the fresh .cubin from this compilation
        for f in os.listdir(backend_output_dir):
            if f.endswith(('.cubin', '.bytecode')) and 'matmul_kernel' in f:
                os.remove(os.path.join(backend_output_dir, f))
        
        artifacts = {}
        
        # Get matrix dimensions from config (default: 8192x8192x8192)
        compile_dims = self.config.get('kernels', {}).get('matmul', {}).get('compile_dims', [8192, 8192, 8192])
        m, n, k = compile_dims[0], compile_dims[1], compile_dims[2]
        A = torch.ones(m, k, dtype=torch.float32, device='cuda')
        B = torch.ones(k, n, dtype=torch.float32, device='cuda')
        C = torch.zeros(m, n, dtype=torch.float32, device='cuda')
        tm, tn, tk = self._get_tile_sizes(A.dtype, m, n, k)
        mock_args = (A, B, C, tm, tn, tk)
        
        # Create dimension suffix for filenames: MxNxK
        dim_suffix = f"_{m}x{n}x{k}"
        
        # Extract Tile IR (typed and optimized, NOT Python AST-level)
        try:
            import functools
            from cuda.tile._compile import _get_final_ir, get_sm_arch
            from cuda.tile._cext import default_tile_context
            from cuda.tile._compiler_options import CompilerOptions
            from cuda.tile._ir2bytecode import generate_bytecode_for_kernel
            import cuda.tile._bytecode as bc
            
            pyfunc = CutileMatmul._kernel_func._pyfunc
            
            # Get the fully typed and optimized IR
            func_ir = _get_final_ir(pyfunc, mock_args, default_tile_context)
            
            # Save the typed cuTile IR (this is the intermediate form)
            typed_ir_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}_typed_ir.txt")
            with open(typed_ir_path, 'w') as f:
                f.write(f"// cuTile Typed IR for matmul_kernel\n")
                f.write(f"// Target: sm_{self.target_sm}\n")
                f.write(f"// Matrix dims: M={m}, N={n}, K={k}\n")
                f.write(f"// Tile sizes: TM={tm}, TN={tn}, TK={tk}\n")
                f.write(f"// This is the typed and optimized Python-level IR\n")
                f.write(f"// (after type inference and optimization passes)\n\n")
                f.write(func_ir.to_string(include_loc=False))
            artifacts['typed_ir'] = typed_ir_path
            print(f"    Typed cuTile IR saved: {typed_ir_path}")
            
            # Generate bytecode to extract MLIR Tile IR
            sm_arch = f"sm_{self.target_sm}"
            compiler_options = CompilerOptions()
            
            bytecode_generator = functools.partial(
                generate_bytecode_for_kernel,
                func_ir, compiler_options, sm_arch
            )
            
            bytecode_buf = bytearray()
            with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
                bytecode_generator(writer, anonymize_debug_attr=False)
            
            # Try to extract MLIR text from bytecode (requires cuda.tile_internal)
            mlir_extracted = False
            try:
                from cuda.tile_internal._internal_cext import bytecode_to_mlir_text
                mlir_text = bytecode_to_mlir_text(bytes(bytecode_buf))
                
                mlir_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}_tile_ir.mlir")
                with open(mlir_path, 'w') as f:
                    f.write(f"// CUDA Tile IR MLIR Dialect for matmul_kernel\n")
                    f.write(f"// Target: sm_{self.target_sm}\n")
                    f.write(f"// Matrix dims: M={m}, N={n}, K={k}\n")
                    f.write(f"// Tile sizes: TM={tm}, TN={tn}, TK={tk}\n\n")
                    f.write(mlir_text)
                artifacts['mlir'] = mlir_path
                artifacts['ir'] = mlir_path  # Use MLIR as primary IR
                mlir_extracted = True
                print(f"    Tile IR MLIR saved: {mlir_path}")
            except ImportError:
                print(f"    Note: cuda.tile_internal not available - using typed IR instead")
                artifacts['ir'] = typed_ir_path
            except Exception as e:
                print(f"    Warning: Could not extract MLIR: {e}")
                artifacts['ir'] = typed_ir_path
            
            # Save bytecode for reference
            bytecode_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}_manual.tilebc")
            with open(bytecode_path, 'wb') as f:
                f.write(bytecode_buf)
            artifacts['tilebc'] = bytecode_path
            
        except Exception as e:
            print(f"    Warning: Could not extract Tile IR: {e}")
            import traceback
            traceback.print_exc()
        
        # Record time before compilation to identify new files
        import time
        compile_start_time = time.time()
        
        # Use the same dummy inputs created above for type inference
        # (m, n, k, A, B, C, tm, tn, tk already defined)
        grid = (ceil(m / tm) * ceil(n / tn), 1, 1)
        
        try:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                CutileMatmul._kernel_func,
                (A, B, C, tm, tn, tk)
            )
            self._compiled = True
            status = 'compiled_and_executed'
        except Exception as e:
            self._compiled = True
            status = f'compiled_only (exec error: {type(e).__name__})'
        
        # Find the newly created artifacts (created after compile_start_time)
        cubin_processed = False
        bytecode_processed = False
        if os.path.exists(backend_output_dir):
            for f in os.listdir(backend_output_dir):
                old_path = os.path.join(backend_output_dir, f)
                
                if f.endswith('.cubin') and not cubin_processed:
                    new_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.cubin")
                    if old_path != new_path:
                        shutil.copy2(old_path, new_path)
                    artifacts['cubin'] = new_path
                    cubin_processed = True
                    
                    # Extract SASS
                    sass_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.sass")
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
                
                elif f.endswith('.bytecode') and not bytecode_processed:
                    new_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.bytecode")
                    if old_path != new_path:
                        shutil.copy2(old_path, new_path)
                    artifacts['bytecode'] = new_path
                    bytecode_processed = True
                    print(f"    Bytecode saved: {new_path}")
                
                elif f.endswith('.ptx') and 'ptx' not in artifacts:
                    new_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.ptx")
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
