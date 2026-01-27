# SPDX-License-Identifier: Apache-2.0
"""
CuTile Fused GEMM + EXP Kernel

Demonstrates Challenge 4: Blocking Synchronization (CuTile approach)

The kernel computes: C = exp(A @ B)

CuTile/TileIR potential advantages:
- Tile-level abstractions can be lowered to multi-warp code
- Async operations can be scheduled across warps
- GEMM on Warp 0, EXP on Warp 1 (if compiler supports it)

This implementation shows the cuTile equivalent and we analyze
the generated SASS to see if it avoids the blocking sync pattern.
"""

import os
import sys
import argparse
import shutil
import subprocess
from math import ceil
from typing import Optional, Dict, Any


# Set output directory before importing cuda.tile
_OUTPUT_DIR = None

def _setup_cutile_env(output_dir: str):
    """Setup cuTile environment."""
    global _OUTPUT_DIR
    _OUTPUT_DIR = os.path.join(output_dir, 'cutile')
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    os.environ['CUDA_TILE_TEMP_DIR'] = _OUTPUT_DIR


class CutileGemmExp:
    """CuTile fused GEMM + EXP kernel for comparison."""
    
    name = "cutile"
    _kernel_func = None  # Class-level kernel cache
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware_mode', 'compile_only')
        self.target_sm = self.config.get('target_sm', 100)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        self._ct = None
        self._torch = None
    
    def _import_and_patch(self):
        """Import cuTile with sm_100 patch if needed."""
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
        # IMPORTANT: Always import first so self._ct is set
        self._import_and_patch()
        
        if CutileGemmExp._kernel_func is not None:
            return
        
        ct = self._ct
        
        @ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
        def gemm_exp_kernel(
            A, B, C,
            tm: ct.Constant[int],
            tn: ct.Constant[int],
            tk: ct.Constant[int]
        ):
            """
            Fused GEMM + EXP kernel: C = exp(A @ B)
            
            CuTile potential optimization:
            - The tile-level Tile IR can represent async dependencies
            - Compiler may distribute across warps if beneficial
            - num_ctas=2 hints at multi-CTA execution
            """
            M = A.shape[0]
            N = B.shape[1]
            
            bidx = ct.bid(0) // ct.cdiv(N, tn)
            bidy = ct.bid(0) % ct.cdiv(N, tn)
            
            num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
            
            # Accumulator in fp32
            accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
            zero_pad = ct.PaddingMode.ZERO
            
            # Use TF32 for tensor cores
            dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
            
            # GEMM loop - Tensor Core MMA
            for k in range(num_tiles_k):
                a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
                b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
                accumulator = ct.mma(a, b, accumulator)
            
            # =============================================================
            # CuTile's approach to the fusion:
            # 
            # Unlike Triton where tl.exp() forces same-warp execution,
            # Tile IR can represent this as a tile-level operation that
            # the backend may schedule differently.
            # =============================================================
            
            # Apply EXP element-wise
            result = ct.exp(accumulator)
            
            # Store
            result = ct.astype(result, C.dtype)
            ct.store(C, index=(bidx, bidy), tile=result)
        
        CutileGemmExp._kernel_func = gemm_exp_kernel
    
    def __call__(self, A, B, C) -> None:
        """Execute C = exp(A @ B)."""
        self._create_kernel()
        ct = self._ct
        torch = self._torch
        
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        tm, tn, tk = 64, 64, 32
        grid = (ceil(M / tm) * ceil(N / tn), 1, 1)
        
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            CutileGemmExp._kernel_func,
            (A, B, C, tm, tn, tk)
        )
    
    def compile(self) -> Dict[str, Any]:
        """Compile kernel and extract artifacts."""
        self._create_kernel()
        ct = self._ct
        torch = self._torch
        
        backend_dir = os.path.join(self.output_dir, 'cutile')
        os.makedirs(backend_dir, exist_ok=True)
        
        # Clean old artifacts
        for f in os.listdir(backend_dir):
            if f.endswith(('.cubin', '.bytecode')) and 'gemm_exp' in f:
                os.remove(os.path.join(backend_dir, f))
        
        artifacts = {}
        
        # Create dummy tensors for compilation
        m, n, k = 1024, 1024, 1024
        tm, tn, tk = 64, 64, 32
        
        A = torch.ones(m, k, dtype=torch.float32, device='cuda')
        B = torch.ones(k, n, dtype=torch.float32, device='cuda')
        C = torch.zeros(m, n, dtype=torch.float32, device='cuda')
        
        # Extract Tile IR
        try:
            import functools
            from cuda.tile._compile import _get_final_ir, get_sm_arch
            from cuda.tile._cext import default_tile_context
            from cuda.tile._compiler_options import CompilerOptions
            from cuda.tile._ir2bytecode import generate_bytecode_for_kernel
            import cuda.tile._bytecode as bc
            
            pyfunc = CutileGemmExp._kernel_func._pyfunc
            mock_args = (A, B, C, tm, tn, tk)
            
            func_ir = _get_final_ir(pyfunc, mock_args, default_tile_context)
            
            # Save typed IR
            ir_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}_typed_ir.txt")
            with open(ir_path, 'w') as f:
                f.write(f"// CuTile GEMM+EXP Typed IR\n")
                f.write(f"// Target: sm_{self.target_sm}\n\n")
                f.write(func_ir.to_string(include_loc=False))
            artifacts['typed_ir'] = ir_path
            print(f"  Typed IR saved: {ir_path}")
            
            # Try MLIR extraction
            sm_arch = f"sm_{self.target_sm}"
            compiler_options = CompilerOptions()
            
            bytecode_generator = functools.partial(
                generate_bytecode_for_kernel,
                func_ir, compiler_options, sm_arch
            )
            
            bytecode_buf = bytearray()
            with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
                bytecode_generator(writer, anonymize_debug_attr=False)
            
            try:
                from cuda.tile_internal._internal_cext import bytecode_to_mlir_text
                mlir_text = bytecode_to_mlir_text(bytes(bytecode_buf))
                
                mlir_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}_tile_ir.mlir")
                with open(mlir_path, 'w') as f:
                    f.write(f"// CuTile GEMM+EXP Tile IR MLIR\n\n")
                    f.write(mlir_text)
                artifacts['mlir'] = mlir_path
                print(f"  MLIR saved: {mlir_path}")
            except ImportError:
                print("  Note: MLIR extraction requires cuda.tile_internal")
            
        except Exception as e:
            print(f"  Warning: Could not extract IR: {e}")
        
        # Launch to trigger compilation
        grid = (ceil(m / tm) * ceil(n / tn), 1, 1)
        
        try:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                CutileGemmExp._kernel_func,
                (A, B, C, tm, tn, tk)
            )
            status = 'compiled_and_executed'
        except Exception as e:
            status = f'compiled_only ({type(e).__name__})'
        
        # Find generated artifacts
        for f in os.listdir(backend_dir):
            old_path = os.path.join(backend_dir, f)
            
            if f.endswith('.cubin') and 'cubin' not in artifacts:
                new_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}.cubin")
                if old_path != new_path:
                    shutil.copy2(old_path, new_path)
                artifacts['cubin'] = new_path
                print(f"  CUBIN saved: {new_path}")
                
                # Extract SASS
                sass_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}.sass")
                try:
                    cuobjdump = self.config.get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
                    with open(sass_path, 'w') as sf:
                        subprocess.run([cuobjdump, '-sass', new_path], stdout=sf, check=True)
                    artifacts['sass'] = sass_path
                    print(f"  SASS saved: {sass_path}")
                except Exception as e:
                    print(f"  Warning: Could not extract SASS: {e}")
                
                # Extract PTX from debug section using strings
                ptx_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}.ptx")
                try:
                    strings_result = subprocess.run(
                        ['strings', new_path],
                        capture_output=True,
                        text=True
                    )
                    
                    if strings_result.returncode == 0:
                        lines = strings_result.stdout.split('\n')
                        ptx_start_idx = -1
                        for i, line in enumerate(lines):
                            if '.version 9.1' in line or '.version 9.' in line:
                                ptx_start_idx = i
                                break
                        
                        if ptx_start_idx >= 0:
                            ptx_content = '\n'.join(lines[ptx_start_idx:])
                            with open(ptx_path, 'w') as f:
                                f.write(ptx_content)
                            ptx_lines = [l for l in ptx_content.split('\n') if l.strip()]
                            artifacts['ptx'] = ptx_path
                            print(f"  PTX saved: {ptx_path} ({len(ptx_lines)} lines)")
                        else:
                            print(f"  Warning: No PTX found in debug section (no .version directive)")
                    else:
                        print(f"  Warning: Failed to run strings: {strings_result.stderr}")
                except Exception as e:
                    print(f"  Warning: Could not extract PTX: {e}")
        
        return {
            'backend': self.name,
            'status': status,
            'target_sm': self.target_sm,
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=None, device='cuda'):
        """Create test inputs."""
        import torch
        if dtype is None:
            dtype = torch.float32
        scale = 0.1
        A = torch.randn(m, k, dtype=dtype, device=device) * scale
        B = torch.randn(k, n, dtype=dtype, device=device) * scale
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference: C = exp(A @ B)."""
        import torch
        return torch.exp(torch.matmul(A, B))


def main():
    parser = argparse.ArgumentParser(description='CuTile GEMM+EXP Kernel')
    parser.add_argument('--compile-only', action='store_true', help='Only compile, do not run')
    parser.add_argument('--size', type=int, default=1024, help='Matrix size (MxNxK)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CuTile GEMM+EXP Kernel - Challenge 4: Blocking Sync")
    print("=" * 60)
    
    config = {
        'hardware_mode': 'compile_only' if args.compile_only else 'native',
        'target_sm': 100,
        'output_dir': args.output_dir,
    }
    
    kernel = CutileGemmExp(config)
    
    # Compile
    print("\n[1] Compiling kernel...")
    result = kernel.compile()
    print(f"    Status: {result['status']}")
    
    if not args.compile_only:
        print(f"\n[2] Running kernel (size={args.size})...")
        A, B, C = kernel.create_inputs(args.size, args.size, args.size)
        kernel(A, B, C)
        
        import torch
        torch.cuda.synchronize()
        
        # Verify
        expected = kernel.reference(A, B)
        if torch.allclose(C, expected, rtol=1e-2, atol=1e-2):
            print("    ✓ Correctness verified!")
        else:
            max_diff = (C - expected).abs().max().item()
            print(f"    ✗ Max diff: {max_diff}")
    
    print("\n" + "=" * 60)
    print("Artifacts generated:")
    for name, path in result.get('artifacts', {}).items():
        print(f"  - {name}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
