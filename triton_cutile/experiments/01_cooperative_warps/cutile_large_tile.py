# SPDX-License-Identifier: Apache-2.0
"""
CuTile Large-Tile GEMM Kernel

Demonstrates Challenge 1: Cooperative Warps (CuTile approach)

CuTile/TileIR can express warp-group-level operations:
1. Large tiles naturally map to WGMMA (Warp Group MMA)
2. Compiler handles warp cooperation implicitly
3. Registers distributed across warp group automatically

Expected SASS patterns:
- WGMMA.128x128x16 or similar warp-group instructions
- Implicit warp synchronization within instruction
- Better register utilization via distribution
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


class CutileLargeTileGemm:
    """CuTile large-tile GEMM leveraging warp-group cooperation."""
    
    name = "cutile"
    _kernel_func = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware_mode', 'compile_only')
        self.target_sm = self.config.get('target_sm', 100)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        # Large tiles - same as Triton for fair comparison
        self.tm = 128
        self.tn = 128
        self.tk = 64
        
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
        
        if self.hardware_mode == 'compile_only':
            import cuda.tile._compile as ct_compile
            ct_compile.get_sm_arch = lambda: f'sm_{self.target_sm}'
    
    def _create_kernel(self):
        """Create the cuTile kernel function."""
        # Always import first so self._ct is set
        self._import_and_patch()
        
        if CutileLargeTileGemm._kernel_func is not None:
            return
        
        ct = self._ct
        
        # Use num_ctas=4 to hint at warp-group execution
        # This allows the compiler to schedule 4 CTAs together
        @ct.kernel(num_ctas=ct.ByTarget(sm_100=4))
        def large_tile_gemm_kernel(
            A, B, C,
            tm: ct.Constant[int],
            tn: ct.Constant[int],
            tk: ct.Constant[int]
        ):
            """
            Large-tile GEMM: C = A @ B
            
            CuTile advantage:
            - TileIR represents this as a tile-level operation
            - Compiler can lower to WGMMA (warp-group MMA)
            - Implicit cooperation between warps in the group
            
            With tm=128, tn=128, this naturally maps to warp-group ops
            """
            M = A.shape[0]
            N = B.shape[1]
            
            bidx = ct.bid(0) // ct.cdiv(N, tn)
            bidy = ct.bid(0) % ct.cdiv(N, tn)
            
            num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
            
            # Accumulator - fp32 for precision
            accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
            zero_pad = ct.PaddingMode.ZERO
            
            # Use TF32 for tensor cores
            dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
            
            # GEMM loop with large tiles
            # CuTile's ct.mma() can generate WGMMA for these sizes
            for k in range(num_tiles_k):
                a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
                b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
                
                # This MMA operation can be lowered to WGMMA
                # requiring cooperation from 4 warps in a warp group
                accumulator = ct.mma(a, b, accumulator)
            
            # Store result
            result = ct.astype(accumulator, C.dtype)
            ct.store(C, index=(bidx, bidy), tile=result)
        
        CutileLargeTileGemm._kernel_func = large_tile_gemm_kernel
    
    def __call__(self, A, B, C) -> None:
        """Execute C = A @ B."""
        self._create_kernel()
        ct = self._ct
        torch = self._torch
        
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        grid = (ceil(M / self.tm) * ceil(N / self.tn), 1, 1)
        
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            CutileLargeTileGemm._kernel_func,
            (A, B, C, self.tm, self.tn, self.tk)
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
            if f.endswith(('.cubin', '.bytecode')) and 'large_tile' in f:
                os.remove(os.path.join(backend_dir, f))
        
        artifacts = {}
        base_name = f"large_tile_gemm_{self.tm}x{self.tn}x{self.tk}_sm{self.target_sm}"
        
        # Create tensors for compilation - use 4096x4096x128 (CuTile wins here!)
        m, n, k = 4096, 128, 4096  # M=4096, K=4096, N=128
        A = torch.ones(m, k, dtype=torch.float16, device='cuda')
        B = torch.ones(k, n, dtype=torch.float16, device='cuda')
        C = torch.zeros(m, n, dtype=torch.float16, device='cuda')
        
        # Extract Tile IR
        try:
            import functools
            from cuda.tile._compile import _get_final_ir
            from cuda.tile._cext import default_tile_context
            
            pyfunc = CutileLargeTileGemm._kernel_func._pyfunc
            mock_args = (A, B, C, self.tm, self.tn, self.tk)
            
            func_ir = _get_final_ir(pyfunc, mock_args, default_tile_context)
            
            ir_path = os.path.join(backend_dir, f"{base_name}_typed_ir.txt")
            with open(ir_path, 'w') as f:
                f.write(f"// CuTile Large-Tile GEMM Typed IR\n")
                f.write(f"// Tiles: {self.tm}x{self.tn}x{self.tk}\n\n")
                f.write(func_ir.to_string(include_loc=False))
            artifacts['typed_ir'] = ir_path
            print(f"  Typed IR saved: {ir_path}")
            
        except Exception as e:
            print(f"  Warning: Could not extract IR: {e}")
        
        # Launch to trigger compilation
        grid = (ceil(m / self.tm) * ceil(n / self.tn), 1, 1)
        
        try:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                CutileLargeTileGemm._kernel_func,
                (A, B, C, self.tm, self.tn, self.tk)
            )
            status = 'compiled_and_executed'
        except Exception as e:
            status = f'compiled_only ({type(e).__name__})'
        
        # Find and save artifacts
        for f in os.listdir(backend_dir):
            old_path = os.path.join(backend_dir, f)
            
            if f.endswith('.cubin') and 'cubin' not in artifacts:
                new_path = os.path.join(backend_dir, f"{base_name}.cubin")
                if old_path != new_path:
                    shutil.copy2(old_path, new_path)
                artifacts['cubin'] = new_path
                print(f"  CUBIN saved: {new_path}")
                
                # Extract SASS
                sass_path = os.path.join(backend_dir, f"{base_name}.sass")
                try:
                    cuobjdump = self.config.get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
                    with open(sass_path, 'w') as sf:
                        subprocess.run([cuobjdump, '-sass', new_path], stdout=sf, check=True)
                    artifacts['sass'] = sass_path
                    print(f"  SASS saved: {sass_path}")
                except Exception as e:
                    print(f"  Warning: Could not extract SASS: {e}")
        
        return {
            'backend': self.name,
            'status': status,
            'target_sm': self.target_sm,
            'tile_size': f'{self.tm}x{self.tn}x{self.tk}',
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=None, device='cuda'):
        """Create test inputs."""
        import torch
        if dtype is None:
            dtype = torch.float16
        A = torch.randn(m, k, dtype=dtype, device=device) * 0.1
        B = torch.randn(k, n, dtype=dtype, device=device) * 0.1
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference implementation."""
        import torch
        return torch.matmul(A.float(), B.float()).half()


def main():
    parser = argparse.ArgumentParser(description='CuTile Large-Tile GEMM (Cooperative Warps)')
    parser.add_argument('--compile-only', action='store_true', help='Only compile')
    parser.add_argument('--size', type=int, default=2048, help='Matrix size')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CuTile Large-Tile GEMM - Challenge 1: Cooperative Warps")
    print("=" * 60)
    
    config = {
        'hardware_mode': 'compile_only' if args.compile_only else 'native',
        'target_sm': 100,
        'output_dir': args.output_dir,
    }
    
    kernel = CutileLargeTileGemm(config)
    
    print(f"\n[1] Compiling kernel (tiles: {kernel.tm}x{kernel.tn}x{kernel.tk})...")
    result = kernel.compile()
    print(f"    Status: {result['status']}")
    
    if not args.compile_only:
        print(f"\n[2] Running kernel (size={args.size})...")
        A, B, C = kernel.create_inputs(args.size, args.size, args.size)
        kernel(A, B, C)
        
        import torch
        torch.cuda.synchronize()
        
        expected = kernel.reference(A, B)
        if torch.allclose(C, expected, rtol=1e-2, atol=1e-2):
            print("    ✓ Correctness verified!")
        else:
            max_diff = (C.float() - expected.float()).abs().max().item()
            print(f"    ✗ Max diff: {max_diff}")
    
    print("\n" + "=" * 60)
    print("Artifacts generated:")
    for name, path in result.get('artifacts', {}).items():
        print(f"  - {name}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
