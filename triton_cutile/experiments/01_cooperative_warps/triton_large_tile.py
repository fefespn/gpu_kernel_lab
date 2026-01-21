# SPDX-License-Identifier: Apache-2.0
"""
Triton Large-Tile GEMM Kernel

Demonstrates Challenge 1: Cooperative Warps

The kernel uses large tile sizes (128x128x64) which on Hopper/Blackwell
require warp-group cooperation (4 warps working together for WGMMA).

Triton's single-warp programming model means:
1. Each "program" conceptually runs on one warp
2. Large tiles get compiled to multiple smaller MMA ops
3. Extra synchronization needed between fragments

Expected SASS patterns:
- Multiple smaller HMMA/MMA ops instead of single WGMMA
- Explicit BAR.SYNC for warp coordination
- Higher register pressure from fragmented accumulator
"""

import os
import sys
import argparse
import subprocess
from math import ceil
from typing import Optional, Dict, Any

import torch
import triton
import triton.language as tl


@triton.jit
def _large_tile_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Large-tile GEMM: C = A @ B
    
    Uses BLOCK_M=128, BLOCK_N=128 which requires warp-group cooperation
    on Hopper/Blackwell for efficient WGMMA utilization.
    
    Challenge: Triton compiles this as single-warp code, potentially
    missing warp-group MMA opportunities.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator - fp32 for precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # GEMM loop
    # With BLOCK_M=128, BLOCK_N=128, this should ideally use WGMMA
    # But Triton's single-warp model may fragment this
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Bounds checking
        k_offs = k * BLOCK_K + offs_k
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        # Load tiles
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Matrix multiply - Triton will compile this to MMA ops
        # On Hopper/Blackwell with 128x128 tiles, optimal code would
        # use WGMMA requiring 4 cooperative warps
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c = accumulator.to(tl.float16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


class TritonLargeTileGemm:
    """Triton large-tile GEMM for cooperative warps experiment."""
    
    name = "triton"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware_mode', 'compile_only')
        self.target_sm = self.config.get('target_sm', 100)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        # Large tiles to require warp-group cooperation
        self.BLOCK_M = 128
        self.BLOCK_N = 128
        self.BLOCK_K = 64
    
    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        """Execute C = A @ B."""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        grid = (ceil(M / self.BLOCK_M), ceil(N / self.BLOCK_N))
        
        _large_tile_gemm_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
        )
    
    def compile(self) -> Dict[str, Any]:
        """Compile kernel and extract artifacts."""
        from triton.backends.compiler import GPUTarget
        
        backend_dir = os.path.join(self.output_dir, 'triton')
        os.makedirs(backend_dir, exist_ok=True)
        
        # Create kernel source
        import triton.compiler as tc
        
        src = tc.ASTSource(
            fn=_large_tile_gemm_kernel,
            signature={
                'A_ptr': '*fp16',
                'B_ptr': '*fp16', 
                'C_ptr': '*fp16',
                'M': 'i32', 'N': 'i32', 'K': 'i32',
                'stride_am': 'i32', 'stride_ak': 'i32',
                'stride_bk': 'i32', 'stride_bn': 'i32',
                'stride_cm': 'i32', 'stride_cn': 'i32',
            },
            constexprs={
                'BLOCK_M': self.BLOCK_M,
                'BLOCK_N': self.BLOCK_N,
                'BLOCK_K': self.BLOCK_K,
            }
        )
        
        target = GPUTarget("cuda", self.target_sm, 64)
        compiled = triton.compile(src, target=target)
        asm = compiled.asm
        
        artifacts = {}
        base_name = f"large_tile_gemm_{self.BLOCK_M}x{self.BLOCK_N}x{self.BLOCK_K}_sm{self.target_sm}"
        
        # Save PTX
        if 'ptx' in asm:
            ptx_path = os.path.join(backend_dir, f"{base_name}.ptx")
            with open(ptx_path, 'w') as f:
                f.write(asm['ptx'])
            artifacts['ptx'] = ptx_path
            print(f"  PTX saved: {ptx_path}")
        
        # Save CUBIN
        if 'cubin' in asm:
            cubin_path = os.path.join(backend_dir, f"{base_name}.cubin")
            with open(cubin_path, 'wb') as f:
                f.write(asm['cubin'])
            artifacts['cubin'] = cubin_path
            print(f"  CUBIN saved: {cubin_path}")
            
            # Extract SASS
            sass_path = os.path.join(backend_dir, f"{base_name}.sass")
            try:
                cuobjdump = self.config.get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
                with open(sass_path, 'w') as f:
                    subprocess.run([cuobjdump, '-sass', cubin_path], stdout=f, check=True)
                artifacts['sass'] = sass_path
                print(f"  SASS saved: {sass_path}")
            except Exception as e:
                print(f"  Warning: Could not extract SASS: {e}")
        
        # Save TTIR
        if 'ttir' in asm:
            ttir_path = os.path.join(backend_dir, f"{base_name}_ttir.txt")
            with open(ttir_path, 'w') as f:
                f.write(str(asm['ttir']))
            artifacts['ttir'] = ttir_path
            print(f"  TTIR saved: {ttir_path}")
        
        # Save TTGIR
        if 'ttgir' in asm:
            ttgir_path = os.path.join(backend_dir, f"{base_name}_ttgir.txt")
            with open(ttgir_path, 'w') as f:
                f.write(str(asm['ttgir']))
            artifacts['ttgir'] = ttgir_path
            print(f"  TTGIR saved: {ttgir_path}")
        
        return {
            'backend': self.name,
            'status': 'compiled',
            'target_sm': self.target_sm,
            'tile_size': f'{self.BLOCK_M}x{self.BLOCK_N}x{self.BLOCK_K}',
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=None, device='cuda'):
        """Create test inputs."""
        if dtype is None:
            dtype = torch.float16
        A = torch.randn(m, k, dtype=dtype, device=device) * 0.1
        B = torch.randn(k, n, dtype=dtype, device=device) * 0.1
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference implementation."""
        return torch.matmul(A.float(), B.float()).half()


def main():
    parser = argparse.ArgumentParser(description='Triton Large-Tile GEMM (Cooperative Warps)')
    parser.add_argument('--compile-only', action='store_true', help='Only compile')
    parser.add_argument('--size', type=int, default=2048, help='Matrix size')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Triton Large-Tile GEMM - Challenge 1: Cooperative Warps")
    print("=" * 60)
    
    config = {
        'hardware_mode': 'compile_only' if args.compile_only else 'native',
        'target_sm': 100,
        'output_dir': args.output_dir,
    }
    
    kernel = TritonLargeTileGemm(config)
    
    print(f"\n[1] Compiling kernel (tiles: {kernel.BLOCK_M}x{kernel.BLOCK_N}x{kernel.BLOCK_K})...")
    result = kernel.compile()
    print(f"    Status: {result['status']}")
    
    if not args.compile_only:
        print(f"\n[2] Running kernel (size={args.size})...")
        A, B, C = kernel.create_inputs(args.size, args.size, args.size)
        kernel(A, B, C)
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
