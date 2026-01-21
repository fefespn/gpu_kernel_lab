# SPDX-License-Identifier: Apache-2.0
"""
Triton Fused GEMM + EXP Kernel

Demonstrates Challenge 4: Blocking Synchronization

The kernel computes: C = exp(A @ B)

On Triton's single-warp execution model:
- GEMM uses Tensor Cores (async)
- EXP uses special function unit (MUFU)
- Both operations execute on the SAME warp
- The GEMM.WAIT instruction BLOCKS EXP execution

Expected SASS pattern:
    HMMA.xxx    ; Tensor Core GEMM
    ...
    BAR.SYNC    ; or WAIT - blocks warp!
    MUFU.EX2    ; EXP only starts after GEMM completes
"""

import os
import sys
import argparse
import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any


@triton.jit
def _gemm_exp_kernel(
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
    Fused GEMM + EXP kernel: C = exp(A @ B)
    
    The key insight from the paper:
    - GEMM (tl.dot) uses tensor cores asynchronously
    - EXP on the result requires waiting for GEMM to complete
    - Since both run on the same warp, the wait blocks all operations
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # GEMM loop - uses tensor cores
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Tensor Core GEMM (async)
        accumulator = tl.dot(a, b, accumulator)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # =============================================================
    # HERE IS THE BLOCKING SYNC ISSUE!
    # 
    # The accumulator must be materialized before EXP can execute.
    # On the same warp, this forces a WAIT that blocks everything.
    # =============================================================
    
    # Apply EXP element-wise - requires GEMM result first!
    # This creates the blocking sync condition described in the paper
    c = tl.exp(accumulator)
    
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class TritonGemmExp:
    """Triton fused GEMM + EXP kernel demonstrating blocking sync issue."""
    
    name = "triton"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware_mode', 'compile_only')
        self.target_sm = self.config.get('target_sm', 100)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        """Execute C = exp(A @ B)."""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _gemm_exp_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    
    def compile(self) -> Dict[str, Any]:
        """AOT compile for sm_100 and extract IR/SASS."""
        import subprocess
        import triton.compiler as tc
        from triton.backends.compiler import GPUTarget
        
        # Signature for AOT compilation
        signature = {
            "A_ptr": "*fp32",
            "B_ptr": "*fp32",
            "C_ptr": "*fp32",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i32",
            "stride_ak": "i32",
            "stride_bk": "i32",
            "stride_bn": "i32",
            "stride_cm": "i32",
            "stride_cn": "i32",
        }
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        
        src = tc.ASTSource(
            fn=_gemm_exp_kernel,
            constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            signature=signature
        )
        
        target = GPUTarget("cuda", self.target_sm, 64)
        compiled = triton.compile(src, target=target)
        asm = compiled.asm
        
        # Create output directory
        backend_dir = os.path.join(self.output_dir, 'triton')
        os.makedirs(backend_dir, exist_ok=True)
        
        artifacts = {}
        
        # Save PTX
        if 'ptx' in asm:
            ptx_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}.ptx")
            with open(ptx_path, 'w') as f:
                f.write(asm['ptx'])
            artifacts['ptx'] = ptx_path
            print(f"  PTX saved: {ptx_path}")
        
        # Save CUBIN
        if 'cubin' in asm:
            cubin_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}.cubin")
            with open(cubin_path, 'wb') as f:
                f.write(asm['cubin'])
            artifacts['cubin'] = cubin_path
            print(f"  CUBIN saved: {cubin_path}")
            
            # Extract SASS
            sass_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}.sass")
            try:
                cuobjdump = self.config.get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
                with open(sass_path, 'w') as f:
                    subprocess.run([cuobjdump, '-sass', cubin_path], stdout=f, check=True)
                artifacts['sass'] = sass_path
                print(f"  SASS saved: {sass_path}")
            except Exception as e:
                print(f"  Warning: Could not extract SASS: {e}")
        
        # Save TTIR (Triton IR)
        if 'ttir' in asm:
            ttir_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}_ttir.txt")
            with open(ttir_path, 'w') as f:
                f.write(str(asm['ttir']))
            artifacts['ttir'] = ttir_path
            print(f"  TTIR saved: {ttir_path}")
        
        # Save TTGIR (Triton GPU IR)
        if 'ttgir' in asm:
            ttgir_path = os.path.join(backend_dir, f"gemm_exp_sm{self.target_sm}_ttgir.txt")
            with open(ttgir_path, 'w') as f:
                f.write(str(asm['ttgir']))
            artifacts['ttgir'] = ttgir_path
            print(f"  TTGIR saved: {ttgir_path}")
        
        return {
            'backend': self.name,
            'status': 'compiled',
            'target_sm': self.target_sm,
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=torch.float32, device='cuda'):
        """Create test inputs."""
        scale = 0.1  # Small values to prevent exp overflow
        A = torch.randn(m, k, dtype=dtype, device=device) * scale
        B = torch.randn(k, n, dtype=dtype, device=device) * scale
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference: C = exp(A @ B)."""
        return torch.exp(torch.matmul(A, B))


def main():
    parser = argparse.ArgumentParser(description='Triton GEMM+EXP Kernel')
    parser.add_argument('--compile-only', action='store_true', help='Only compile, do not run')
    parser.add_argument('--size', type=int, default=1024, help='Matrix size (MxNxK)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Triton GEMM+EXP Kernel - Challenge 4: Blocking Sync")
    print("=" * 60)
    
    config = {
        'hardware_mode': 'compile_only' if args.compile_only else 'native',
        'target_sm': 100,
        'output_dir': args.output_dir,
    }
    
    kernel = TritonGemmExp(config)
    
    # Compile
    print("\n[1] Compiling kernel...")
    result = kernel.compile()
    print(f"    Status: {result['status']}")
    
    if not args.compile_only:
        print(f"\n[2] Running kernel (size={args.size})...")
        A, B, C = kernel.create_inputs(args.size, args.size, args.size)
        kernel(A, B, C)
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
