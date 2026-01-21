# SPDX-License-Identifier: Apache-2.0
"""
Triton matrix multiplication kernel with support for:
- Native mode: JIT compilation on Blackwell (sm_100)
- Compile-only mode: AOT compilation for sm_100 on any GPU
"""

import os
import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any


@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """Triton kernel for matrix multiplication C = A @ B."""
    # Program ID
    pid = tl.program_id(0)
    
    # Number of blocks in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzle for better L2 cache utilization
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block starting positions - NO modulo wrapping
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first block of A and B
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator in fp32 for precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K-loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load with proper bounds checking for M, N, and K dimensions
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply accumulate
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply output type conversion
    c = accumulator.to(tl.float16) if C_ptr.dtype.element_ty == tl.float16 else accumulator
    
    # Store result with proper bounds checking
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class TritonMatmul:
    """Triton-based matrix multiplication kernel."""
    
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
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        self._compiled = None
    
    def _get_tile_sizes(self, dtype, M, N, K):
        """Get optimal tile sizes based on dtype and matrix dimensions."""
        if dtype in (torch.float16, torch.bfloat16):
            # Larger tiles for Tensor Core friendly types
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        else:
            # Smaller tiles for fp32
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        
        # Ensure tile sizes don't exceed matrix dimensions (for efficiency)
        # But don't go below minimum for tl.dot
        BLOCK_M = min(BLOCK_M, max(16, M))
        BLOCK_N = min(BLOCK_N, max(16, N))
        BLOCK_K = min(BLOCK_K, max(16, K))
        
        # Triton tl.dot requires block sizes to be power of 2 and >= 16
        def next_power_of_2(x):
            return 1 << (x - 1).bit_length()
        
        BLOCK_M = next_power_of_2(BLOCK_M)
        BLOCK_N = next_power_of_2(BLOCK_N)
        BLOCK_K = next_power_of_2(BLOCK_K)
        
        return BLOCK_M, BLOCK_N, BLOCK_K
    
    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        """
        Perform matrix multiplication: C = A @ B.
        
        Args:
            A: First input matrix (M x K)
            B: Second input matrix (K x N)
            C: Output matrix (M x N, will be modified in-place)
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Incompatible dimensions: A is {A.shape}, B is {B.shape}"
        
        # Get optimal tile sizes
        BLOCK_M, BLOCK_N, BLOCK_K = self._get_tile_sizes(A.dtype, M, N, K)
        
        # Grid
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )
        
        # Launch kernel
        _matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    
    def compile(self, visualize: bool = False) -> Dict[str, Any]:
        """
        Compile the kernel for sm_100 (AOT compilation).
        
        Args:
            visualize: If True, also generate dependency graph visualization
        
        Returns:
            Dict with compilation artifacts
        """
        import subprocess
        import triton.compiler as tc
        from triton.backends.compiler import GPUTarget
        
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
        
        # Use 32x32x32 block sizes for standardized comparison with other backends
        # This is the minimum compatible size for tl.dot (requires >= 16)
        src = tc.ASTSource(
            fn=_matmul_kernel,
            constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            signature=signature
        )
        
        target = GPUTarget("cuda", self.target_sm, 64)
        compiled_result = triton.compile(src, target=target)
        asm = compiled_result.asm
        self._compiled = asm
        
        # Get matrix dimensions from config for filename (default: 8192x8192x8192)
        compile_dims = self.config.get('kernels', {}).get('matmul', {}).get('compile_dims', [8192, 8192, 8192])
        m, n, k = compile_dims[0], compile_dims[1], compile_dims[2]
        dim_suffix = f"_{m}x{n}x{k}"
        
        artifacts = {}
        
        # Create backend-specific output directory
        backend_output_dir = os.path.join(self.output_dir, 'triton_matmul')
        os.makedirs(backend_output_dir, exist_ok=True)
        
        # Save PTX
        if 'ptx' in asm:
            ptx_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.ptx")
            with open(ptx_path, 'w') as f:
                f.write(asm['ptx'])
            artifacts['ptx'] = ptx_path
        
        # Save CUBIN
        if 'cubin' in asm:
            cubin_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.cubin")
            with open(cubin_path, 'wb') as f:
                f.write(asm['cubin'])
            artifacts['cubin'] = cubin_path
            
            # Extract SASS
            sass_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}.sass")
            cuobjdump_path = self.config.get('hardware', {}).get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
            
            try:
                with open(sass_path, 'w') as f:
                    subprocess.run(
                        [cuobjdump_path, '-sass', cubin_path],
                        stdout=f,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )
                artifacts['sass'] = sass_path
                print(f"    SASS extracted: {sass_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # Save TTIR (Triton IR)
        if 'ttir' in asm:
            ttir_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}_ttir.txt")
            with open(ttir_path, 'w') as f:
                print(asm['ttir'], file=f)
            artifacts['ttir'] = ttir_path
            print(f"    TTIR saved: {ttir_path}")
        
        # Save TTGIR (Triton GPU IR)
        if 'ttgir' in asm:
            ttgir_path = os.path.join(backend_output_dir, f"matmul_sm{self.target_sm}{dim_suffix}_ttgir.txt")
            with open(ttgir_path, 'w') as f:
                print(asm['ttgir'], file=f)
            artifacts['ttgir'] = ttgir_path
            print(f"    TTGIR saved: {ttgir_path}")
            
            # Generate dependency graph visualization if requested
            if visualize:
                try:
                    from analysis.ttgir_visualizer import visualize_ttgir
                    html_path = visualize_ttgir(asm['ttgir'], backend_output_dir)
                    artifacts['visualization'] = html_path
                    print(f"    Visualization: {html_path}")
                except Exception as e:
                    print(f"    Warning: Could not generate visualization: {e}")
        
        return {
            'backend': self.name,
            'status': 'compiled',
            'target_sm': self.target_sm,
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(m: int, n: int, k: int, dtype=torch.float32, device='cuda'):
        """
        Create input matrices for testing/benchmarking.
        
        Args:
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Number of columns in A / rows in B
            dtype: Data type
            device: Device to create tensors on
            
        Returns:
            Tuple of (A, B, C) tensors
        """
        # Scale inputs to control accumulation magnitude (standard matmul init)
        scale = 1.0 / (k ** 0.5)
        A = torch.randn(m, k, dtype=dtype, device=device) * scale
        B = torch.randn(k, n, dtype=dtype, device=device) * scale
        C = torch.empty(m, n, dtype=dtype, device=device)
        return A, B, C
    
    @staticmethod
    def reference(A, B):
        """Reference implementation."""
        return torch.matmul(A, B)
