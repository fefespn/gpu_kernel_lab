#!/usr/bin/env python3
"""
Modal.com deployment for triton_cutile experiments on B200 GPUs.

Usage:
    # Run specific experiment
    modal run run_modal.py --experiment 04_blocking_sync
    
    # Run all experiments
    modal run run_modal.py --all
    
    # Compile only (no execution)
    modal run run_modal.py --experiment 04_blocking_sync --compile-only
"""

import modal
import os

app = modal.App("triton-vs-cutile")

# CUDA 13.1 for Blackwell support
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu24.04",
        add_python="3.11"
    )
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so.12",
        "ln -sf /usr/local/cuda/lib64/libnvrtc-builtins.so.13.1 /usr/local/cuda/lib64/libnvrtc-builtins.so.12",
        "ldconfig",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "cuda-tile",
        "torch",
        "triton",
    )
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[".git", "__pycache__", "venv", "outputs", "*.pyc"]
    )
)


@app.function(gpu="B200", image=cuda_image, timeout=600)
def run_experiment(experiment: str, compile_only: bool = False):
    """Run a specific experiment."""
    import subprocess
    import os
    
    os.chdir("/app")
    os.makedirs("outputs/triton", exist_ok=True)
    os.makedirs("outputs/cutile", exist_ok=True)
    
    results = {"experiment": experiment, "triton": None, "cutile": None, "analysis": None}
    
    if experiment == "04_blocking_sync":
        exp_dir = "experiments/04_blocking_sync"
        
        # Run Triton kernel
        print("\n" + "=" * 60)
        print("Running Triton GEMM+EXP...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/triton_gemm_exp.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["triton"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run CuTile kernel
        print("\n" + "=" * 60)
        print("Running CuTile GEMM+EXP...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/cutile_gemm_exp.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["cutile"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run analysis
        print("\n" + "=" * 60)
        print("Running SASS Analysis...")
        print("=" * 60)
        
        result = subprocess.run(
            ["python", f"{exp_dir}/analyze.py", "--output-dir", "outputs"],
            capture_output=True, text=True
        )
        print(result.stdout)
        results["analysis"] = {"output": result.stdout}
    
    elif experiment == "01_cooperative_warps":
        exp_dir = "experiments/01_cooperative_warps"
        
        # Run Triton kernel
        print("\n" + "=" * 60)
        print("Running Triton Large-Tile GEMM (128x128x64)...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/triton_large_tile.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["triton"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run CuTile kernel
        print("\n" + "=" * 60)
        print("Running CuTile Large-Tile GEMM (128x128x64)...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/cutile_large_tile.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["cutile"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run analysis
        print("\n" + "=" * 60)
        print("Running SASS Analysis for Cooperative Warps...")
        print("=" * 60)
        
        result = subprocess.run(
            ["python", f"{exp_dir}/analyze.py", "--output-dir", "outputs"],
            capture_output=True, text=True
        )
        print(result.stdout)
        results["analysis"] = {"output": result.stdout}
    
    return results


@app.function(gpu="B200", image=cuda_image, timeout=1800)
def run_all_experiments(compile_only: bool = False):
    """Run all experiments."""
    experiments = ["01_cooperative_warps", "04_blocking_sync"]
    results = {}
    
    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {exp}")
        print("=" * 60)
        results[exp] = run_experiment.local(exp, compile_only)
    
    return results


@app.function(gpu="B200", image=cuda_image, timeout=1800)
def run_benchmarks(experiment: str = "04_blocking_sync", num_warmup: int = 5, num_runs: int = 20):
    """Run speed and accuracy benchmarks for an experiment."""
    import subprocess
    import os
    import time
    import torch
    
    os.chdir("/app")
    
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {experiment} (warmup={num_warmup}, runs={num_runs})")
    print("=" * 70)
    
    if experiment == "04_blocking_sync":
        # Import kernels
        import sys
        sys.path.insert(0, "/app/experiments/04_blocking_sync")
        from triton_gemm_exp import TritonGemmExp
        from cutile_gemm_exp import CutileGemmExp
        
        results = {"experiment": experiment, "benchmarks": []}
        
        # Test configurations: (M, K, N)
        # M is batch size, K is hidden inner dimension, K (KV Projection Size: 8 heads* 128 dim)
        sizes = [
            (32, 8192, 1024),
            (8192, 8192, 1024),
            (8192, 8192, 128),
        ]
        
        for M, K, N in sizes:
            print(f"\n[Matrix Size: {M}x{K}x{N}]")
            
            # Create inputs
            A, B, C_triton = TritonGemmExp.create_inputs(M, N, K, device='cuda')
            _, _, C_cutile = TritonGemmExp.create_inputs(M, N, K, device='cuda')
            
            # Reference
            reference = TritonGemmExp.reference(A, B)
            
            # Triton benchmark
            triton_kernel = TritonGemmExp({'hardware_mode': 'native', 'output_dir': 'outputs'})
            
            # Warmup
            for _ in range(num_warmup):
                C_triton.zero_()
                triton_kernel(A, B, C_triton)
            torch.cuda.synchronize()
            
            # Timed runs
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                C_triton.zero_()
                triton_kernel(A, B, C_triton)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / num_runs * 1000  # ms
            
            triton_error = (C_triton - reference).abs().max().item()
            
            # CuTile benchmark
            cutile_kernel = CutileGemmExp({'hardware_mode': 'native', 'output_dir': 'outputs'})
            
            # Warmup
            for _ in range(num_warmup):
                C_cutile.zero_()
                cutile_kernel(A, B, C_cutile)
            torch.cuda.synchronize()
            
            # Timed runs
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                C_cutile.zero_()
                cutile_kernel(A, B, C_cutile)
            torch.cuda.synchronize()
            cutile_time = (time.perf_counter() - start) / num_runs * 1000  # ms
            
            cutile_error = (C_cutile - reference).abs().max().item()
            
            # Calculate TFLOPS (2*M*N*K for matmul + M*N for exp)
            flops = (2 * M * N * K + M * N) 
            triton_tflops = flops / (triton_time / 1000) / 1e12
            cutile_tflops = flops / (cutile_time / 1000) / 1e12
            
            speedup = triton_time / cutile_time
            
            print(f"  Triton: {triton_time:.3f} ms, {triton_tflops:.2f} TFLOPS, max_error={triton_error:.2e}")
            print(f"  CuTile: {cutile_time:.3f} ms, {cutile_tflops:.2f} TFLOPS, max_error={cutile_error:.2e}")
            print(f"  Speedup (CuTile/Triton): {speedup:.2f}x")
            
            results["benchmarks"].append({
                "M": M, "K": K, "N": N,
                "triton_ms": triton_time,
                "cutile_ms": cutile_time,
                "triton_tflops": triton_tflops,
                "cutile_tflops": cutile_tflops,
                "speedup": speedup,
                "triton_error": triton_error,
                "cutile_error": cutile_error,
            })
        
        return results
    
    elif experiment == "01_cooperative_warps":
        # Import kernels for large-tile GEMM
        import sys
        sys.path.insert(0, "/app/experiments/01_cooperative_warps")
        from triton_large_tile import TritonLargeTileGemm
        from cutile_large_tile import CutileLargeTileGemm
        
        results = {"experiment": experiment, "benchmarks": []}
        
        # Systematic test: varying which dimension is small
        sizes = [
            # Group 1: Small N (output width) - hypothesis: CuTile wins
            (4096, 4096, 128),   # CuTile should win
            (8192, 4096, 128),   # CuTile should win
            (4096, 8192, 128),   # Large K, small N - CuTile should win
            
            # Group 2: Small K (inner dim) - hypothesis: Triton wins  
            (4096, 128, 4096),   # Few GEMM iterations - Triton should win
            (8192, 128, 4096),   # Few GEMM iterations - Triton should win
            
            # Group 3: Large K, moderate N - compare
            (4096, 8192, 512),   # Many iterations, moderate output
            (8192, 8192, 256),   # Many iterations, narrow output
        ]
        
        for M, K, N in sizes:
            print(f"\n[Matrix Size: {M}x{K}x{N} with 128x128x64 tiles]")
            
            # Create inputs (fp16 for tensor cores)
            A, B, C_triton = TritonLargeTileGemm.create_inputs(M, N, K, device='cuda')
            _, _, C_cutile = TritonLargeTileGemm.create_inputs(M, N, K, device='cuda')
            
            # Reference
            reference = TritonLargeTileGemm.reference(A, B)
            
            # Triton benchmark
            triton_kernel = TritonLargeTileGemm({'hardware_mode': 'native', 'output_dir': 'outputs'})
            
            # Warmup
            for _ in range(num_warmup):
                C_triton.zero_()
                triton_kernel(A, B, C_triton)
            torch.cuda.synchronize()
            
            # Timed runs
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                C_triton.zero_()
                triton_kernel(A, B, C_triton)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / num_runs * 1000
            
            triton_error = (C_triton.float() - reference.float()).abs().max().item()
            
            # CuTile benchmark
            cutile_kernel = CutileLargeTileGemm({'hardware_mode': 'native', 'output_dir': 'outputs'})
            
            # Warmup
            for _ in range(num_warmup):
                C_cutile.zero_()
                cutile_kernel(A, B, C_cutile)
            torch.cuda.synchronize()
            
            # Timed runs
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                C_cutile.zero_()
                cutile_kernel(A, B, C_cutile)
            torch.cuda.synchronize()
            cutile_time = (time.perf_counter() - start) / num_runs * 1000
            
            cutile_error = (C_cutile.float() - reference.float()).abs().max().item()
            
            # Calculate TFLOPS (2*M*N*K for matmul)
            flops = 2 * M * N * K
            triton_tflops = flops / (triton_time / 1000) / 1e12
            cutile_tflops = flops / (cutile_time / 1000) / 1e12
            
            speedup = triton_time / cutile_time
            
            print(f"  Triton: {triton_time:.3f} ms, {triton_tflops:.2f} TFLOPS, max_error={triton_error:.2e}")
            print(f"  CuTile: {cutile_time:.3f} ms, {cutile_tflops:.2f} TFLOPS, max_error={cutile_error:.2e}")
            print(f"  Speedup (CuTile/Triton): {speedup:.2f}x")
            
            results["benchmarks"].append({
                "M": M, "K": K, "N": N,
                "triton_ms": triton_time,
                "cutile_ms": cutile_time,
                "triton_tflops": triton_tflops,
                "cutile_tflops": cutile_tflops,
                "speedup": speedup,
                "triton_error": triton_error,
                "cutile_error": cutile_error,
            })
        
        return results
    
    return {"experiment": experiment, "error": "Unknown experiment"}


@app.local_entrypoint()
def main(
    experiment: str = None,
    all: bool = False,
    compile_only: bool = False,
    benchmark: bool = False,
):
    """
    Run triton_cutile experiments on Modal B200.
    
    Args:
        experiment: Specific experiment to run (e.g., "04_blocking_sync")
        all: Run all experiments
        compile_only: Only compile, don't execute
        benchmark: Run speed and accuracy benchmarks
    """
    print("=" * 60)
    print("Triton vs CuTile - Running on Modal B200")
    print("=" * 60)
    
    if benchmark:
        exp = experiment or "04_blocking_sync"
        print(f"\n⚡ Running benchmarks for {exp}...")
        result = run_benchmarks.remote(exp)
    elif all:
        result = run_all_experiments.remote(compile_only)
    elif experiment:
        result = run_experiment.remote(experiment, compile_only)
    else:
        print("\nUsage:")
        print("  modal run run_modal.py --experiment 04_blocking_sync")
        print("  modal run run_modal.py --experiment 04_blocking_sync --benchmark")
        print("  modal run run_modal.py --all")
        print("  modal run run_modal.py --experiment 04_blocking_sync --compile-only")
        return
    
    print("\n" + "=" * 60)
    print("✓ Experiment completed!")
    print("=" * 60)


@app.function(gpu="B200", image=cuda_image, timeout=3600)
def interactive_session():
    """Start interactive session for debugging."""
    import os
    import subprocess
    
    os.chdir("/app")
    
    print("Interactive session ready!")
    print(f"CWD: {os.getcwd()}")
    
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
    
    return {"status": "ready"}
