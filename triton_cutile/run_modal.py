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
    
    # Set TRITON_PTXAS_PATH to use CUDA 13.1 ptxas for PTX 9.1 generation
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    
    # Verify ptxas version
    print("\n" + "=" * 60)
    print("PTX Configuration")
    print("=" * 60)
    result = subprocess.run(["/usr/local/cuda/bin/ptxas", "--version"], 
                           capture_output=True, text=True)
    print(f"TRITON_PTXAS_PATH: {os.environ.get('TRITON_PTXAS_PATH')}")
    print(f"ptxas version: {result.stdout.strip()}")
    print("Expected PTX version: 9.1")
    
    results = {"experiment": experiment, "triton": None, "cutile": None, "analysis": None, "ptx_version": "9.1"}
    
    if experiment == "04_blocking_sync":
        exp_dir = "experiments/04_blocking_sync"
        
        # Run Triton kernel
        print("\n" + "=" * 60)
        print("Running Triton GEMM+EXP...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/triton_gemm_exp.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["cutile"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Verify PTX version 9.1 was generated
        print("\n" + "=" * 60)
        print("Verifying PTX Version 9.1...")
        print("=" * 60)
        
        for name, ptx_path in [("Triton", "outputs/triton/gemm_exp_sm100.ptx"), 
                                ("CuTile", "outputs/cutile/gemm_exp_sm100.ptx")]:
            if os.path.exists(ptx_path):
                with open(ptx_path, 'r') as f:
                    for line in f:
                        if '.version' in line:
                            version = line.strip()
                            expected = ".version 9.1"
                            status = "✓" if expected in version else "✗"
                            print(f"{name}: {version} {status}")
                            break
            else:
                print(f"{name}: PTX file not found at {ptx_path}")
        
        # Run SASS and PTX analysis
        print("\n" + "=" * 60)
        print("Running SASS & PTX Analysis...")
        print("=" * 60)
        
        result = subprocess.run(
            ["python", f"{exp_dir}/analyze.py", "--output-dir", "outputs"],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
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
    
    elif experiment == "02_register_pressure":
        exp_dir = "experiments/02_register_pressure"
        
        # Run Triton kernel with deep pipeline
        print("\n" + "=" * 60)
        print("Running Triton Deep Pipeline GEMM (256x256x64, stages=4)...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/triton_deep_pipeline.py", "--output-dir", "outputs", "--num-stages", "4"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["triton"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run CuTile kernel
        print("\n" + "=" * 60)
        print("Running CuTile Deep Pipeline GEMM (256x256x64)...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/cutile_deep_pipeline.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["cutile"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run analysis
        print("\n" + "=" * 60)
        print("Running SASS Analysis for Register Pressure...")
        print("=" * 60)
        
        result = subprocess.run(
            ["python", f"{exp_dir}/analyze.py", "--output-dir", "outputs"],
            capture_output=True, text=True
        )
        print(result.stdout)
        results["analysis"] = {"output": result.stdout}
    
    elif experiment == "03_variable_latency":
        exp_dir = "experiments/03_variable_latency"
        
        # Run Triton strided GEMM
        print("\n" + "=" * 60)
        print("Running Triton Strided GEMM (stride=1, baseline)...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/triton_strided_gemm.py", "--output-dir", "outputs", "--stride", "1"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["triton"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run CuTile strided GEMM
        print("\n" + "=" * 60)
        print("Running CuTile Strided GEMM with Async TMA...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/cutile_strided_gemm.py", "--output-dir", "outputs", "--stride", "1"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["cutile"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run analysis
        print("\n" + "=" * 60)
        print("Running SASS Analysis for Variable Latency...")
        print("=" * 60)
        
        result = subprocess.run(
            ["python", f"{exp_dir}/analyze.py", "--output-dir", "outputs"],
            capture_output=True, text=True
        )
        print(result.stdout)
        results["analysis"] = {"output": result.stdout}
    
    elif experiment == "05_gather_latency":
        exp_dir = "experiments/05_gather_latency"
        
        # Run Triton gather
        print("\n" + "=" * 60)
        print("Running Triton Gather (Embedding Lookup)...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/triton_gather.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["triton"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run CuTile gather
        print("\n" + "=" * 60)
        print("Running CuTile Gather with TMA...")
        print("=" * 60)
        
        cmd = ["python", f"{exp_dir}/cutile_gather.py", "--output-dir", "outputs"]
        if compile_only:
            cmd.append("--compile-only")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        results["cutile"] = {"returncode": result.returncode, "output": result.stdout}
        
        # Run analysis
        print("\n" + "=" * 60)
        print("Running SASS Analysis for Gather Operations...")
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
    experiments = ["01_cooperative_warps", "02_register_pressure", "03_variable_latency", "04_blocking_sync", "05_gather_latency"]
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
    
    # Set TRITON_PTXAS_PATH to use CUDA 13.1 ptxas for PTX 9.1 generation
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {experiment} (warmup={num_warmup}, runs={num_runs})")
    print(f"Using ptxas: {os.environ.get('TRITON_PTXAS_PATH')} (PTX 9.1)")
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
    
    elif experiment == "02_register_pressure":
        # Import kernels for deep pipeline GEMM
        import sys
        sys.path.insert(0, "/app/experiments/02_register_pressure")
        from triton_deep_pipeline import TritonDeepPipeline
        from cutile_deep_pipeline import CutileDeepPipeline
        
        results = {"experiment": experiment, "benchmarks": []}
        M, K, N = 4096, 4096, 4096  # Fixed matrix size
        
        # Test different tile sizes and stages - CuTile requires power-of-2!
        tile_configs = [
            # (tile_m, tile_n, num_stages)
            (128, 128, 4),   # Baseline
            (128, 128, 6),   # More stages
            (256, 256, 2),   # Larger tiles, fewer stages
            (256, 256, 3),   # Larger tiles, medium stages
            (256, 256, 4),   # Larger tiles, more stages - Triton spilling?
        ]
        
        for tile_m, tile_n, stages in tile_configs:
            print(f"\n{'='*60}")
            print(f"[Tiles: {tile_m}x{tile_n}x64, stages={stages}, Matrix: {M}x{K}x{N}]")
            print("="*60)
            
            # Create inputs
            A, B, C_triton = TritonDeepPipeline.create_inputs(M, N, K, device='cuda')
            _, _, C_cutile = TritonDeepPipeline.create_inputs(M, N, K, device='cuda')
            reference = TritonDeepPipeline.reference(A, B)
            
            triton_time = None
            triton_tflops = None
            triton_error = None
            triton_status = "ok"
            
            # Try Triton - may fail with large tiles
            try:
                triton_kernel = TritonDeepPipeline({
                    'hardware_mode': 'native', 
                    'output_dir': 'outputs',
                    'num_stages': stages,
                    'BLOCK_M': tile_m,
                    'BLOCK_N': tile_n,
                })
                # Override tile sizes
                triton_kernel.BLOCK_M = tile_m
                triton_kernel.BLOCK_N = tile_n
                triton_kernel.num_stages = stages
                
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
                flops = 2 * M * N * K
                triton_tflops = flops / (triton_time / 1000) / 1e12
                
                print(f"  Triton: {triton_time:.3f} ms, {triton_tflops:.2f} TFLOPS, max_error={triton_error:.2e}")
            except Exception as e:
                triton_status = f"FAILED: {type(e).__name__}"
                print(f"  Triton: FAILED - {type(e).__name__}: {str(e)[:100]}")
            
            # CuTile - should work with larger tiles due to TMA staging
            cutile_time = None
            cutile_tflops = None
            cutile_error = None
            cutile_status = "ok"
            
            try:
                cutile_kernel = CutileDeepPipeline({'hardware_mode': 'native', 'output_dir': 'outputs'})
                # Override tile sizes
                cutile_kernel.tm = tile_m
                cutile_kernel.tn = tile_n
                
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
                flops = 2 * M * N * K
                cutile_tflops = flops / (cutile_time / 1000) / 1e12
                
                print(f"  CuTile: {cutile_time:.3f} ms, {cutile_tflops:.2f} TFLOPS, max_error={cutile_error:.2e}")
            except Exception as e:
                cutile_status = f"FAILED: {type(e).__name__}"
                print(f"  CuTile: FAILED - {type(e).__name__}: {str(e)[:100]}")
            
            # Speedup if both succeeded
            if triton_time and cutile_time:
                speedup = triton_time / cutile_time
                print(f"  Speedup (CuTile/Triton): {speedup:.2f}x")
            elif cutile_time and not triton_time:
                print(f"  ✓ CuTile WINS by default (Triton failed with this tile size!)")
            
            results["benchmarks"].append({
                "tile_m": tile_m, "tile_n": tile_n, "stages": stages,
                "M": M, "K": K, "N": N,
                "triton_status": triton_status,
                "cutile_status": cutile_status,
                "triton_ms": triton_time,
                "cutile_ms": cutile_time,
                "triton_tflops": triton_tflops,
                "cutile_tflops": cutile_tflops,
            })
        
        return results
    
    elif experiment == "03_variable_latency":
        # Import kernels for strided GEMM
        import sys
        sys.path.insert(0, "/app/experiments/03_variable_latency")
        from triton_strided_gemm import TritonStridedGemm
        from cutile_strided_gemm import CutileStridedGemm
        
        results = {"experiment": experiment, "benchmarks": []}
        M, K, N = 4096, 4096, 4096
        
        # Test different stride patterns
        strides = [1, 4, 16, 64]  # 1=normal, higher=more cache misses
        
        for stride in strides:
            print(f"\n{'='*60}")
            print(f"[Stride: {stride}, Matrix: {M}x{K}x{N}]")
            print("="*60)
            
            A, B, C_triton = TritonStridedGemm.create_inputs(M, N, K, device='cuda')
            _, _, C_cutile = TritonStridedGemm.create_inputs(M, N, K, device='cuda')
            reference = TritonStridedGemm.reference(A, B)
            
            # Triton benchmark
            triton_kernel = TritonStridedGemm({
                'hardware_mode': 'native',
                'output_dir': 'outputs',
                'access_stride': stride,
            })
            
            for _ in range(num_warmup):
                C_triton.zero_()
                triton_kernel(A, B, C_triton)
            torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                C_triton.zero_()
                triton_kernel(A, B, C_triton)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / num_runs * 1000
            
            triton_error = (C_triton.float() - reference.float()).abs().max().item()
            
            # CuTile benchmark
            cutile_kernel = CutileStridedGemm({
                'hardware_mode': 'native',
                'output_dir': 'outputs',
                'access_stride': stride,
            })
            
            for _ in range(num_warmup):
                C_cutile.zero_()
                cutile_kernel(A, B, C_cutile)
            torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                C_cutile.zero_()
                cutile_kernel(A, B, C_cutile)
            torch.cuda.synchronize()
            cutile_time = (time.perf_counter() - start) / num_runs * 1000
            
            cutile_error = (C_cutile.float() - reference.float()).abs().max().item()
            
            flops = 2 * M * N * K
            triton_tflops = flops / (triton_time / 1000) / 1e12
            cutile_tflops = flops / (cutile_time / 1000) / 1e12
            speedup = triton_time / cutile_time
            
            print(f"  Triton: {triton_time:.3f} ms, {triton_tflops:.2f} TFLOPS, max_error={triton_error:.2e}")
            print(f"  CuTile: {cutile_time:.3f} ms, {cutile_tflops:.2f} TFLOPS, max_error={cutile_error:.2e}")
            print(f"  Speedup (CuTile/Triton): {speedup:.2f}x")
            
            results["benchmarks"].append({
                "stride": stride, "M": M, "K": K, "N": N,
                "triton_ms": triton_time,
                "cutile_ms": cutile_time,
                "triton_tflops": triton_tflops,
                "cutile_tflops": cutile_tflops,
                "speedup": speedup,
            })
        
        return results
    
    elif experiment == "05_gather_latency":
        # Import kernels for gather operations
        import sys
        sys.path.insert(0, "/app/experiments/05_gather_latency")
        from triton_gather import TritonGather
        from cutile_gather import CutileGather
        
        results = {"experiment": experiment, "benchmarks": []}
        
        # Test different configurations to see memory latency effects
        configs = [
            # (table_size, embed_dim, batch_size, random_indices)
            (1024, 512, 4096, True),      # Small table - fits in L2
            (65536, 512, 4096, True),     # Large table - L2 misses
            (262144, 512, 4096, True),    # Very large - DRAM bound
            
            # Larger batch sizes to make kernel runtime dominate over launch overhead
            (65536, 1024, 65536, True),     # Large batch, random
            (65536, 1024, 65536, False),    # Large batch, sequential
            (262144, 1024, 65536, True),    # Very large table, random
            (1048576, 512, 131072, True),   # Huge table, huge batch
            (1048576, 512, 131072, False),  # Huge table, sequential
        ]
        
        for table_size, embed_dim, batch_size, random in configs:
            pattern = "random" if random else "sequential"
            print(f"\n{'='*60}")
            print(f"[Table: {table_size}, Embed: {embed_dim}, Batch: {batch_size}, {pattern}]")
            print("="*60)
            
            table, indices, out_triton = TritonGather.create_inputs(
                table_size, embed_dim, batch_size, random_indices=random, device='cuda')
            _, _, out_cutile = TritonGather.create_inputs(
                table_size, embed_dim, batch_size, random_indices=False, device='cuda')
            reference = TritonGather.reference(table, indices)
            
            # Triton benchmark
            triton_kernel = TritonGather({'hardware_mode': 'native', 'output_dir': 'outputs'})
            
            for _ in range(num_warmup):
                triton_kernel(table, indices, out_triton)
            torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                triton_kernel(table, indices, out_triton)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / num_runs * 1000
            
            triton_error = (out_triton.float() - reference.float()).abs().max().item()
            
            # CuTile benchmark
            cutile_kernel = CutileGather({'hardware_mode': 'native', 'output_dir': 'outputs'})
            
            for _ in range(num_warmup):
                cutile_kernel(table, indices, out_cutile)
            torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                cutile_kernel(table, indices, out_cutile)
            torch.cuda.synchronize()
            cutile_time = (time.perf_counter() - start) / num_runs * 1000
            
            cutile_error = (out_cutile.float() - reference.float()).abs().max().item()
            
            # Bandwidth (GB/s) - table[idx] read + output write
            bytes_moved = batch_size * embed_dim * 2 * 2  # fp16 = 2 bytes
            triton_gbps = bytes_moved / (triton_time / 1000) / 1e9
            cutile_gbps = bytes_moved / (cutile_time / 1000) / 1e9
            speedup = triton_time / cutile_time
            
            print(f"  Triton: {triton_time:.3f} ms, {triton_gbps:.1f} GB/s, error={triton_error:.2e}")
            print(f"  CuTile: {cutile_time:.3f} ms, {cutile_gbps:.1f} GB/s, error={cutile_error:.2e}")
            print(f"  Speedup (CuTile/Triton): {speedup:.2f}x")
            
            results["benchmarks"].append({
                "table_size": table_size, "embed_dim": embed_dim, 
                "batch_size": batch_size, "pattern": pattern,
                "triton_ms": triton_time, "cutile_ms": cutile_time,
                "triton_gbps": triton_gbps, "cutile_gbps": cutile_gbps,
                "speedup": speedup,
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
