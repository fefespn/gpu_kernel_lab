# SPDX-License-Identifier: Apache-2.0
"""
Metrics calculation for GPU kernel benchmarking.
Includes latency, TFLOPS, and memory bandwidth computations.
"""

import time
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    backend: str
    size: int
    dtype: str
    latency_us: float  # Microseconds
    latency_std_us: float
    tflops: float
    memory_bandwidth_gbps: float
    iterations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backend': self.backend,
            'size': self.size,
            'dtype': self.dtype,
            'latency_us': self.latency_us,
            'latency_std_us': self.latency_std_us,
            'tflops': self.tflops,
            'memory_bandwidth_gbps': self.memory_bandwidth_gbps,
            'iterations': self.iterations
        }


def compute_latency(
    fn: Callable,
    args: tuple,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    sync_fn: Optional[Callable] = None
) -> tuple[float, float, List[float]]:
    """
    Compute kernel latency with warmup.
    
    Args:
        fn: Function to benchmark
        args: Arguments to pass to fn
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of timed runs
        sync_fn: Synchronization function (e.g., torch.cuda.synchronize)
        
    Returns:
        Tuple of (median_latency_us, std_latency_us, all_latencies_us)
    """
    import numpy as np
    
    # Warmup
    for _ in range(warmup_iterations):
        fn(*args)
        if sync_fn:
            sync_fn()
    
    # Benchmark
    latencies = []
    for _ in range(benchmark_iterations):
        start = time.perf_counter()
        fn(*args)
        if sync_fn:
            sync_fn()
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # Convert to microseconds
    
    latencies = np.array(latencies)
    
    # Use median to reduce impact of outliers
    median_latency = float(np.median(latencies))
    std_latency = float(np.std(latencies))
    
    return median_latency, std_latency, latencies.tolist()


def compute_latency_cuda_events(
    fn: Callable,
    args: tuple,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    use_torch: bool = True
) -> tuple[float, float, List[float]]:
    """
    Compute kernel latency using CUDA events for more accurate timing.
    
    Args:
        fn: Function to benchmark
        args: Arguments to pass to fn
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of timed runs
        use_torch: If True, use PyTorch CUDA events, else use CuPy
        
    Returns:
        Tuple of (median_latency_us, std_latency_us, all_latencies_us)
    """
    import numpy as np
    
    if use_torch:
        import torch
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        sync_fn = torch.cuda.synchronize
    else:
        import cupy as cp
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        sync_fn = cp.cuda.Stream.null.synchronize
    
    # Warmup
    for _ in range(warmup_iterations):
        fn(*args)
        sync_fn()
    
    # Benchmark with CUDA events
    latencies = []
    for _ in range(benchmark_iterations):
        start_event.record()
        fn(*args)
        end_event.record()
        
        if use_torch:
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event) * 1000)  # ms to us
        else:
            end_event.synchronize()
            elapsed = cp.cuda.get_elapsed_time(start_event, end_event) * 1000
            latencies.append(elapsed)
    
    latencies = np.array(latencies)
    median_latency = float(np.median(latencies))
    std_latency = float(np.std(latencies))
    
    return median_latency, std_latency, latencies.tolist()


def compute_tflops(
    flops: int,
    latency_us: float
) -> float:
    """
    Compute TFLOPS from FLOP count and latency.
    
    Args:
        flops: Number of floating-point operations
        latency_us: Latency in microseconds
        
    Returns:
        TFLOPS (Tera FLOPS)
    """
    if latency_us <= 0:
        return 0.0
    
    latency_s = latency_us / 1e6
    tflops = (flops / latency_s) / 1e12
    return tflops


def compute_memory_bandwidth(
    bytes_transferred: int,
    latency_us: float
) -> float:
    """
    Compute memory bandwidth in GB/s.
    
    Args:
        bytes_transferred: Total bytes read + written
        latency_us: Latency in microseconds
        
    Returns:
        Memory bandwidth in GB/s
    """
    if latency_us <= 0:
        return 0.0
    
    latency_s = latency_us / 1e6
    bandwidth_gbps = (bytes_transferred / latency_s) / 1e9
    return bandwidth_gbps


def get_vector_add_metrics(size: int, dtype_bytes: int = 4) -> Dict[str, int]:
    """
    Get theoretical metrics for vector addition.
    
    Args:
        size: Number of elements
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        
    Returns:
        Dict with 'flops' and 'bytes' keys
    """
    # Vector add: c = a + b
    # 1 FLOP per element (the addition)
    flops = size
    
    # Memory: read a, read b, write c
    bytes_transferred = 3 * size * dtype_bytes
    
    return {
        'flops': flops,
        'bytes': bytes_transferred
    }


def get_matmul_metrics(m: int, n: int, k: int, dtype_bytes: int = 4) -> Dict[str, int]:
    """
    Get theoretical metrics for matrix multiplication C = A @ B.
    
    Args:
        m: Number of rows in A and C
        n: Number of columns in B and C
        k: Number of columns in A / rows in B
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        
    Returns:
        Dict with 'flops' and 'bytes' keys
    """
    # GEMM: 2*M*N*K FLOPs (multiply-add = 2 ops per element)
    flops = 2 * m * n * k
    
    # Memory: read A (M*K), read B (K*N), write C (M*N)
    bytes_transferred = (m * k + k * n + m * n) * dtype_bytes
    
    return {
        'flops': flops,
        'bytes': bytes_transferred
    }


def benchmark_kernel(
    kernel,
    size: int,
    dtype_str: str = 'float32',
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    use_cuda_events: bool = True
) -> BenchmarkResult:
    """
    Benchmark a kernel with full metrics.
    
    Args:
        kernel: Kernel instance with __call__ and create_inputs methods
        size: Input size
        dtype_str: Data type string ('float32' or 'float16')
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of benchmark runs
        use_cuda_events: Use CUDA events for timing
        
    Returns:
        BenchmarkResult with all metrics
    """
    import torch
    import numpy as np
    
    # Determine dtype
    dtype_bytes = 4 if dtype_str == 'float32' else 2
    dtype = torch.float32 if dtype_str == 'float32' else torch.float16
    
    # Create inputs
    try:
        a, b, c = kernel.create_inputs(size, dtype=dtype)
    except TypeError:
        # Some kernels might not support dtype parameter
        a, b, c = kernel.create_inputs(size)
    
    # Benchmark
    def run():
        kernel(a, b, c)
    
    if use_cuda_events:
        try:
            latency_us, std_us, _ = compute_latency_cuda_events(
                run, (),
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                use_torch=hasattr(a, 'device')  # True for PyTorch tensors
            )
        except Exception:
            # Fallback to CPU timing
            latency_us, std_us, _ = compute_latency(
                run, (),
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                sync_fn=torch.cuda.synchronize
            )
    else:
        latency_us, std_us, _ = compute_latency(
            run, (),
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            sync_fn=torch.cuda.synchronize
        )
    
    # Compute metrics
    metrics = get_vector_add_metrics(size, dtype_bytes)
    tflops = compute_tflops(metrics['flops'], latency_us)
    bandwidth = compute_memory_bandwidth(metrics['bytes'], latency_us)
    
    return BenchmarkResult(
        backend=kernel.name,
        size=size,
        dtype=dtype_str,
        latency_us=latency_us,
        latency_std_us=std_us,
        tflops=tflops,
        memory_bandwidth_gbps=bandwidth,
        iterations=benchmark_iterations
    )


def benchmark_matmul_kernel(
    kernel,
    m: int,
    n: int,
    k: int,
    dtype_str: str = 'float32',
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    use_cuda_events: bool = True
) -> BenchmarkResult:
    """
    Benchmark a matmul kernel with full metrics.
    
    Args:
        kernel: Kernel instance with __call__ and create_inputs methods
        m: Number of rows in A and C
        n: Number of columns in B and C
        k: Number of columns in A / rows in B
        dtype_str: Data type string ('float32' or 'float16')
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of benchmark runs
        use_cuda_events: Use CUDA events for timing
        
    Returns:
        BenchmarkResult with all metrics
    """
    import torch
    import numpy as np
    
    # Determine dtype
    dtype_bytes = 4 if dtype_str == 'float32' else 2
    dtype = torch.float32 if dtype_str == 'float32' else torch.float16
    
    # Create inputs
    A, B, C = kernel.create_inputs(m, n, k, dtype=dtype)
    
    # Benchmark
    def run():
        kernel(A, B, C)
    
    if use_cuda_events:
        try:
            latency_us, std_us, _ = compute_latency_cuda_events(
                run, (),
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                use_torch=True
            )
        except Exception:
            # Fallback to CPU timing
            latency_us, std_us, _ = compute_latency(
                run, (),
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                sync_fn=torch.cuda.synchronize
            )
    else:
        latency_us, std_us, _ = compute_latency(
            run, (),
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            sync_fn=torch.cuda.synchronize
        )
    
    # Compute metrics for matmul
    metrics = get_matmul_metrics(m, n, k, dtype_bytes)
    tflops = compute_tflops(metrics['flops'], latency_us)
    bandwidth = compute_memory_bandwidth(metrics['bytes'], latency_us)
    
    # Size string for matmul: "MxNxK"
    size_str = f"{m}x{n}x{k}"
    
    return BenchmarkResult(
        backend=kernel.name,
        size=m * n,  # Use M*N as representative size for sorting
        dtype=dtype_str,
        latency_us=latency_us,
        latency_std_us=std_us,
        tflops=tflops,
        memory_bandwidth_gbps=bandwidth,
        iterations=benchmark_iterations
    )

