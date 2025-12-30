# SPDX-License-Identifier: Apache-2.0
"""
Benchmark runner that loads config and runs benchmarks for selected backends.
"""

import os
import json
import csv
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime

from .metrics import benchmark_kernel, benchmark_matmul_kernel, BenchmarkResult


class BenchmarkRunner:
    """Run benchmarks based on configuration."""
    
    def __init__(self, config_path: str = 'config.yaml', kernel: str = 'add'):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to configuration file
            kernel: Kernel name ('add' or 'matmul')
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.kernel = kernel
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        self.output_dir = self.config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
    
    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmark config based on kernel type."""
        if self.kernel == 'matmul':
            return self.config.get('benchmarks_matmul', {})
        return self.config.get('benchmarks', {})
    
    def get_enabled_backends(self) -> List[str]:
        """Get list of enabled backends for benchmarking."""
        benchmark_config = self.get_benchmark_config()
        if not benchmark_config.get('enabled', True):
            return []
        
        return benchmark_config.get('backends', [])
    
    def get_sizes(self) -> List:
        """Get benchmark sizes from config."""
        benchmark_config = self.get_benchmark_config()
        
        if self.kernel == 'matmul':
            # Generate cartesian product of M × N × K
            M_values = benchmark_config.get('M', [1024])
            N_values = benchmark_config.get('N', [1024])
            K_values = benchmark_config.get('K', [1024])
            
            from itertools import product
            return list(product(M_values, N_values, K_values))
        
        return benchmark_config.get('sizes', [65536])
    
    def get_dtypes(self) -> List[str]:
        """Get data types to benchmark."""
        if self.kernel == 'matmul':
            return self.get_benchmark_config().get('dtypes', ['float32'])
        return self.config.get('tests', {}).get('dtypes', ['float32'])
    
    def _create_kernel(self, backend: str):
        """Create a kernel instance for the given backend."""
        if self.kernel == 'matmul':
            from kernels.matmul import get_backend
        else:
            from kernels.add import get_backend
        kernel_class = get_backend(backend)
        return kernel_class(self.config)
    
    def run(
        self,
        backends: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
        dtypes: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for specified backends and sizes.
        
        Args:
            backends: List of backends to benchmark (default: from config)
            sizes: List of sizes to benchmark (default: from config)
            dtypes: List of dtypes to benchmark (default: from config)
            
        Returns:
            List of BenchmarkResult objects
        """
        backends = backends or self.get_enabled_backends()
        sizes = sizes or self.get_sizes()
        dtypes = dtypes or self.get_dtypes()
        
        benchmark_config = self.get_benchmark_config()
        warmup = benchmark_config.get('warmup_iterations', 10)
        iterations = benchmark_config.get('benchmark_iterations', 100)
        
        print(f"Running {self.kernel} benchmarks in {self.hardware_mode} mode")
        print(f"Backends: {backends}")
        print(f"Sizes: {sizes}")
        print(f"Dtypes: {dtypes}")
        print(f"Warmup iterations: {warmup}")
        print(f"Benchmark iterations: {iterations}")
        print("-" * 60)
        
        for backend in backends:
            # Skip execution-based benchmarks in compile_only mode
            if self.hardware_mode == 'compile_only' and backend in ['triton', 'cutile', 'pytorch']:
                print(f"Skipping {backend} benchmark (compile_only mode)")
                continue
            
            try:
                kernel = self._create_kernel(backend)
            except Exception as e:
                print(f"Error creating {backend} kernel: {e}")
                continue
            
            for size in sizes:
                for dtype in dtypes:
                    try:
                        if self.kernel == 'matmul':
                            # size is [M, N, K]
                            m, n, k = size if isinstance(size, (list, tuple)) else (size, size, size)
                            print(f"Benchmarking {backend} | size={m}x{n}x{k} | dtype={dtype}...", end=" ")
                            
                            result = benchmark_matmul_kernel(
                                kernel,
                                m=m, n=n, k=k,
                                dtype_str=dtype,
                                warmup_iterations=warmup,
                                benchmark_iterations=iterations
                            )
                        else:
                            print(f"Benchmarking {backend} | size={size} | dtype={dtype}...", end=" ")
                            
                            result = benchmark_kernel(
                                kernel,
                                size=size,
                                dtype_str=dtype,
                                warmup_iterations=warmup,
                                benchmark_iterations=iterations
                            )
                        
                        self.results.append(result)
                        
                        print(f"latency={result.latency_us:.2f}µs, "
                              f"TFLOPS={result.tflops:.6f}, "
                              f"BW={result.memory_bandwidth_gbps:.2f} GB/s")
                        
                    except Exception as e:
                        print(f"Error: {e}")
        
        return self.results
    
    def save_json(self, filename: Optional[str] = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'hardware_mode': self.hardware_mode,
            'config': self.config.get('benchmarks', {}),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def save_csv(self, filename: Optional[str] = None) -> str:
        """
        Save results to CSV file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not self.results:
            print("No results to save")
            return filepath
        
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary table of results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Backend':<10} {'Size':>10} {'Dtype':<8} {'Latency (µs)':>14} "
              f"{'TFLOPS':>10} {'BW (GB/s)':>12}")
        print("-" * 80)
        
        for r in self.results:
            print(f"{r.backend:<10} {r.size:>10} {r.dtype:<8} "
                  f"{r.latency_us:>14.2f} {r.tflops:>10.6f} "
                  f"{r.memory_bandwidth_gbps:>12.2f}")
        
        print("=" * 80)
