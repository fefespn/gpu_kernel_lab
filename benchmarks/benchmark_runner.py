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
                            
                            result = benchmark_matmul_kernel(
                                kernel,
                                m=m, n=n, k=k,
                                dtype_str=dtype,
                                warmup_iterations=warmup,
                                benchmark_iterations=iterations
                            )
                        else:
                            result = benchmark_kernel(
                                kernel,
                                size=size,
                                dtype_str=dtype,
                                warmup_iterations=warmup,
                                benchmark_iterations=iterations
                            )
                        
                        self.results.append(result)
                        
                    except Exception as e:
                        size_str = f"{m}x{n}x{k}" if self.kernel == 'matmul' else str(size)
                        print(f"Error benchmarking {backend} | {size_str}: {e}")
        
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
        """Print a summary table of results, grouped by size for easy comparison."""
        if not self.results:
            print("No results to display")
            return
        
        # Check if we have matmul results (have m, n, k)
        is_matmul = any(r.m is not None for r in self.results)
        
        # Sort by (m, n, k, dtype, backend) for matmul, or (size, dtype, backend) for add
        if is_matmul:
            sorted_results = sorted(self.results, key=lambda r: (r.m or 0, r.n or 0, r.k or 0, r.dtype, r.backend))
        else:
            sorted_results = sorted(self.results, key=lambda r: (r.size, r.dtype, r.backend))
        
        print("\n" + "=" * 85)
        print("BENCHMARK SUMMARY")
        print("=" * 85)
        
        if is_matmul:
            print(f"{'Backend':<10} {'M':>6} {'N':>6} {'K':>6} {'Dtype':<8} {'Latency (µs)':>14} "
                  f"{'TFLOPS':>10} {'BW (GB/s)':>12}")
        else:
            print(f"{'Backend':<10} {'Size':>12} {'Dtype':<8} {'Latency (µs)':>14} "
                  f"{'TFLOPS':>10} {'BW (GB/s)':>12}")
        print("-" * 85)
        
        # Track current group for separators
        current_key = None
        for r in sorted_results:
            # Determine group key
            if is_matmul:
                group_key = (r.m, r.n, r.k, r.dtype)
            else:
                group_key = (r.size, r.dtype)
            
            # Add separator between different groups for readability
            if current_key is not None and group_key != current_key:
                print("-" * 85)
            current_key = group_key
            
            if is_matmul:
                print(f"{r.backend:<10} {r.m:>6} {r.n:>6} {r.k:>6} {r.dtype:<8} "
                      f"{r.latency_us:>14.2f} {r.tflops:>10.4f} "
                      f"{r.memory_bandwidth_gbps:>12.2f}")
            else:
                print(f"{r.backend:<10} {r.size:>12} {r.dtype:<8} "
                      f"{r.latency_us:>14.2f} {r.tflops:>10.6f} "
                      f"{r.memory_bandwidth_gbps:>12.2f}")
        
        print("=" * 85)
        
        # Print comparison analysis vs PyTorch baseline
        self._print_comparison_analysis(is_matmul)
    
    def _print_comparison_analysis(self, is_matmul: bool):
        """Print comparison analysis between all backends."""
        # Group results by test case (m,n,k,dtype for matmul, size,dtype for add)
        from collections import defaultdict
        
        test_cases = defaultdict(dict)  # {test_key: {backend: result}}
        all_backends = set()
        
        for r in self.results:
            if is_matmul:
                key = (r.m, r.n, r.k, r.dtype)
            else:
                key = (r.size, r.dtype)
            test_cases[key][r.backend] = r
            all_backends.add(r.backend)
        
        if len(all_backends) < 2:
            return
        
        # Compute pairwise stats
        pairwise_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'speedups': []}))
        
        for key, backends in test_cases.items():
            backend_list = list(backends.keys())
            for i, b1 in enumerate(backend_list):
                for b2 in backend_list[i+1:]:
                    lat1 = backends[b1].latency_us
                    lat2 = backends[b2].latency_us
                    
                    # b1 stats vs b2
                    speedup1 = lat2 / lat1  # >1 means b1 is faster
                    pairwise_stats[b1][b2]['speedups'].append(speedup1)
                    if lat1 < lat2:
                        pairwise_stats[b1][b2]['wins'] += 1
                    else:
                        pairwise_stats[b1][b2]['losses'] += 1
                    
                    # b2 stats vs b1
                    speedup2 = lat1 / lat2  # >1 means b2 is faster
                    pairwise_stats[b2][b1]['speedups'].append(speedup2)
                    if lat2 < lat1:
                        pairwise_stats[b2][b1]['wins'] += 1
                    else:
                        pairwise_stats[b2][b1]['losses'] += 1
        
        # Print pairwise comparison table
        print("\n" + "=" * 85)
        print("HEAD-TO-HEAD COMPARISON")
        print("=" * 85)
        print(f"{'Backend A':<12} {'vs':<4} {'Backend B':<12} {'Avg Speedup':>12} {'Wins':>6} {'Losses':>8} {'Win Rate':>10}")
        print("-" * 85)
        
        # Sort backends for consistent output
        sorted_backends = sorted(all_backends)
        
        for b1 in sorted_backends:
            for b2 in sorted_backends:
                if b1 >= b2:  # Skip self and duplicates (only show A vs B, not B vs A)
                    continue
                
                stats = pairwise_stats[b1][b2]
                speedups = stats['speedups']
                if not speedups:
                    continue
                    
                avg_speedup = sum(speedups) / len(speedups)
                wins = stats['wins']
                losses = stats['losses']
                total = wins + losses
                win_rate = (wins / total * 100) if total > 0 else 0
                
                speedup_str = f"{avg_speedup:.2f}x"
                if avg_speedup > 1:
                    speedup_str = f"🚀 {speedup_str}"
                else:
                    speedup_str = f"🐢 {speedup_str}"
                
                print(f"{b1:<12} {'vs':<4} {b2:<12} {speedup_str:>12} {wins:>6} {losses:>8} {win_rate:>9.1f}%")
        
        print("=" * 85)
        print("Note: Speedup > 1.0 means Backend A is faster than Backend B")
