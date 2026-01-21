#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark runner script that reads config.yaml and runs benchmarks for all enabled kernels.

Usage:
    python run_benchmarks.py                          # Run all enabled benchmarks
    python run_benchmarks.py --config custom.yaml     # Custom config
    python run_benchmarks.py --backend triton pytorch # Specific backends
    python run_benchmarks.py --compile-only           # Only compile kernels

For SASS comparison, use compare_sass.py (separate phase).
"""

import argparse
import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_enabled_benchmark_configs(config: dict) -> list:
    """
    Get all enabled benchmark configurations from config.
    
    Returns:
        List of (kernel_name, benchmark_config) tuples for enabled kernels
    """
    enabled = []
    
    # Check benchmarks (add kernel)
    benchmarks_config = config.get('benchmarks', {})
    if benchmarks_config.get('enabled', True):
        enabled.append(('add', benchmarks_config))
    
    # Check benchmarks_matmul
    benchmarks_matmul_config = config.get('benchmarks_matmul', {})
    if benchmarks_matmul_config.get('enabled', True):
        enabled.append(('matmul', benchmarks_matmul_config))
    
    return enabled


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU kernel benchmarks for all enabled kernels in config'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--backend', '-b',
        nargs='+',
        dest='backends',
        help='Backend(s) to benchmark (overrides config)'
    )
    parser.add_argument(
        '--dtypes', '-d',
        type=str,
        default=None,
        help='Comma-separated list of dtypes (default: from config)'
    )
    parser.add_argument(
        '--output-format', '-o',
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format for results'
    )
    parser.add_argument(
        '--compile-only',
        action='store_true',
        help='Only compile kernels, skip benchmarks'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Parse dtypes
    dtypes = None
    if args.dtypes:
        dtypes = [d.strip() for d in args.dtypes.split(',')]
    
    # Get enabled benchmark configs
    enabled_configs = get_enabled_benchmark_configs(config)
    
    if not enabled_configs:
        print("No kernels enabled for benchmarking in config")
        return
    
    print(f"Enabled kernels: {[k for k, _ in enabled_configs]}")
    
    # Import benchmark runner
    from benchmarks.benchmark_runner import BenchmarkRunner
    
    all_results = []
    
    # Run benchmarks for each enabled kernel
    for kernel, benchmark_config in enabled_configs:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {kernel.upper()}")
        print(f"{'='*60}")
        
        if args.compile_only:
            print(f"Compile-only mode: Compiling {kernel} kernels...")
            if kernel == 'matmul':
                from kernels.matmul import get_backend
            else:
                from kernels.add import get_backend
            
            backends = args.backends or benchmark_config.get('backends', [])
            
            for backend_name in backends:
                print(f"\nCompiling {backend_name}...")
                try:
                    kernel_class = get_backend(backend_name)
                    kernel_instance = kernel_class(config)
                    result = kernel_instance.compile()
                    print(f"  Status: {result['status']}")
                    if result.get('artifacts'):
                        for k, v in result['artifacts'].items():
                            print(f"  {k}: {v}")
                except Exception as e:
                    print(f"  Error: {e}")
            continue
        
        # Run benchmarks
        runner = BenchmarkRunner(args.config, kernel=kernel)
        
        results = runner.run(
            backends=args.backends,
            dtypes=dtypes
        )
        
        if results:
            all_results.extend(results)
            
            # Save results
            if args.output_format in ['json', 'both']:
                runner.save_json(filename=f"benchmark_{kernel}_results.json")
            if args.output_format in ['csv', 'both']:
                runner.save_csv(filename=f"benchmark_{kernel}_results.csv")
            
            # Print summary
            runner.print_summary()
    
    # Print final summary
    if all_results and not args.compile_only:
        print(f"\n{'='*60}")
        print(f"TOTAL: {len(all_results)} benchmark results across {len(enabled_configs)} kernels")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
