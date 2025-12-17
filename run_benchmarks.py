#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark runner script that reads config.yaml and runs benchmarks.

Usage:
    python run_benchmarks.py                          # Run all benchmarks
    python run_benchmarks.py --config custom.yaml     # Custom config
    python run_benchmarks.py --backend triton cublas  # Specific backends
    python run_benchmarks.py --sizes 65536,262144     # Specific sizes

For SASS comparison, use compare_sass.py (separate phase).
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU kernel benchmarks based on configuration'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--kernel', '-k',
        default='add',
        help='Kernel name to benchmark (default: add)'
    )
    parser.add_argument(
        '--backend', '-b',
        nargs='+',
        dest='backends',
        help='Backend(s) to benchmark'
    )
    parser.add_argument(
        '--sizes', '-s',
        type=str,
        help='Comma-separated list of sizes to benchmark'
    )
    parser.add_argument(
        '--dtypes', '-d',
        type=str,
        default='float32',
        help='Comma-separated list of dtypes (default: float32)'
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
    
    # Parse sizes
    sizes = None
    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    # Parse dtypes
    dtypes = [d.strip() for d in args.dtypes.split(',')]
    
    # Import and run
    from benchmarks.benchmark_runner import BenchmarkRunner
    
    runner = BenchmarkRunner(args.config)
    
    if args.compile_only:
        print("Compile-only mode: Compiling kernels...")
        from kernels.add import get_backend
        import yaml
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        backends = args.backends or config.get('benchmarks', {}).get('backends', [])
        
        for backend_name in backends:
            print(f"\nCompiling {backend_name}...")
            try:
                kernel_class = get_backend(backend_name)
                kernel = kernel_class(config)
                result = kernel.compile()
                print(f"  Status: {result['status']}")
                if result.get('artifacts'):
                    for k, v in result['artifacts'].items():
                        print(f"  {k}: {v}")
            except Exception as e:
                print(f"  Error: {e}")
        
        return
    
    # Run benchmarks
    results = runner.run(
        backends=args.backends,
        sizes=sizes,
        dtypes=dtypes
    )
    
    if not results:
        print("No benchmark results generated!")
        sys.exit(1)
    
    # Save results
    if args.output_format in ['json', 'both']:
        runner.save_json()
    if args.output_format in ['csv', 'both']:
        runner.save_csv()
    
    # Print summary
    runner.print_summary()


if __name__ == '__main__':
    main()
