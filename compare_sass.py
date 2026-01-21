#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SASS extraction and comparison script.

This script is separate from benchmarking because:
1. SASS analysis is a different phase from performance benchmarking
2. Can run on systems that can't execute benchmarks (e.g., non-Blackwell GPUs)
3. Only requires compiled kernel artifacts, not actual execution

Usage:
    python compare_sass.py                          # Compare all backends for 'add' kernel
    python compare_sass.py --kernel add             # Specific kernel
    python compare_sass.py --kernel matmul          # Matmul kernel
    python compare_sass.py --kernel all             # All kernels (add + matmul)
    python compare_sass.py --backends triton cutile # Specific backends to compare
    python compare_sass.py --config custom.yaml     # Custom config
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Available kernels
AVAILABLE_KERNELS = ['add', 'matmul']


def compare_kernel(kernel_name: str, backends: list, config_path: str, extract_only: bool, verbose: bool):
    """Compare SASS for a single kernel."""
    from analysis.sass_extractor import SassExtractor
    from analysis.sass_comparator import SassComparator
    
    extractor = SassExtractor(config_path)
    
    print(f"\nExtracting SASS for kernel: {kernel_name}")
    print(f"Backends: {', '.join(backends)}")
    
    # Map backend names to extractor methods
    extract_methods = {
        'triton': extractor.extract_triton,
        'cutile': extractor.extract_cutile,
    }
    
    artifacts = {}
    for backend in backends:
        print(f"\n  Extracting {backend}...")
        try:
            if backend not in extract_methods:
                print(f"    Skipping: no SASS extraction for {backend}")
                continue
            
            artifact = extract_methods[backend](kernel_name)
            if artifact:
                artifacts[backend] = artifact
                print(f"    CUBIN: {artifact.cubin_path}")
                print(f"    SASS: {artifact.sass_path}")
                if verbose and artifact.sass_content:
                    lines = artifact.sass_content.split('\n')
                    print(f"    SASS lines: {len(lines)}")
        except Exception as e:
            print(f"    Error: {e}")
    
    if extract_only:
        print(f"\nExtraction complete for {kernel_name}. Artifacts: {list(artifacts.keys())}")
        return True
    
    # Compare if we have at least two backends
    if len(artifacts) < 2:
        print(f"\nNeed at least 2 backends for comparison, got {len(artifacts)}")
        if artifacts:
            print(f"Available: {list(artifacts.keys())}")
        return False
    
    comparator = SassComparator(config_path)
    
    # Compare all pairs
    backend_list = list(artifacts.keys())
    for i, backend1 in enumerate(backend_list):
        for backend2 in backend_list[i+1:]:
            sass1 = artifacts[backend1].sass_content
            sass2 = artifacts[backend2].sass_content
            
            if sass1 and sass2:
                print(f"\nComparing {backend1} vs {backend2}...")
                comparison = comparator.compare(
                    sass1, sass2,
                    backend1, backend2
                )
                # Include kernel name in output filename
                filename = f"sass_comparison_{kernel_name}_{backend1}_vs_{backend2}.txt"
                output_path = comparator.save_comparison(comparison, filename)
                print(f"  Comparison saved to: {output_path}")
            else:
                missing = []
                if not sass1:
                    missing.append(backend1)
                if not sass2:
                    missing.append(backend2)
                print(f"\n  Skipping {backend1} vs {backend2}: missing SASS for {missing}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract and compare SASS from compiled GPU kernels'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--kernel', '-k',
        default='add',
        choices=AVAILABLE_KERNELS + ['all'],
        help='Kernel to analyze: add, matmul, or all (default: add)'
    )
    parser.add_argument(
        '--backends', '-b',
        nargs='+',
        default=['triton', 'cutile'],
        help='Backends to extract and compare (default: triton cutile)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for comparison results (default: from config)'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Only extract SASS, skip comparison'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SASS EXTRACTION AND COMPARISON")
    print("=" * 50)
    
    # Determine which kernels to process
    if args.kernel == 'all':
        kernels = AVAILABLE_KERNELS
    else:
        kernels = [args.kernel]
    
    # Process each kernel
    success_count = 0
    for kernel_name in kernels:
        print("\n" + "-" * 50)
        print(f"Processing kernel: {kernel_name}")
        print("-" * 50)
        
        if compare_kernel(kernel_name, args.backends, args.config, args.extract_only, args.verbose):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"SASS comparison complete! ({success_count}/{len(kernels)} kernels)")


if __name__ == '__main__':
    main()

