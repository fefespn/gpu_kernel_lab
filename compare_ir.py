#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tile IR comparison tool.

Compare Triton TTIR and CuTile Typed IR as tile-based programming models,
extracting and comparing control flow, tile operations, and memory patterns.

Usage:
    python compare_ir.py                          # Compare matmul kernel
    python compare_ir.py --kernel matmul          # Specific kernel
    python compare_ir.py --verbose                # Show detailed output
    python compare_ir.py --triton-ir PATH --cutile-ir PATH  # Compare specific files
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Available kernels
AVAILABLE_KERNELS = ['matmul']  # Only matmul has both IR outputs currently


def compare_kernel(
    kernel_name: str, 
    config_path: str, 
    verbose: bool,
    triton_ir_path: str = None,
    cutile_ir_path: str = None
) -> bool:
    """Compare IR for a single kernel."""
    from analysis.ir_comparator import IRComparator
    from analysis.triton_ir_parser import parse_triton_ir
    from analysis.cutile_ir_parser import parse_cutile_ir
    
    comparator = IRComparator(config_path)
    
    # Determine file paths
    if triton_ir_path is None:
        triton_ir_path = os.path.join('outputs', f'triton_{kernel_name}', f'{kernel_name}_sm100_ttir.txt')
    
    if cutile_ir_path is None:
        cutile_ir_path = os.path.join('outputs', f'cutile_{kernel_name}', f'{kernel_name}_sm100_typed_ir.txt')
    
    print(f"\n{'='*60}")
    print(f"Tile IR Comparison: {kernel_name}")
    print(f"{'='*60}")
    print(f"\nTriton IR:  {triton_ir_path}")
    print(f"CuTile IR:  {cutile_ir_path}")
    
    # Check if files exist
    if not os.path.exists(triton_ir_path):
        print(f"\n✗ Error: Triton IR file not found: {triton_ir_path}")
        return False
    
    if not os.path.exists(cutile_ir_path):
        print(f"\n✗ Error: CuTile IR file not found: {cutile_ir_path}")
        return False
    
    try:
        # Parse both IRs
        print("\nParsing IRs...")
        triton_ir = parse_triton_ir(triton_ir_path)
        cutile_ir = parse_cutile_ir(cutile_ir_path)
        
        if verbose:
            print(f"\n  Triton IR parsed:")
            print(f"    - Kernel: {triton_ir.name}")
            print(f"    - Tile shapes: {[str(s) for s in triton_ir.tile_shapes[:3]]}")
            print(f"    - Loads: {len(triton_ir.tile_loads)}, Stores: {len(triton_ir.tile_stores)}")
            print(f"    - Compute ops: {len(triton_ir.tile_computes)}")
            print(f"    - Loops: {len(triton_ir.loops)}")
            
            print(f"\n  CuTile IR parsed:")
            print(f"    - Kernel: {cutile_ir.name}")
            print(f"    - Tile shapes: {[str(s) for s in cutile_ir.tile_shapes[:3]]}")
            print(f"    - Loads: {len(cutile_ir.tile_loads)}, Stores: {len(cutile_ir.tile_stores)}")
            print(f"    - Compute ops: {len(cutile_ir.tile_computes)}")
            print(f"    - Loops: {len(cutile_ir.loops)}")
        
        # Compare
        print("\nComparing...")
        comparison = comparator.compare(triton_ir, cutile_ir)
        
        # Print comparison
        print("\n" + comparison.summary_text)
        
        # Save to file
        output_path = comparator.save_comparison(comparison)
        print(f"\n✓ Comparison saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare Triton TTIR and CuTile Typed IR for tile-based kernel analysis'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--kernel', '-k',
        default='matmul',
        choices=AVAILABLE_KERNELS,
        help='Kernel to analyze (default: matmul)'
    )
    parser.add_argument(
        '--triton-ir',
        help='Path to Triton TTIR file (overrides default)'
    )
    parser.add_argument(
        '--cutile-ir',
        help='Path to CuTile Typed IR file (overrides default)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with parsed details'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TILE IR COMPARISON TOOL")
    print("Comparing Triton TTIR vs CuTile Typed IR")
    print("=" * 60)
    
    success = compare_kernel(
        args.kernel,
        args.config,
        args.verbose,
        args.triton_ir,
        args.cutile_ir
    )
    
    print("\n" + "=" * 60)
    if success:
        print("IR comparison complete!")
    else:
        print("IR comparison failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
