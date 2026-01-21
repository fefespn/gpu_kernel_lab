# SPDX-License-Identifier: Apache-2.0
"""
SASS Analysis for Challenge 1: Cooperative Warps

Compares SASS output from Triton and CuTile large-tile GEMM kernels:
1. WGMMA instructions (Warp Group MMA - cooperative)
2. HMMA/UTCHMMA instructions (single-warp or TC hierarchy MMA)
3. BAR.SYNC patterns (explicit warp synchronization)
4. Register usage indicators

Key insight: If CuTile uses WGMMA and Triton uses fragmented HMMA,
we've demonstrated the cooperative warp limitation.
"""

import os
import re
import argparse
from typing import Dict, List, Tuple


# Instruction patterns for Blackwell/Hopper SASS
PATTERNS = {
    'wgmma': re.compile(r'\bWGMMA[\.\s]\w+', re.IGNORECASE),         # Warp Group MMA (cooperative)
    'hmma': re.compile(r'\b(HMMA[\.\s]|UTCHMMA\s)', re.IGNORECASE),  # Single-warp MMA / TC hierarchy
    'bar_sync': re.compile(r'\bBAR\.SYNC\b', re.IGNORECASE),         # Explicit barrier sync
    'depbar': re.compile(r'\bDEPBAR\b', re.IGNORECASE),              # Dependency barrier
    'ldgsts': re.compile(r'\b(LDG|STG|LDGSTS)\.', re.IGNORECASE),    # Global memory ops
    'shfl': re.compile(r'\bSHFL\.', re.IGNORECASE),                  # Warp shuffle (data sharing)
    'tcgen': re.compile(r'\bTCGEN\.', re.IGNORECASE),                # Tensor Core generation
}


def parse_sass(sass_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """Parse SASS file and extract relevant instructions."""
    results = {name: [] for name in PATTERNS}
    
    if not os.path.exists(sass_path):
        print(f"  Warning: {sass_path} not found")
        return results
    
    with open(sass_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    results[name].append((line_num, line))
    
    return results


def analyze_warp_cooperation(sass_results: Dict[str, List]) -> str:
    """Analyze warp cooperation patterns in SASS."""
    wgmma_count = len(sass_results['wgmma'])
    hmma_count = len(sass_results['hmma'])
    bar_count = len(sass_results['bar_sync'])
    shfl_count = len(sass_results['shfl'])
    
    analysis = []
    analysis.append(f"WGMMA (warp-group MMA): {wgmma_count}")
    analysis.append(f"HMMA/UTCHMMA (single-warp/TC): {hmma_count}")
    analysis.append(f"BAR.SYNC (explicit sync): {bar_count}")
    analysis.append(f"SHFL (warp shuffle): {shfl_count}")
    analysis.append("")
    
    if wgmma_count > 0:
        analysis.append("✓ Uses WGMMA - true warp-group cooperation!")
        analysis.append("  Warps collaborate implicitly within instruction")
    elif hmma_count > 0:
        analysis.append("⚠️  Uses HMMA/UTCHMMA - fragmented execution")
        if bar_count > 10:
            analysis.append(f"  High BAR.SYNC count ({bar_count}) suggests explicit coordination")
        if shfl_count > 0:
            analysis.append(f"  Uses SHFL ({shfl_count}x) for data sharing between warps")
    else:
        analysis.append("? No MMA instructions found")
    
    # Register pressure indicator
    total_ops = wgmma_count + hmma_count
    sync_ratio = bar_count / max(total_ops, 1)
    if sync_ratio > 0.5:
        analysis.append(f"\n⚠️  High sync-to-MMA ratio ({sync_ratio:.2f}) - potential bottleneck")
    
    return "\n".join(analysis)


def compare_backends(triton_sass: str, cutile_sass: str) -> str:
    """Compare Triton and CuTile SASS patterns."""
    
    output = []
    output.append("=" * 70)
    output.append("SASS COMPARISON: Triton vs CuTile (Challenge 1: Cooperative Warps)")
    output.append("=" * 70)
    
    # Parse both
    triton_results = parse_sass(triton_sass)
    cutile_results = parse_sass(cutile_sass)
    
    # Triton analysis
    output.append("\n" + "-" * 35)
    output.append("TRITON ANALYSIS (128x128 tiles)")
    output.append("-" * 35)
    output.append(analyze_warp_cooperation(triton_results))
    
    # CuTile analysis
    output.append("\n" + "-" * 35)
    output.append("CUTILE ANALYSIS (128x128 tiles)")
    output.append("-" * 35)
    output.append(analyze_warp_cooperation(cutile_results))
    
    # Comparison
    output.append("\n" + "=" * 70)
    output.append("COMPARISON SUMMARY")
    output.append("=" * 70)
    
    triton_wgmma = len(triton_results['wgmma'])
    cutile_wgmma = len(cutile_results['wgmma'])
    triton_hmma = len(triton_results['hmma'])
    cutile_hmma = len(cutile_results['hmma'])
    
    output.append(f"Triton: {triton_wgmma} WGMMA, {triton_hmma} HMMA/UTCHMMA")
    output.append(f"CuTile: {cutile_wgmma} WGMMA, {cutile_hmma} HMMA/UTCHMMA")
    
    if cutile_wgmma > triton_wgmma:
        output.append("\n✓ CuTile uses more WGMMA - better warp-group utilization!")
    elif triton_wgmma > cutile_wgmma:
        output.append("\n? Triton uses more WGMMA - unexpected result")
    else:
        # Both have same WGMMA count, compare sync patterns
        triton_sync = len(triton_results['bar_sync']) + len(triton_results['depbar'])
        cutile_sync = len(cutile_results['bar_sync']) + len(cutile_results['depbar'])
        
        if cutile_sync < triton_sync:
            output.append(f"\n✓ CuTile has fewer sync ops ({cutile_sync} vs {triton_sync})")
        elif triton_sync < cutile_sync:
            output.append(f"\n≈ Triton has fewer sync ops ({triton_sync} vs {cutile_sync})")
        else:
            output.append("\n≈ Similar patterns - need deeper analysis")
    
    output.append("=" * 70)
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Analyze SASS for cooperative warp patterns')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--triton-sass', type=str, help='Path to Triton SASS')
    parser.add_argument('--cutile-sass', type=str, help='Path to CuTile SASS')
    args = parser.parse_args()
    
    # Default paths for 128x128x64 tiles
    triton_sass = args.triton_sass or os.path.join(
        args.output_dir, 'triton', 'large_tile_gemm_128x128x64_sm100.sass'
    )
    cutile_sass = args.cutile_sass or os.path.join(
        args.output_dir, 'cutile', 'large_tile_gemm_128x128x64_sm100.sass'
    )
    
    print("\n" + "=" * 70)
    print("Challenge 1: Cooperative Warps - SASS Analysis")
    print("=" * 70)
    print(f"\nTriton SASS: {triton_sass}")
    print(f"CuTile SASS: {cutile_sass}")
    
    # Check files exist
    if not os.path.exists(triton_sass):
        print(f"\n❌ Triton SASS not found. Run: python triton_large_tile.py --compile-only")
        return
    if not os.path.exists(cutile_sass):
        print(f"\n❌ CuTile SASS not found. Run: python cutile_large_tile.py --compile-only")
        return
    
    # Compare
    report = compare_backends(triton_sass, cutile_sass)
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(args.output_dir, 'cooperative_warps_analysis.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
