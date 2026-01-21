# SPDX-License-Identifier: Apache-2.0
"""
SASS Analysis for Challenge 4: Blocking Synchronization

Compares SASS output from Triton and CuTile GEMM+EXP kernels to identify:
1. HMMA instructions (Tensor Core GEMM)
2. MUFU.EX2 instructions (EXP)
3. BAR.SYNC / DEPBAR / WAIT instructions (blocking sync)

Key insight: If Triton shows HMMA → WAIT → MUFU, while CuTile shows
interleaved or distributed execution, we've demonstrated the limitation.
"""

import os
import re
import argparse
from typing import Dict, List, Tuple


# Instruction patterns for Blackwell/Hopper SASS
PATTERNS = {
    'hmma': re.compile(r'\b(HMMA|UTCHMMA)\.\w+', re.IGNORECASE),  # Tensor Core GEMM (incl. Blackwell UTCHMMA)
    'exp': re.compile(r'\bMUFU\.EX2\b', re.IGNORECASE),           # exp2 (used for exp)
    'wait': re.compile(r'\b(BAR\.SYNC|DEPBAR|\.WAIT)\b', re.IGNORECASE),  # Sync ops
    'bar': re.compile(r'\bBAR\.\w+\b', re.IGNORECASE),            # Barrier ops
    'wgmma': re.compile(r'\bWGMMA\.\w+', re.IGNORECASE),          # Warp Group MMA (Hopper+)
    'mufu': re.compile(r'\bMUFU\.\w+\b', re.IGNORECASE),          # All MUFU ops
}


def parse_sass(sass_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Parse SASS file and extract relevant instructions.
    
    Returns:
        Dict mapping instruction type to list of (line_num, instruction)
    """
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


def analyze_instruction_scheduling(sass_results: Dict[str, List]) -> str:
    """
    Analyze the scheduling pattern between GEMM and EXP instructions.
    
    Looks for:
    - Pattern A: HMMA → WAIT → MUFU (blocking)
    - Pattern B: Interleaved HMMA/MUFU (potentially optimized)
    """
    hmma_lines = [ln for ln, _ in sass_results['hmma']]
    wgmma_lines = [ln for ln, _ in sass_results['wgmma']]
    exp_lines = [ln for ln, _ in sass_results['exp']]
    wait_lines = [ln for ln, _ in sass_results['wait']]
    
    gemm_lines = sorted(hmma_lines + wgmma_lines)
    
    if not gemm_lines or not exp_lines:
        return "INCOMPLETE: Missing GEMM or EXP instructions"
    
    # Find the last GEMM and first EXP
    last_gemm = max(gemm_lines)
    first_exp = min(exp_lines)
    
    # Check for waits between GEMM and EXP
    waits_between = [ln for ln in wait_lines if last_gemm < ln < first_exp]
    
    analysis = []
    analysis.append(f"GEMM instructions: {len(gemm_lines)}")
    analysis.append(f"EXP instructions: {len(exp_lines)}")
    analysis.append(f"WAIT/SYNC instructions: {len(wait_lines)}")
    analysis.append("")
    
    if waits_between:
        analysis.append("⚠️  BLOCKING PATTERN DETECTED!")
        analysis.append(f"   Found {len(waits_between)} wait(s) between last GEMM (L{last_gemm}) and first EXP (L{first_exp})")
        analysis.append("   This confirms the paper's observation: EXP is blocked by GEMM.WAIT")
    else:
        if first_exp > last_gemm:
            analysis.append("📊 Sequential pattern (no explicit wait between GEMM and EXP)")
        else:
            analysis.append("✓ Potentially overlapped: EXP starts before last GEMM")
    
    return "\n".join(analysis)


def compare_backends(triton_sass: str, cutile_sass: str) -> str:
    """Compare Triton and CuTile SASS patterns."""
    
    output = []
    output.append("=" * 70)
    output.append("SASS COMPARISON: Triton vs CuTile (Challenge 4: Blocking Sync)")
    output.append("=" * 70)
    
    # Parse both
    triton_results = parse_sass(triton_sass)
    cutile_results = parse_sass(cutile_sass)
    
    # Triton analysis
    output.append("\n" + "-" * 35)
    output.append("TRITON ANALYSIS")
    output.append("-" * 35)
    output.append(analyze_instruction_scheduling(triton_results))
    
    # CuTile analysis
    output.append("\n" + "-" * 35)
    output.append("CUTILE ANALYSIS")
    output.append("-" * 35)
    output.append(analyze_instruction_scheduling(cutile_results))
    
    # Comparison summary
    output.append("\n" + "=" * 70)
    output.append("COMPARISON SUMMARY")
    output.append("=" * 70)
    
    triton_waits = len(triton_results['wait'])
    cutile_waits = len(cutile_results['wait'])
    
    triton_gemm = len(triton_results['hmma']) + len(triton_results['wgmma'])
    cutile_gemm = len(cutile_results['hmma']) + len(cutile_results['wgmma'])
    
    output.append(f"Triton: {triton_gemm} GEMM ops, {len(triton_results['exp'])} EXP ops, {triton_waits} sync ops")
    output.append(f"CuTile: {cutile_gemm} GEMM ops, {len(cutile_results['exp'])} EXP ops, {cutile_waits} sync ops")
    
    if triton_waits > cutile_waits:
        output.append("\n✓ CuTile has fewer sync operations - potential advantage!")
    elif triton_waits == cutile_waits:
        output.append("\n≈ Both have similar sync patterns - may need deeper analysis")
    else:
        output.append("\n? Triton has fewer syncs - unexpected result")
    
    output.append("=" * 70)
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Analyze SASS for blocking sync patterns')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--triton-sass', type=str, help='Path to Triton SASS (default: auto-detect)')
    parser.add_argument('--cutile-sass', type=str, help='Path to CuTile SASS (default: auto-detect)')
    args = parser.parse_args()
    
    # Default paths
    triton_sass = args.triton_sass or os.path.join(args.output_dir, 'triton', 'gemm_exp_sm100.sass')
    cutile_sass = args.cutile_sass or os.path.join(args.output_dir, 'cutile', 'gemm_exp_sm100.sass')
    
    print("\n" + "=" * 70)
    print("Challenge 4: Blocking Synchronization - SASS Analysis")
    print("=" * 70)
    print(f"\nTriton SASS: {triton_sass}")
    print(f"CuTile SASS: {cutile_sass}")
    
    # Check files exist
    if not os.path.exists(triton_sass):
        print(f"\n❌ Triton SASS not found. Run: python triton_gemm_exp.py --compile-only")
        return
    if not os.path.exists(cutile_sass):
        print(f"\n❌ CuTile SASS not found. Run: python cutile_gemm_exp.py --compile-only")
        return
    
    # Compare
    report = compare_backends(triton_sass, cutile_sass)
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(args.output_dir, 'blocking_sync_analysis.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
