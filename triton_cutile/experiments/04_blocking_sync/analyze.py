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

# PTX instruction patterns (including Blackwell sm100 specific)
PTX_PATTERNS = {
    # Traditional MMA
    'mma': re.compile(r'\b(mma|wmma|wgmma)\.', re.IGNORECASE),      # Matrix multiply (pre-Blackwell)
    
    # Blackwell sm100 Tensor Core Gen5 (tcgen05) operations
    'tcgen05_mma': re.compile(r'\btcgen05\.mma', re.IGNORECASE),    # Blackwell Tensor Core MMA
    'tcgen05_ld': re.compile(r'\btcgen05\.ld', re.IGNORECASE),      # Tensor Core loads
    'tcgen05_st': re.compile(r'\btcgen05\.st', re.IGNORECASE),      # Tensor Core stores
    'tcgen05_commit': re.compile(r'\btcgen05\.commit', re.IGNORECASE),  # Tensor Core commit
    'tcgen05_wait': re.compile(r'\btcgen05\.wait', re.IGNORECASE),  # Tensor Core wait/sync
    'tcgen05_alloc': re.compile(r'\btcgen05\.(alloc|dealloc)', re.IGNORECASE),  # TC allocation
    
    # Math operations
    'exp': re.compile(r'\bex2\.', re.IGNORECASE),                   # exp2 (exp uses ex2)
    'mufu': re.compile(r'\b(sin|cos|lg2|ex2|rcp|rsqrt)\.', re.IGNORECASE),  # Special function unit
    
    # Barriers and synchronization (Blackwell specific)
    'bar': re.compile(r'\bbar\.(sync|arrive|wait)', re.IGNORECASE), # Basic barriers
    'mbarrier': re.compile(r'\bmbarrier\.', re.IGNORECASE),         # Memory barriers (sm80+)
    'barrier_cluster': re.compile(r'\bbarrier\.cluster', re.IGNORECASE),  # Cluster barriers (sm90+)
    'fence': re.compile(r'\bfence\.', re.IGNORECASE),               # Memory fences
    
    # Async memory operations (TMA - Tensor Memory Accelerator)
    'cp_async': re.compile(r'\bcp\.async', re.IGNORECASE),          # Async copy (basic)
    'cp_async_bulk': re.compile(r'\bcp\.async\.bulk', re.IGNORECASE),  # Bulk async copy (TMA)
    
    # Memory operations
    'ld_global': re.compile(r'\bld\.global', re.IGNORECASE),        # Global loads
    'st_global': re.compile(r'\bst\.global', re.IGNORECASE),        # Global stores
    'ld_shared': re.compile(r'\bld\.shared', re.IGNORECASE),        # Shared loads
    'st_shared': re.compile(r'\bst\.shared', re.IGNORECASE),        # Shared stores
    
    # Register management (Blackwell dynamic)
    'setmaxnreg': re.compile(r'\bsetmaxnreg\.', re.IGNORECASE),     # Dynamic register allocation
    
    # Arithmetic
    'cvt': re.compile(r'\bcvt\.', re.IGNORECASE),                   # Type conversions
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


def parse_ptx(ptx_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Parse PTX file and extract relevant instructions.
    
    Returns:
        Dict mapping instruction type to list of (line_num, instruction)
    """
    results = {name: [] for name in PTX_PATTERNS}
    
    if not os.path.exists(ptx_path):
        print(f"  Warning: {ptx_path} not found")
        return results
    
    with open(ptx_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line_stripped = line.strip()
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('//'):
                continue
            for name, pattern in PTX_PATTERNS.items():
                if pattern.search(line_stripped):
                    results[name].append((line_num, line_stripped))
    
    return results


def analyze_ptx_scheduling(ptx_results: Dict[str, List]) -> str:
    """
    Analyze PTX instruction patterns for GEMM+EXP fusion.
    Includes Blackwell sm100 specific instructions (tcgen05).
    """
    # Traditional MMA + Blackwell tcgen05 MMA
    mma_lines = [ln for ln, _ in ptx_results.get('mma', [])]
    tcgen05_mma_lines = [ln for ln, _ in ptx_results.get('tcgen05_mma', [])]
    all_mma_lines = sorted(mma_lines + tcgen05_mma_lines)
    
    # EXP operations
    exp_lines = [ln for ln, _ in ptx_results.get('exp', [])]
    mufu_lines = [ln for ln, _ in ptx_results.get('mufu', [])]
    
    # Synchronization (including Blackwell specific)
    bar_lines = [ln for ln, _ in ptx_results.get('bar', [])]
    mbarrier_lines = [ln for ln, _ in ptx_results.get('mbarrier', [])]
    cluster_bar_lines = [ln for ln, _ in ptx_results.get('barrier_cluster', [])]
    fence_lines = [ln for ln, _ in ptx_results.get('fence', [])]
    all_sync_lines = sorted(bar_lines + mbarrier_lines + cluster_bar_lines + fence_lines)
    
    # Blackwell tcgen05 operations
    tcgen05_ld_lines = [ln for ln, _ in ptx_results.get('tcgen05_ld', [])]
    tcgen05_st_lines = [ln for ln, _ in ptx_results.get('tcgen05_st', [])]
    tcgen05_wait_lines = [ln for ln, _ in ptx_results.get('tcgen05_wait', [])]
    tcgen05_commit_lines = [ln for ln, _ in ptx_results.get('tcgen05_commit', [])]
    
    # Async operations
    cp_async_lines = [ln for ln, _ in ptx_results.get('cp_async', [])]
    cp_async_bulk_lines = [ln for ln, _ in ptx_results.get('cp_async_bulk', [])]
    
    analysis = []
    analysis.append("=== Standard PTX Operations ===")
    analysis.append(f"MMA instructions (mma/wmma/wgmma): {len(mma_lines)}")
    analysis.append(f"EXP (ex2) instructions: {len(exp_lines)}")
    analysis.append(f"Basic barriers (bar.sync): {len(bar_lines)}")
    analysis.append(f"Fence instructions: {len(fence_lines)}")
    analysis.append(f"Async copy (cp.async): {len(cp_async_lines)}")
    analysis.append("")
    
    analysis.append("=== Blackwell sm100 Operations ===")
    analysis.append(f"tcgen05.mma (Tensor Core Gen5): {len(tcgen05_mma_lines)}")
    analysis.append(f"tcgen05.ld (TC loads): {len(tcgen05_ld_lines)}")
    analysis.append(f"tcgen05.st (TC stores): {len(tcgen05_st_lines)}")
    analysis.append(f"tcgen05.wait: {len(tcgen05_wait_lines)}")
    analysis.append(f"tcgen05.commit: {len(tcgen05_commit_lines)}")
    analysis.append(f"mbarrier (memory barriers): {len(mbarrier_lines)}")
    analysis.append(f"barrier.cluster: {len(cluster_bar_lines)}")
    analysis.append(f"cp.async.bulk (TMA): {len(cp_async_bulk_lines)}")
    analysis.append(f"setmaxnreg (dynamic regs): {len(ptx_results.get('setmaxnreg', []))}")
    analysis.append("")
    
    # Scheduling analysis
    if all_mma_lines and (exp_lines or mufu_lines):
        last_mma = max(all_mma_lines)
        exp_or_mufu = exp_lines if exp_lines else mufu_lines
        first_exp = min(exp_or_mufu) if exp_or_mufu else 0
        
        if first_exp > 0:
            # Check for sync ops between MMA and EXP
            syncs_between = [ln for ln in all_sync_lines if last_mma < ln < first_exp]
            
            if syncs_between:
                analysis.append(f"⚠️  Found {len(syncs_between)} sync ops between last MMA (L{last_mma}) and first EXP (L{first_exp})")
            else:
                analysis.append(f"Last MMA at L{last_mma}, first EXP at L{first_exp}")
                if first_exp > last_mma:
                    analysis.append("Sequential: EXP after all MMA complete")
    elif all_mma_lines:
        analysis.append(f"Total MMA ops: {len(all_mma_lines)} (last at L{max(all_mma_lines)})")
        if not exp_lines:
            analysis.append("Note: No ex2 (EXP) instructions found - may use different implementation")
    
    return "\n".join(analysis)


def compare_ptx(triton_ptx: str, cutile_ptx: str) -> str:
    """Compare Triton and CuTile PTX patterns."""
    
    output = []
    output.append("=" * 70)
    output.append("PTX COMPARISON: Triton vs CuTile (Challenge 4: Blocking Sync)")
    output.append("=" * 70)
    
    # Check PTX version
    for name, path in [("Triton", triton_ptx), ("CuTile", cutile_ptx)]:
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    if '.version' in line:
                        output.append(f"{name} PTX: {line.strip()}")
                        break
    
    output.append("")
    
    # Parse both
    triton_results = parse_ptx(triton_ptx)
    cutile_results = parse_ptx(cutile_ptx)
    
    # Triton analysis
    output.append("-" * 35)
    output.append("TRITON PTX ANALYSIS")
    output.append("-" * 35)
    if os.path.exists(triton_ptx):
        output.append(analyze_ptx_scheduling(triton_results))
    else:
        output.append("PTX file not found")
    
    # CuTile analysis
    output.append("\n" + "-" * 35)
    output.append("CUTILE PTX ANALYSIS")
    output.append("-" * 35)
    if os.path.exists(cutile_ptx):
        output.append(analyze_ptx_scheduling(cutile_results))
    else:
        output.append("PTX file not found")
    
    # Comparison summary
    output.append("\n" + "=" * 70)
    output.append("PTX COMPARISON SUMMARY")
    output.append("=" * 70)
    
    # Standard metrics
    output.append("\n--- Standard Operations ---")
    std_metrics = ['mma', 'exp', 'bar', 'fence', 'cp_async']
    output.append(f"{'Metric':<25} {'Triton':>10} {'CuTile':>10} {'Diff':>10}")
    output.append("-" * 57)
    
    for metric in std_metrics:
        t_count = len(triton_results.get(metric, []))
        c_count = len(cutile_results.get(metric, []))
        diff = c_count - t_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        output.append(f"{metric:<25} {t_count:>10} {c_count:>10} {diff_str:>10}")
    
    # Blackwell sm100 metrics
    output.append("\n--- Blackwell sm100 Operations ---")
    bw_metrics = [
        ('tcgen05_mma', 'tcgen05.mma (TC Gen5)'),
        ('tcgen05_ld', 'tcgen05.ld'),
        ('tcgen05_st', 'tcgen05.st'),
        ('tcgen05_wait', 'tcgen05.wait'),
        ('tcgen05_commit', 'tcgen05.commit'),
        ('mbarrier', 'mbarrier'),
        ('barrier_cluster', 'barrier.cluster'),
        ('cp_async_bulk', 'cp.async.bulk (TMA)'),
        ('setmaxnreg', 'setmaxnreg'),
    ]
    output.append(f"{'Metric':<25} {'Triton':>10} {'CuTile':>10} {'Diff':>10}")
    output.append("-" * 57)
    
    for key, label in bw_metrics:
        t_count = len(triton_results.get(key, []))
        c_count = len(cutile_results.get(key, []))
        diff = c_count - t_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        output.append(f"{label:<25} {t_count:>10} {c_count:>10} {diff_str:>10}")
    
    output.append("")
    
    # Key insights
    output.append("--- Key Insights ---")
    
    # Check for Blackwell-specific advantages
    t_tcgen05 = len(triton_results.get('tcgen05_mma', []))
    c_tcgen05 = len(cutile_results.get('tcgen05_mma', []))
    if c_tcgen05 > 0 and t_tcgen05 == 0:
        output.append("✓ CuTile uses Blackwell tcgen05.mma, Triton uses older MMA")
    elif c_tcgen05 > t_tcgen05:
        output.append(f"✓ CuTile has more tcgen05.mma ops ({c_tcgen05} vs {t_tcgen05})")
    
    # Async copy comparison
    t_bulk = len(triton_results.get('cp_async_bulk', []))
    c_bulk = len(cutile_results.get('cp_async_bulk', []))
    if c_bulk > t_bulk:
        output.append(f"✓ CuTile uses more bulk TMA ({c_bulk} vs {t_bulk})")
    elif t_bulk > c_bulk:
        output.append(f"✓ Triton uses more bulk TMA ({t_bulk} vs {c_bulk})")
    
    # Barrier comparison (all types)
    t_all_bars = (len(triton_results.get('bar', [])) + 
                  len(triton_results.get('mbarrier', [])) +
                  len(triton_results.get('barrier_cluster', [])) +
                  len(triton_results.get('fence', [])))
    c_all_bars = (len(cutile_results.get('bar', [])) + 
                  len(cutile_results.get('mbarrier', [])) +
                  len(cutile_results.get('barrier_cluster', [])) +
                  len(cutile_results.get('fence', [])))
    
    if t_all_bars > c_all_bars:
        output.append(f"✓ CuTile has fewer total sync ops ({c_all_bars} vs {t_all_bars})")
    elif c_all_bars > t_all_bars:
        output.append(f"? CuTile has more sync ops ({c_all_bars} vs {t_all_bars})")
    
    # Dynamic register management
    t_regs = len(triton_results.get('setmaxnreg', []))
    c_regs = len(cutile_results.get('setmaxnreg', []))
    if c_regs > 0:
        output.append(f"✓ CuTile uses dynamic register allocation ({c_regs} setmaxnreg)")
    
    output.append("=" * 70)
    
    return "\n".join(output)


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
    parser = argparse.ArgumentParser(description='Analyze SASS and PTX for blocking sync patterns')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--triton-sass', type=str, help='Path to Triton SASS (default: auto-detect)')
    parser.add_argument('--cutile-sass', type=str, help='Path to CuTile SASS (default: auto-detect)')
    parser.add_argument('--triton-ptx', type=str, help='Path to Triton PTX (default: auto-detect)')
    parser.add_argument('--cutile-ptx', type=str, help='Path to CuTile PTX (default: auto-detect)')
    parser.add_argument('--sass-only', action='store_true', help='Only analyze SASS')
    parser.add_argument('--ptx-only', action='store_true', help='Only analyze PTX')
    args = parser.parse_args()
    
    # Default paths
    triton_sass = args.triton_sass or os.path.join(args.output_dir, 'triton', 'gemm_exp_sm100.sass')
    cutile_sass = args.cutile_sass or os.path.join(args.output_dir, 'cutile', 'gemm_exp_sm100.sass')
    triton_ptx = args.triton_ptx or os.path.join(args.output_dir, 'triton', 'gemm_exp_sm100.ptx')
    cutile_ptx = args.cutile_ptx or os.path.join(args.output_dir, 'cutile', 'gemm_exp_sm100.ptx')
    
    print("\n" + "=" * 70)
    print("Challenge 4: Blocking Synchronization - SASS & PTX Analysis")
    print("=" * 70)
    
    full_report = []
    
    # SASS Analysis
    if not args.ptx_only:
        print(f"\nTriton SASS: {triton_sass}")
        print(f"CuTile SASS: {cutile_sass}")
        
        sass_missing = False
        if not os.path.exists(triton_sass):
            print(f"\n⚠️  Triton SASS not found. Run: python triton_gemm_exp.py --compile-only")
            sass_missing = True
        if not os.path.exists(cutile_sass):
            print(f"\n⚠️  CuTile SASS not found. Run: python cutile_gemm_exp.py --compile-only")
            sass_missing = True
        
        if not sass_missing:
            sass_report = compare_backends(triton_sass, cutile_sass)
            print("\n" + sass_report)
            full_report.append(sass_report)
    
    # PTX Analysis
    if not args.sass_only:
        print(f"\nTriton PTX: {triton_ptx}")
        print(f"CuTile PTX: {cutile_ptx}")
        
        ptx_missing = False
        if not os.path.exists(triton_ptx):
            print(f"\n⚠️  Triton PTX not found. Run: python triton_gemm_exp.py --compile-only")
            ptx_missing = True
        if not os.path.exists(cutile_ptx):
            print(f"\n⚠️  CuTile PTX not found. Run: python cutile_gemm_exp.py --compile-only")
            ptx_missing = True
        
        if not ptx_missing:
            ptx_report = compare_ptx(triton_ptx, cutile_ptx)
            print("\n" + ptx_report)
            full_report.append(ptx_report)
    
    # Save combined report
    if full_report:
        report_path = os.path.join(args.output_dir, 'blocking_sync_analysis.txt')
        with open(report_path, 'w') as f:
            f.write("\n\n".join(full_report))
        print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
