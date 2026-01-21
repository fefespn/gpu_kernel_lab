# SPDX-License-Identifier: Apache-2.0
"""
Semantic comparison between Triton TTIR and CuTile Typed IR.

Compares tile-based programming patterns including control flow,
tile operations, memory access patterns, and data types.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .ir_common import KernelIRSummary, TileShape, TileOperation, LoopInfo
from .triton_ir_parser import parse_triton_ir
from .cutile_ir_parser import parse_cutile_ir


@dataclass
class ComparisonItem:
    """Single comparison item between two IRs."""
    category: str           # e.g., "Tile Shape", "Control Flow"
    aspect: str             # e.g., "M×K tiles", "K-loop iterations"
    triton_value: str
    cutile_value: str
    match_status: str       # "✓" (match), "≈" (equivalent), "~" (different), "✗" (mismatch)
    notes: str = ""


@dataclass
class IRComparison:
    """Complete comparison between Triton and CuTile IR."""
    kernel_name: str
    triton_summary: KernelIRSummary
    cutile_summary: KernelIRSummary
    comparisons: List[ComparisonItem] = field(default_factory=list)
    summary_text: str = ""
    
    def get_match_statistics(self) -> Dict[str, int]:
        """Get statistics on matches."""
        stats = {"match": 0, "equivalent": 0, "different": 0, "mismatch": 0}
        for item in self.comparisons:
            if item.match_status == "✓":
                stats["match"] += 1
            elif item.match_status == "≈":
                stats["equivalent"] += 1
            elif item.match_status == "~":
                stats["different"] += 1
            else:
                stats["mismatch"] += 1
        return stats


class IRComparator:
    """Compare Triton TTIR and CuTile Typed IR semantically."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize comparator with config."""
        self.config_path = config_path
        self.output_dir = 'outputs'
        self._load_config()
    
    def _load_config(self):
        """Load configuration if available."""
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.output_dir = config.get('output_dir', 'outputs')
        except Exception:
            pass
    
    def compare(
        self, 
        triton_ir: KernelIRSummary, 
        cutile_ir: KernelIRSummary
    ) -> IRComparison:
        """
        Compare two IR summaries.
        
        Args:
            triton_ir: Parsed Triton TTIR summary
            cutile_ir: Parsed CuTile Typed IR summary
            
        Returns:
            IRComparison with detailed comparison
        """
        comparison = IRComparison(
            kernel_name=triton_ir.name or cutile_ir.name,
            triton_summary=triton_ir,
            cutile_summary=cutile_ir
        )
        
        # Compare tile shapes
        self._compare_tile_shapes(triton_ir, cutile_ir, comparison)
        
        # Compare control flow (loops)
        self._compare_control_flow(triton_ir, cutile_ir, comparison)
        
        # Compare parallelization (computed values)
        self._compare_parallelization(triton_ir, cutile_ir, comparison)
        
        # Compare tile operations
        self._compare_tile_operations(triton_ir, cutile_ir, comparison)
        
        # Compare bounds handling
        self._compare_bounds_handling(triton_ir, cutile_ir, comparison)
        
        # Compare data types and precision
        self._compare_data_types(triton_ir, cutile_ir, comparison)
        
        # Generate summary
        comparison.summary_text = self._generate_summary(comparison)
        
        return comparison
    
    def _compare_tile_shapes(
        self, 
        triton: KernelIRSummary, 
        cutile: KernelIRSummary, 
        comparison: IRComparison
    ):
        """Compare tile dimensions used, categorized by purpose."""
        # Get shapes by category
        triton_by_cat = triton.get_shapes_by_category()
        cutile_by_cat = cutile.get_shapes_by_category()
        
        # Compare DATA tiles (the main tiles for computation)
        triton_data = triton_by_cat.get("data", [])
        cutile_data = cutile_by_cat.get("data", [])
        
        triton_data_dims = set((s.rows, s.cols) for s in triton_data)
        cutile_data_dims = set((s.rows, s.cols) for s in cutile_data)
        
        triton_str = ", ".join(f"{r}×{c}" for r, c in sorted(triton_data_dims))
        cutile_str = ", ".join(f"{r}×{c}" for r, c in sorted(cutile_data_dims))
        
        match_status = "✓" if triton_data_dims == cutile_data_dims else "~"
        
        comparison.comparisons.append(ComparisonItem(
            category="Tile Shapes",
            aspect="Data tiles (compute)",
            triton_value=triton_str or "none",
            cutile_value=cutile_str or "none",
            match_status=match_status,
            notes="Both use same tile blocking for computation" if match_status == "✓" else ""
        ))
        
        # Show INDEX tiles (only Triton has these typically)
        triton_index = triton_by_cat.get("index", [])
        cutile_index = cutile_by_cat.get("index", [])
        
        triton_idx_str = f"{len(triton_index)} shapes" if triton_index else "implicit"
        cutile_idx_str = f"{len(cutile_index)} shapes" if cutile_index else "implicit"
        
        comparison.comparisons.append(ComparisonItem(
            category="Tile Shapes",
            aspect="Index tiles (addressing)",
            triton_value=triton_idx_str,
            cutile_value=cutile_idx_str,
            match_status="≈",
            notes="Triton uses explicit index tensors; CuTile abstracts indexing"
        ))
        
        # Show MASK tiles (bounds checking)
        triton_mask = triton_by_cat.get("mask", [])
        cutile_mask = cutile_by_cat.get("mask", [])
        
        triton_mask_str = f"{len(triton_mask)} shapes" if triton_mask else "none"
        cutile_mask_str = f"{len(cutile_mask)} shapes" if cutile_mask else "none"
        
        comparison.comparisons.append(ComparisonItem(
            category="Tile Shapes",
            aspect="Mask tiles (bounds)",
            triton_value=triton_mask_str,
            cutile_value=cutile_mask_str,
            match_status="≈",
            notes="Triton uses explicit masks; CuTile uses padding_mode"
        ))
        
        # Total shape count for reference
        comparison.comparisons.append(ComparisonItem(
            category="Tile Shapes",
            aspect="Total unique shapes",
            triton_value=str(len(triton.tile_shapes)),
            cutile_value=str(len(cutile.tile_shapes)),
            match_status="~",
            notes="Triton IR is lower-level with more explicit intermediates"
        ))
    
    def _compare_control_flow(
        self,
        triton: KernelIRSummary,
        cutile: KernelIRSummary,
        comparison: IRComparison
    ):
        """Compare loop structures."""
        triton_loops = triton.loops
        cutile_loops = cutile.loops
        
        # Number of loops
        comparison.comparisons.append(ComparisonItem(
            category="Control Flow",
            aspect="Number of loops",
            triton_value=str(len(triton_loops)),
            cutile_value=str(len(cutile_loops)),
            match_status="✓" if len(triton_loops) == len(cutile_loops) else "~"
        ))
        
        # K-loop iterations (primary loop)
        if triton_loops:
            triton_k = triton_loops[0]
            triton_iter = triton_k.iterations_expr or f"{triton_k.start} to {triton_k.end}"
        else:
            triton_iter = "no loop"
        
        if cutile_loops:
            cutile_k = cutile_loops[0]
            cutile_iter = cutile_k.iterations_expr or f"{cutile_k.start} to {cutile_k.end}"
        else:
            cutile_iter = "no loop"
        
        # Check if iterations are equivalent (cdiv vs num_tiles)
        equiv = ("ceil" in triton_iter.lower() or "cdiv" in triton_iter.lower()) and "num_tiles" in cutile_iter.lower()
        
        comparison.comparisons.append(ComparisonItem(
            category="Control Flow",
            aspect="K-loop iterations",
            triton_value=triton_iter,
            cutile_value=cutile_iter,
            match_status="≈" if equiv else ("✓" if triton_iter == cutile_iter else "~"),
            notes="Equivalent: ceil(K/tile) = num_tiles" if equiv else ""
        ))
        
        # Operations per iteration
        triton_ops = len(triton_loops[0].body_ops) if triton_loops else 0
        cutile_ops = len(cutile_loops[0].body_ops) if cutile_loops else 0
        
        comparison.comparisons.append(ComparisonItem(
            category="Control Flow",
            aspect="Ops per K-iteration",
            triton_value=str(triton_ops),
            cutile_value=str(cutile_ops),
            match_status="✓" if triton_ops == cutile_ops else "~"
        ))
    
    def _compare_parallelization(
        self,
        triton: KernelIRSummary,
        cutile: KernelIRSummary,
        comparison: IRComparison
    ):
        """Compare parallelization and compute actual iteration counts."""
        import math
        import yaml
        
        # Try to load matrix dimensions from config
        m, n, k = 8192, 8192, 8192  # defaults
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                compile_dims = config.get('kernels', {}).get('matmul', {}).get('compile_dims', [8192, 8192, 8192])
                m, n, k = compile_dims[0], compile_dims[1], compile_dims[2]
        except Exception:
            pass
        
        # Get tile sizes from parsed IR
        triton_by_cat = triton.get_shapes_by_category()
        cutile_by_cat = cutile.get_shapes_by_category()
        
        triton_data = triton_by_cat.get("data", [])
        cutile_data = cutile_by_cat.get("data", [])
        
        # Get tile dimensions (assume first data tile is the main compute tile)
        triton_tile_m = triton_data[0].rows if triton_data else 32
        triton_tile_n = triton_data[0].cols if triton_data else 32
        triton_tile_k = 32  # Usually same as tile_m for Triton
        
        # For CuTile, find the output tile (64x64 typically)
        cutile_main = [s for s in cutile_data if s.rows == s.cols]
        cutile_tile_m = cutile_main[0].rows if cutile_main else 64
        cutile_tile_n = cutile_main[0].cols if cutile_main else 64
        cutile_tile_k = 32  # Typical K tile
        
        # Compute K-loop iterations
        triton_k_iters = math.ceil(k / triton_tile_k)
        cutile_k_iters = math.ceil(k / cutile_tile_k)
        
        comparison.comparisons.append(ComparisonItem(
            category="Parallelization",
            aspect=f"K-loop iterations (K={k})",
            triton_value=f"{triton_k_iters} (K/{triton_tile_k})",
            cutile_value=f"{cutile_k_iters} (K/{cutile_tile_k})",
            match_status="✓" if triton_k_iters == cutile_k_iters else "~"
        ))
        
        # Compute grid size (M×N blocks)
        triton_grid_m = math.ceil(m / triton_tile_m)
        triton_grid_n = math.ceil(n / triton_tile_n)
        triton_grid = triton_grid_m * triton_grid_n
        
        cutile_grid_m = math.ceil(m / cutile_tile_m)
        cutile_grid_n = math.ceil(n / cutile_tile_n)
        cutile_grid = cutile_grid_m * cutile_grid_n
        
        comparison.comparisons.append(ComparisonItem(
            category="Parallelization",
            aspect=f"Grid size (M={m}, N={n})",
            triton_value=f"{triton_grid_m}×{triton_grid_n} = {triton_grid:,} blocks",
            cutile_value=f"{cutile_grid_m}×{cutile_grid_n} = {cutile_grid:,} blocks",
            match_status="~",
            notes="M×N parallelized across GPU blocks"
        ))
        
        # Total MMA operations
        total_mma = triton_grid * triton_k_iters
        comparison.comparisons.append(ComparisonItem(
            category="Parallelization",
            aspect="Total tile MMAs",
            triton_value=f"{total_mma:,}",
            cutile_value=f"{cutile_grid * cutile_k_iters:,}",
            match_status="≈",
            notes="grid_blocks × K_iterations"
        ))
    
    def _compare_tile_operations(
        self,
        triton: KernelIRSummary,
        cutile: KernelIRSummary,
        comparison: IRComparison
    ):
        """Compare tile load, store, and compute operations."""
        # Tile loads
        triton_loads = len(triton.tile_loads)
        cutile_loads = len(cutile.tile_loads)
        
        comparison.comparisons.append(ComparisonItem(
            category="Memory Ops",
            aspect="Tile loads (total)",
            triton_value=str(triton_loads),
            cutile_value=str(cutile_loads),
            match_status="✓" if triton_loads == cutile_loads else "~"
        ))
        
        # Loads by array
        triton_arrays = {}
        for op in triton.tile_loads:
            triton_arrays[op.array_name] = triton_arrays.get(op.array_name, 0) + 1
        
        cutile_arrays = {}
        for op in cutile.tile_loads:
            cutile_arrays[op.array_name] = cutile_arrays.get(op.array_name, 0) + 1
        
        for array in ['A', 'B']:
            comparison.comparisons.append(ComparisonItem(
                category="Memory Ops",
                aspect=f"Loads from {array}",
                triton_value=str(triton_arrays.get(array, 0)),
                cutile_value=str(cutile_arrays.get(array, 0)),
                match_status="✓" if triton_arrays.get(array, 0) == cutile_arrays.get(array, 0) else "~"
            ))
        
        # Tile stores
        comparison.comparisons.append(ComparisonItem(
            category="Memory Ops",
            aspect="Tile stores",
            triton_value=str(len(triton.tile_stores)),
            cutile_value=str(len(cutile.tile_stores)),
            match_status="✓" if len(triton.tile_stores) == len(cutile.tile_stores) else "~"
        ))
        
        # Compute operations
        triton_compute = len(triton.tile_computes)
        cutile_compute = len(cutile.tile_computes)
        
        # Identify compute type
        triton_compute_type = triton.tile_computes[0].op_type if triton.tile_computes else "none"
        cutile_compute_type = cutile.tile_computes[0].op_type if cutile.tile_computes else "none"
        
        comparison.comparisons.append(ComparisonItem(
            category="Compute",
            aspect="MMA/Dot operations",
            triton_value=f"{triton_compute} ({triton_compute_type})",
            cutile_value=f"{cutile_compute} ({cutile_compute_type})",
            match_status="✓" if triton_compute == cutile_compute else "~",
            notes="Both use matrix multiply-accumulate" if triton_compute == cutile_compute else ""
        ))
    
    def _compare_bounds_handling(
        self,
        triton: KernelIRSummary,
        cutile: KernelIRSummary,
        comparison: IRComparison
    ):
        """Compare how bounds/masking is handled."""
        triton_bounds = triton.bounds_checks[0].check_type if triton.bounds_checks else "none"
        cutile_bounds = cutile.bounds_checks[0].check_type if cutile.bounds_checks else "none"
        
        triton_detail = triton.bounds_checks[0].details if triton.bounds_checks else ""
        cutile_detail = cutile.bounds_checks[0].details if cutile.bounds_checks else ""
        
        # mask vs padding_mode are different approaches to same problem
        equiv = (triton_bounds == "mask" and cutile_bounds == "padding_mode")
        
        comparison.comparisons.append(ComparisonItem(
            category="Bounds Handling",
            aspect="Approach",
            triton_value=triton_bounds,
            cutile_value=cutile_bounds,
            match_status="~" if equiv else ("✓" if triton_bounds == cutile_bounds else "✗"),
            notes="Different approaches: explicit masking vs implicit zero-padding" if equiv else ""
        ))
    
    def _compare_data_types(
        self,
        triton: KernelIRSummary,
        cutile: KernelIRSummary,
        comparison: IRComparison
    ):
        """Compare data types and precision."""
        # Primary data type
        triton_shapes = list(triton.get_unique_tile_shapes())
        cutile_shapes = list(cutile.get_unique_tile_shapes())
        
        triton_dtypes = set(s.dtype for s in triton_shapes if 'i1' not in s.dtype and 'i32' not in s.dtype)
        cutile_dtypes = set(s.dtype for s in cutile_shapes if 'bool' not in s.dtype and 'int' not in s.dtype)
        
        comparison.comparisons.append(ComparisonItem(
            category="Data Types",
            aspect="Primary types",
            triton_value=", ".join(sorted(triton_dtypes)) or "unknown",
            cutile_value=", ".join(sorted(cutile_dtypes)) or "unknown",
            match_status="✓" if triton_dtypes == cutile_dtypes else "~"
        ))
        
        # Input precision for MMA
        triton_precision = triton.metadata.get('input_precision', 'unknown')
        # Check CuTile type conversions for tfloat32
        cutile_precision = 'unknown'
        for conv in cutile.type_conversions:
            if conv.tile_shape and 'tfloat32' in conv.tile_shape.dtype:
                cutile_precision = 'tf32'
                break
        
        comparison.comparisons.append(ComparisonItem(
            category="Data Types",
            aspect="MMA input precision",
            triton_value=triton_precision,
            cutile_value=cutile_precision,
            match_status="✓" if triton_precision == cutile_precision else "~",
            notes="Both use TF32 for MMA" if triton_precision == cutile_precision == 'tf32' else ""
        ))
    
    def _generate_summary(self, comparison: IRComparison) -> str:
        """Generate textual summary of comparison."""
        stats = comparison.get_match_statistics()
        total = sum(stats.values())
        
        lines = [
            f"=== Tile IR Comparison: {comparison.kernel_name} ===",
            "",
            f"Match Statistics:",
            f"  ✓ Full match:     {stats['match']}/{total}",
            f"  ≈ Equivalent:     {stats['equivalent']}/{total}",
            f"  ~ Different:      {stats['different']}/{total}",
            f"  ✗ Mismatch:       {stats['mismatch']}/{total}",
            "",
            "Comparison Details:",
            ""
        ]
        
        # Table header
        lines.append(f"{'Category':<18} {'Aspect':<24} {'Triton TTIR':<24} {'CuTile Typed IR':<24} {'Status':<6}")
        lines.append("─" * 100)
        
        current_category = ""
        for item in comparison.comparisons:
            cat_display = item.category if item.category != current_category else ""
            current_category = item.category
            
            lines.append(
                f"{cat_display:<18} {item.aspect:<24} {item.triton_value:<24} {item.cutile_value:<24} {item.match_status:<6}"
            )
            if item.notes:
                lines.append(f"{'':>43} └─ {item.notes}")
        
        lines.append("")
        lines.append("Legend: ✓ = match, ≈ = equivalent, ~ = different approach, ✗ = mismatch")
        
        return "\n".join(lines)
    
    def compare_from_files(
        self,
        triton_path: str,
        cutile_path: str
    ) -> IRComparison:
        """
        Compare IR from file paths.
        
        Args:
            triton_path: Path to Triton TTIR file
            cutile_path: Path to CuTile Typed IR file
            
        Returns:
            IRComparison
        """
        triton_ir = parse_triton_ir(triton_path)
        cutile_ir = parse_cutile_ir(cutile_path)
        
        return self.compare(triton_ir, cutile_ir)
    
    def save_comparison(
        self, 
        comparison: IRComparison, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save comparison to file.
        
        Args:
            comparison: IRComparison to save
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if not filename:
            filename = f"ir_comparison_{comparison.kernel_name}.txt"
        
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(comparison.summary_text)
            f.write("\n\n")
            f.write("=" * 60 + "\n")
            f.write("Detailed Shape Breakdown\n")
            f.write("=" * 60 + "\n\n")
            
            # Triton shapes by category
            f.write("Triton TTIR Tile Shapes:\n")
            triton_cats = comparison.triton_summary.get_shapes_by_category()
            f.write(f"  Data tiles:  {[str(s) for s in triton_cats.get('data', [])]}\n")
            f.write(f"  Index tiles: {[str(s) for s in triton_cats.get('index', [])]}\n")
            f.write(f"  Mask tiles:  {[str(s) for s in triton_cats.get('mask', [])]}\n")
            
            # CuTile shapes by category
            f.write("\nCuTile Typed IR Tile Shapes:\n")
            cutile_cats = comparison.cutile_summary.get_shapes_by_category()
            f.write(f"  Data tiles:  {[str(s) for s in cutile_cats.get('data', [])]}\n")
            f.write(f"  Index tiles: {[str(s) for s in cutile_cats.get('index', [])]}\n")
            f.write(f"  Mask tiles:  {[str(s) for s in cutile_cats.get('mask', [])]}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Operation Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Triton TTIR:\n")
            f.write(f"  Loads: {len(comparison.triton_summary.tile_loads)}, ")
            f.write(f"Stores: {len(comparison.triton_summary.tile_stores)}, ")
            f.write(f"Computes: {len(comparison.triton_summary.tile_computes)}, ")
            f.write(f"Loops: {len(comparison.triton_summary.loops)}\n")
            
            f.write("\nCuTile Typed IR:\n")
            f.write(f"  Loads: {len(comparison.cutile_summary.tile_loads)}, ")
            f.write(f"Stores: {len(comparison.cutile_summary.tile_stores)}, ")
            f.write(f"Computes: {len(comparison.cutile_summary.tile_computes)}, ")
            f.write(f"Loops: {len(comparison.cutile_summary.loops)}\n")
        
        return output_path
