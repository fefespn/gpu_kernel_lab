# SPDX-License-Identifier: Apache-2.0
"""
Common data structures for tile-based IR analysis.

Provides normalized representations for comparing Triton TTIR and CuTile Typed IR.
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any


@dataclass
class TileShape:
    """Tile dimensions (e.g., 32x32)."""
    rows: int
    cols: int
    dtype: str = "unknown"
    category: str = "data"  # "data", "index", "mask"
    
    def __hash__(self):
        return hash((self.rows, self.cols, self.dtype, self.category))
    
    def __eq__(self, other):
        if not isinstance(other, TileShape):
            return False
        return (self.rows == other.rows and self.cols == other.cols 
                and self.dtype == other.dtype and self.category == other.category)
    
    def __str__(self):
        return f"{self.rows}×{self.cols} ({self.dtype})"


@dataclass
class TileOperation:
    """A tile operation (load, store, mma, dot)."""
    op_type: str           # "load", "store", "mma", "dot", "astype"
    tile_shape: Optional[TileShape] = None
    array_name: str = ""   # Which array (A, B, C)
    source_info: str = ""  # Additional context from IR
    
    def __str__(self):
        shape_str = str(self.tile_shape) if self.tile_shape else "?"
        return f"{self.op_type}({self.array_name}, {shape_str})"


@dataclass
class LoopInfo:
    """Loop structure information."""
    loop_var: str              # Variable name (k, etc.)
    start: str = "0"           # Start value/expression
    end: str = "?"             # End value/expression  
    step: str = "1"            # Step value
    iterations_expr: str = ""  # Human-readable iteration description
    body_ops: List[TileOperation] = field(default_factory=list)
    
    def __str__(self):
        return f"for {self.loop_var} in [{self.start}, {self.end}): {len(self.body_ops)} ops"


@dataclass
class BoundsCheck:
    """Bounds checking / masking information."""
    check_type: str     # "mask", "padding_mode", "bounds_check"
    details: str = ""   # Additional info


@dataclass
class KernelIRSummary:
    """Normalized summary of a kernel's tile IR."""
    name: str
    source: str  # "triton" or "cutile"
    
    # Tile dimensions used
    tile_shapes: List[TileShape] = field(default_factory=list)
    
    # Operations categorized
    tile_loads: List[TileOperation] = field(default_factory=list)
    tile_stores: List[TileOperation] = field(default_factory=list)
    tile_computes: List[TileOperation] = field(default_factory=list)
    type_conversions: List[TileOperation] = field(default_factory=list)
    
    # Control flow
    loops: List[LoopInfo] = field(default_factory=list)
    
    # Bounds handling
    bounds_checks: List[BoundsCheck] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_unique_tile_shapes(self) -> Set[TileShape]:
        """Get all unique tile shapes used."""
        shapes = set()
        for op in self.tile_loads + self.tile_stores + self.tile_computes:
            if op.tile_shape:
                shapes.add(op.tile_shape)
        return shapes
    
    def get_shapes_by_category(self) -> Dict[str, List[TileShape]]:
        """Get tile shapes organized by category (data, index, mask)."""
        result: Dict[str, List[TileShape]] = {"data": [], "index": [], "mask": []}
        for shape in self.tile_shapes:
            cat = getattr(shape, 'category', 'data')
            if cat not in result:
                result[cat] = []
            if shape not in result[cat]:
                result[cat].append(shape)
        return result
    
    def get_arrays_accessed(self) -> Dict[str, List[str]]:
        """Get arrays accessed and operation types."""
        arrays: Dict[str, List[str]] = {}
        for op in self.tile_loads + self.tile_stores:
            if op.array_name:
                if op.array_name not in arrays:
                    arrays[op.array_name] = []
                arrays[op.array_name].append(op.op_type)
        return arrays
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary for comparison."""
        return {
            "name": self.name,
            "source": self.source,
            "tile_shapes": [str(s) for s in self.get_unique_tile_shapes()],
            "num_loads": len(self.tile_loads),
            "num_stores": len(self.tile_stores),
            "num_computes": len(self.tile_computes),
            "num_loops": len(self.loops),
            "arrays_accessed": self.get_arrays_accessed(),
            "bounds_check_type": self.bounds_checks[0].check_type if self.bounds_checks else "none",
        }
