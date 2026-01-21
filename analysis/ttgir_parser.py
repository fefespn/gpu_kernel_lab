"""
TTGIR (Triton GPU IR) Parser

Parses MLIR-style TTGIR to extract dependency graphs for visualization.
Inspired by the Twill paper: "Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs"
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple


class OpCategory(Enum):
    """Operation categories for visualization coloring."""
    MEMORY = "memory"       # Load, store, async_copy
    COMPUTE = "compute"     # Dot, arithmetic
    CONTROL = "control"     # Loops, branches, yield
    CONSTANT = "constant"   # Constants
    OTHER = "other"


@dataclass
class Operation:
    """Represents a single TTGIR operation."""
    name: str                           # SSA value name (e.g., "%a_51")
    op_type: str                        # Operation type (e.g., "tt.dot", "arith.addi")
    category: OpCategory                # Category for coloring
    operands: List[str] = field(default_factory=list)  # Input SSA values
    result_type: str = ""               # Result type
    line_num: int = 0                   # Line number in source
    attributes: Dict[str, str] = field(default_factory=dict)
    
    @property
    def short_name(self) -> str:
        """Get a short display name."""
        return self.name.replace("%", "")


@dataclass  
class DependencyEdge:
    """Represents a data dependency edge."""
    source: str                         # Producer operation name
    target: str                         # Consumer operation name
    is_loop_carried: bool = False       # Is this a loop-carried dependency?
    operand_index: int = 0              # Which operand position


@dataclass
class DependencyGraph:
    """Dependency graph extracted from TTGIR."""
    operations: Dict[str, Operation] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    loop_bounds: Optional[Tuple[int, int]] = None  # (start_line, end_line) of main loop
    
    def get_ops_by_category(self, category: OpCategory) -> List[Operation]:
        """Get all operations of a given category."""
        return [op for op in self.operations.values() if op.category == category]
    
    def get_successors(self, op_name: str) -> List[str]:
        """Get all operations that depend on the given operation."""
        return [e.target for e in self.edges if e.source == op_name]
    
    def get_predecessors(self, op_name: str) -> List[str]:
        """Get all operations that the given operation depends on."""
        return [e.source for e in self.edges if e.target == op_name]


class TTGIRParser:
    """Parser for TTGIR (Triton GPU IR)."""
    
    # Operation type to category mapping
    OP_CATEGORIES = {
        # Memory operations
        'tt.load': OpCategory.MEMORY,
        'tt.store': OpCategory.MEMORY,
        'ttg.async_copy_global_to_local': OpCategory.MEMORY,
        'ttg.local_load': OpCategory.MEMORY,
        'ttg.local_alloc': OpCategory.MEMORY,
        'ttg.local_dealloc': OpCategory.MEMORY,
        'ttg.async_wait': OpCategory.MEMORY,
        'ttg.async_commit_group': OpCategory.MEMORY,
        'ttg.memdesc_index': OpCategory.MEMORY,
        
        # Compute operations
        'tt.dot': OpCategory.COMPUTE,
        'arith.addi': OpCategory.COMPUTE,
        'arith.subi': OpCategory.COMPUTE,
        'arith.muli': OpCategory.COMPUTE,
        'arith.divsi': OpCategory.COMPUTE,
        'arith.remsi': OpCategory.COMPUTE,
        'arith.addf': OpCategory.COMPUTE,
        'arith.subf': OpCategory.COMPUTE,
        'arith.mulf': OpCategory.COMPUTE,
        'arith.divf': OpCategory.COMPUTE,
        'arith.minsi': OpCategory.COMPUTE,
        'arith.maxsi': OpCategory.COMPUTE,
        'arith.cmpi': OpCategory.COMPUTE,
        'arith.cmpf': OpCategory.COMPUTE,
        'arith.select': OpCategory.COMPUTE,
        'arith.andi': OpCategory.COMPUTE,
        'arith.ori': OpCategory.COMPUTE,
        'arith.xori': OpCategory.COMPUTE,
        'arith.extsi': OpCategory.COMPUTE,
        'arith.extui': OpCategory.COMPUTE,
        'arith.trunci': OpCategory.COMPUTE,
        'arith.sitofp': OpCategory.COMPUTE,
        'arith.fptosi': OpCategory.COMPUTE,
        'math.exp': OpCategory.COMPUTE,
        'math.exp2': OpCategory.COMPUTE,
        'math.log': OpCategory.COMPUTE,
        'math.log2': OpCategory.COMPUTE,
        'math.sqrt': OpCategory.COMPUTE,
        'math.rsqrt': OpCategory.COMPUTE,
        
        # Tensor operations (also compute)
        'tt.splat': OpCategory.COMPUTE,
        'tt.broadcast': OpCategory.COMPUTE,
        'tt.expand_dims': OpCategory.COMPUTE,
        'tt.make_range': OpCategory.COMPUTE,
        'tt.addptr': OpCategory.COMPUTE,
        'tt.get_program_id': OpCategory.COMPUTE,
        'ttg.convert_layout': OpCategory.COMPUTE,
        
        # Control flow
        'scf.for': OpCategory.CONTROL,
        'scf.yield': OpCategory.CONTROL,
        'scf.if': OpCategory.CONTROL,
        'scf.while': OpCategory.CONTROL,
        'tt.return': OpCategory.CONTROL,
        
        # Constants
        'arith.constant': OpCategory.CONSTANT,
    }
    
    def __init__(self):
        self.graph = DependencyGraph()
        self._current_line = 0
        self._in_loop = False
        self._loop_depth = 0
    
    def parse(self, ttgir_text: str) -> DependencyGraph:
        """
        Parse TTGIR text and extract dependency graph.
        
        Args:
            ttgir_text: Raw TTGIR text content
            
        Returns:
            DependencyGraph with operations and edges
        """
        self.graph = DependencyGraph()
        self._current_line = 0
        
        lines = ttgir_text.split('\n')
        
        for i, line in enumerate(lines):
            self._current_line = i + 1
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            
            # Skip module attributes and location info
            if line.startswith('module') or line.startswith('#loc'):
                continue
            
            # Track loop boundaries
            if 'scf.for' in line:
                self._in_loop = True
                self._loop_depth += 1
                if self.graph.loop_bounds is None:
                    self.graph.loop_bounds = (self._current_line, None)
            
            if 'scf.yield' in line and self._in_loop:
                if self.graph.loop_bounds and self.graph.loop_bounds[1] is None:
                    self.graph.loop_bounds = (self.graph.loop_bounds[0], self._current_line)
            
            # Parse SSA operations
            self._parse_line(line)
        
        # Build edges from operand references
        self._build_edges()
        
        return self.graph
    
    def _parse_line(self, line: str) -> Optional[Operation]:
        """Parse a single line of TTGIR."""
        # Pattern: %name = op_type operands : types loc(...)
        # Or: op_type operands : types loc(...)  (for void ops like tt.store)
        
        # Match SSA definition: %name = ...
        ssa_match = re.match(r'(%[\w_]+(?::\d+)?)\s*=\s*(.+)', line)
        
        if ssa_match:
            name = ssa_match.group(1)
            rest = ssa_match.group(2)
        else:
            # Check for void operations (store, return, etc.)
            if any(op in line for op in ['tt.store', 'tt.return', 'scf.yield', 'ttg.local_dealloc']):
                name = f"_void_{self._current_line}"
                rest = line
            else:
                return None
        
        # Extract operation type
        op_match = re.match(r'([\w.]+)\s*(.*)$', rest)
        if not op_match:
            return None
        
        op_type = op_match.group(1)
        operands_str = op_match.group(2)
        
        # Get category
        category = self.OP_CATEGORIES.get(op_type, OpCategory.OTHER)
        
        # Extract operands (SSA values)
        operands = self._extract_operands(operands_str)
        
        # Extract result type (between : and loc)
        type_match = re.search(r':\s*([^l]+?)(?:\s+loc\(|$)', operands_str)
        result_type = type_match.group(1).strip() if type_match else ""
        
        # Create operation
        op = Operation(
            name=name,
            op_type=op_type,
            category=category,
            operands=operands,
            result_type=result_type,
            line_num=self._current_line
        )
        
        self.graph.operations[name] = op
        return op
    
    def _extract_operands(self, text: str) -> List[str]:
        """Extract SSA value references from operand text."""
        # Match %name patterns (SSA values)
        # Handle both simple %name and %name#N (tuple results)
        pattern = r'%[\w_]+(?:#\d+)?(?::\d+)?'
        matches = re.findall(pattern, text)
        return list(set(matches))  # Unique operands
    
    def _build_edges(self):
        """Build dependency edges from operand references."""
        for op in self.graph.operations.values():
            for i, operand in enumerate(op.operands):
                # Normalize operand name (handle %name#0 -> %name)
                base_operand = re.sub(r'#\d+', '', operand)
                
                # Check if this operand is defined by another operation
                if operand in self.graph.operations:
                    edge = DependencyEdge(
                        source=operand,
                        target=op.name,
                        operand_index=i
                    )
                    self.graph.edges.append(edge)
                elif base_operand in self.graph.operations:
                    edge = DependencyEdge(
                        source=base_operand,
                        target=op.name,
                        operand_index=i
                    )
                    self.graph.edges.append(edge)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the parsed graph."""
        stats = {
            'total_operations': len(self.graph.operations),
            'total_edges': len(self.graph.edges),
        }
        
        for category in OpCategory:
            ops = self.graph.get_ops_by_category(category)
            stats[f'{category.value}_ops'] = len(ops)
        
        return stats


def parse_ttgir(ttgir_text: str) -> DependencyGraph:
    """Convenience function to parse TTGIR text."""
    parser = TTGIRParser()
    return parser.parse(ttgir_text)


def parse_ttgir_file(filepath: str) -> DependencyGraph:
    """Parse TTGIR from a file."""
    with open(filepath, 'r') as f:
        return parse_ttgir(f.read())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ttgir_parser.py <ttgir_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    graph = parse_ttgir_file(filepath)
    
    parser = TTGIRParser()
    parser.graph = graph
    stats = parser.get_statistics()
    
    print(f"\n=== TTGIR Dependency Graph Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n=== Operations by Category ===")
    for category in OpCategory:
        ops = graph.get_ops_by_category(category)
        if ops:
            print(f"\n{category.value.upper()} ({len(ops)}):")
            for op in ops[:10]:  # Show first 10
                print(f"  {op.short_name}: {op.op_type}")
            if len(ops) > 10:
                print(f"  ... and {len(ops) - 10} more")
