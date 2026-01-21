# SPDX-License-Identifier: Apache-2.0
"""
Parser for CuTile Typed IR.

Extracts tile operations, control flow, and memory patterns from CuTile's
Python-level typed intermediate representation.
"""

import re
from typing import List, Optional
from .ir_common import (
    TileShape, TileOperation, LoopInfo, BoundsCheck, KernelIRSummary
)


class CuTileIRParser:
    """Parse CuTile Typed IR to extract tile semantics."""
    
    # Regex patterns for CuTile IR constructs
    TILE_TYPE_PATTERN = re.compile(r'Tile\[(\w+),\((\d+),(\d+)\)\]')
    TILE_LOAD_PATTERN = re.compile(r'tile_load_token_ordered\(array=(\w+),.*?order=\(([^)]+)\).*?padding_mode=(\w+\.?\w*)')
    TILE_STORE_PATTERN = re.compile(r'tile_store_token_ordered\(array=(\w+)')
    TILE_MMA_PATTERN = re.compile(r'tile_mma\(x=([^,]+),\s*y=([^,]+),\s*acc=([^)]+)\)')
    TILE_ASTYPE_PATTERN = re.compile(r'tile_astype\(x=([^)]+)\)')
    FOR_LOOP_PATTERN = re.compile(r'for\s+(\w+(?:\.\d+)?)\s+in\s+(\$\w+)')
    RANGE_PATTERN = re.compile(r'range\(start=(\$?\w+),\s*stop=(\$?\w+),\s*step=(\$?\w+)\)')
    NUM_TILES_PATTERN = re.compile(r'num_tiles\(array=(\w+),\s*axis=(\d+),\s*shape=\((\d+),\s*(\d+)\)')
    FUNC_PATTERN = re.compile(r'func\s+@(\w+)\(([^)]*)\):')
    ACCUMULATOR_INIT_PATTERN = re.compile(r'with\s+(\w+(?:\.\d+)?):.*?=\s*(\$\d+)')
    CONST_PATTERN = re.compile(r'\$(\d+):\s*const\s+int32\s*=\s*typed_const\(value=(\d+)\)')
    ARRAY_SHAPE_PATTERN = re.compile(r'Array\[(\w+),\(\?,\?\):\(\?,1\)\]')
    
    def __init__(self):
        self.constants = {}   # Track constant values
        self.variables = {}   # Track variable definitions
    
    def parse(self, ir_content: str) -> KernelIRSummary:
        """
        Parse CuTile Typed IR content and extract tile semantics.
        
        Args:
            ir_content: Raw typed IR text content
            
        Returns:
            KernelIRSummary with extracted information
        """
        # Extract kernel name
        func_match = self.FUNC_PATTERN.search(ir_content)
        kernel_name = func_match.group(1) if func_match else "unknown"
        
        summary = KernelIRSummary(
            name=kernel_name,
            source="cutile"
        )
        
        # Extract constants first
        self._extract_constants(ir_content)
        
        # Parse line by line
        lines = ir_content.split('\n')
        in_loop = False
        current_loop_ops = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Extract tile shapes
            self._extract_tile_shapes(line_stripped, summary)
            
            # Check for loop start
            loop_info = self._parse_loop_start(line_stripped, ir_content)
            if loop_info:
                in_loop = True
                current_loop_ops = []
                summary.loops.append(loop_info)
            
            # Check for loop end (continue statement)
            if line_stripped.startswith('continue '):
                if in_loop and summary.loops:
                    summary.loops[-1].body_ops = current_loop_ops
                in_loop = False
            
            # Parse tile operations
            load_op = self._parse_tile_load(line_stripped)
            if load_op:
                summary.tile_loads.append(load_op)
                if in_loop:
                    current_loop_ops.append(load_op)
                
                # Extract bounds check from padding mode
                if 'padding_mode=PaddingMode.ZERO' in line_stripped:
                    if not any(bc.check_type == "padding_mode" for bc in summary.bounds_checks):
                        summary.bounds_checks.append(BoundsCheck(
                            check_type="padding_mode",
                            details="ZERO padding for out-of-bounds access"
                        ))
            
            store_op = self._parse_tile_store(line_stripped)
            if store_op:
                summary.tile_stores.append(store_op)
            
            mma_op = self._parse_tile_mma(line_stripped)
            if mma_op:
                summary.tile_computes.append(mma_op)
                if in_loop:
                    current_loop_ops.append(mma_op)
            
            astype_op = self._parse_tile_astype(line_stripped, ir_content)
            if astype_op:
                summary.type_conversions.append(astype_op)
                if in_loop:
                    current_loop_ops.append(astype_op)
        
        # Extract num_tiles info for loop bounds
        self._extract_num_tiles_info(ir_content, summary)
        
        return summary
    
    def _extract_constants(self, content: str):
        """Extract constant values from IR."""
        self.constants = {}
        for match in self.CONST_PATTERN.finditer(content):
            self.constants[f"${match.group(1)}"] = int(match.group(2))
    
    def _extract_tile_shapes(self, line: str, summary: KernelIRSummary):
        """Extract Tile[dtype,(rows,cols)] declarations."""
        for match in self.TILE_TYPE_PATTERN.finditer(line):
            dtype = match.group(1)
            rows = int(match.group(2))
            cols = int(match.group(3))
            shape = TileShape(rows=rows, cols=cols, dtype=dtype)
            if shape not in summary.tile_shapes:
                summary.tile_shapes.append(shape)
    
    def _parse_loop_start(self, line: str, full_content: str) -> Optional[LoopInfo]:
        """Parse for loop structure."""
        match = self.FOR_LOOP_PATTERN.search(line)
        if match:
            loop_var = match.group(1)
            range_var = match.group(2)
            
            # Try to find the range definition
            range_match = self.RANGE_PATTERN.search(full_content)
            if range_match:
                start = range_match.group(1)
                stop = range_match.group(2)
                step = range_match.group(3)
                
                # Check if stop refers to num_tiles
                num_tiles_match = self.NUM_TILES_PATTERN.search(full_content)
                iterations_expr = "unknown"
                if num_tiles_match:
                    array_name = num_tiles_match.group(1)
                    axis = num_tiles_match.group(2)
                    tile_rows = num_tiles_match.group(3)
                    tile_cols = num_tiles_match.group(4)
                    iterations_expr = f"num_tiles({array_name}, axis={axis}, shape=({tile_rows},{tile_cols}))"
                
                return LoopInfo(
                    loop_var=loop_var,
                    start=str(self.constants.get(start, start)),
                    end=stop,
                    step=str(self.constants.get(step, step)),
                    iterations_expr=iterations_expr
                )
            
            return LoopInfo(
                loop_var=loop_var,
                start="0",
                end="?",
                step="1",
                iterations_expr="unknown"
            )
        
        return None
    
    def _parse_tile_load(self, line: str) -> Optional[TileOperation]:
        """Parse tile_load_token_ordered operation."""
        if 'tile_load_token_ordered' not in line:
            return None
        
        match = self.TILE_LOAD_PATTERN.search(line)
        if match:
            array_name = match.group(1)
            order = match.group(2)
            padding = match.group(3)
            
            # Try to extract tile shape from the same line or context
            shape_match = self.TILE_TYPE_PATTERN.search(line)
            shape = None
            if shape_match:
                shape = TileShape(
                    rows=int(shape_match.group(2)),
                    cols=int(shape_match.group(3)),
                    dtype=shape_match.group(1)
                )
            
            return TileOperation(
                op_type="load",
                tile_shape=shape,
                array_name=array_name,
                source_info=f"order={order}, padding={padding}"
            )
        
        # Fallback simpler pattern
        if 'array=' in line:
            array_match = re.search(r'array=(\w+)', line)
            if array_match:
                return TileOperation(
                    op_type="load",
                    array_name=array_match.group(1),
                    source_info=line[:60]
                )
        
        return None
    
    def _parse_tile_store(self, line: str) -> Optional[TileOperation]:
        """Parse tile_store_token_ordered operation."""
        if 'tile_store_token_ordered' not in line:
            return None
        
        match = self.TILE_STORE_PATTERN.search(line)
        if match:
            array_name = match.group(1)
            
            shape_match = self.TILE_TYPE_PATTERN.search(line)
            shape = None
            if shape_match:
                shape = TileShape(
                    rows=int(shape_match.group(2)),
                    cols=int(shape_match.group(3)),
                    dtype=shape_match.group(1)
                )
            
            return TileOperation(
                op_type="store",
                tile_shape=shape,
                array_name=array_name,
                source_info=line[:60]
            )
        
        return None
    
    def _parse_tile_mma(self, line: str) -> Optional[TileOperation]:
        """Parse tile_mma (matrix multiply-accumulate) operation."""
        if 'tile_mma' not in line:
            return None
        
        match = self.TILE_MMA_PATTERN.search(line)
        if match:
            # Extract output type from context
            shape_match = self.TILE_TYPE_PATTERN.search(line)
            shape = None
            if shape_match:
                shape = TileShape(
                    rows=int(shape_match.group(2)),
                    cols=int(shape_match.group(3)),
                    dtype=shape_match.group(1)
                )
            
            return TileOperation(
                op_type="mma",
                tile_shape=shape,
                array_name="C",
                source_info="tile matrix multiply-accumulate"
            )
        
        return None
    
    def _parse_tile_astype(self, line: str, full_content: str) -> Optional[TileOperation]:
        """Parse tile_astype (type conversion) operation."""
        if 'tile_astype' not in line:
            return None
        
        # Check what type we're converting to
        # Look for the output type pattern
        type_match = self.TILE_TYPE_PATTERN.search(line)
        if type_match:
            dtype = type_match.group(1)
            return TileOperation(
                op_type="astype",
                tile_shape=TileShape(
                    rows=int(type_match.group(2)),
                    cols=int(type_match.group(3)),
                    dtype=dtype
                ),
                source_info=f"convert to {dtype}"
            )
        
        return TileOperation(
            op_type="astype",
            source_info="type conversion"
        )
    
    def _extract_num_tiles_info(self, content: str, summary: KernelIRSummary):
        """Extract num_tiles computation info for metadata."""
        match = self.NUM_TILES_PATTERN.search(content)
        if match:
            summary.metadata['num_tiles'] = {
                'array': match.group(1),
                'axis': int(match.group(2)),
                'tile_shape': (int(match.group(3)), int(match.group(4)))
            }


def parse_cutile_ir(file_path: str) -> KernelIRSummary:
    """
    Parse CuTile IR from a file.
    
    Args:
        file_path: Path to typed IR file
        
    Returns:
        KernelIRSummary with extracted information
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    parser = CuTileIRParser()
    return parser.parse(content)
