# SPDX-License-Identifier: Apache-2.0
"""
Parser for Triton TTIR (Triton IR) dialect.

Extracts tile operations, control flow, and memory patterns from Triton's 
MLIR-based intermediate representation.
"""

import re
from typing import List, Optional, Tuple
from .ir_common import (
    TileShape, TileOperation, LoopInfo, BoundsCheck, KernelIRSummary
)


class TritonIRParser:
    """Parse Triton TTIR to extract tile semantics."""
    
    # Regex patterns for Triton IR constructs
    TENSOR_SHAPE_PATTERN = re.compile(r'tensor<(\d+)x(\d+)x([^>]+)>')
    LOAD_PATTERN = re.compile(r'tt\.load\s+(%\w+)')
    STORE_PATTERN = re.compile(r'tt\.store\s+(%\w+),\s*(%\w+)')
    DOT_PATTERN = re.compile(r'tt\.dot\s+(%\w+),\s*(%\w+),\s*(%\w+).*?:\s*tensor<(\d+)x(\d+)x([^>]+)>')
    SCF_FOR_PATTERN = re.compile(r'scf\.for\s+(%\w+)\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)')
    FUNC_PATTERN = re.compile(r'tt\.func\s+public\s+@(\w+)\s*\(([^)]*)\)')
    MAKE_RANGE_PATTERN = re.compile(r'tt\.make_range\s*\{end\s*=\s*(\d+).*?start\s*=\s*(\d+)')
    SPLAT_PATTERN = re.compile(r'tt\.splat\s+(%\w+)\s*:\s*([^-]+)\s*->\s*tensor<([^>]+)>')
    MASK_PATTERN = re.compile(r'(%\w+_mask\w*)')
    CONST_PATTERN = re.compile(r'(%c\d+_i32)\s*=\s*arith\.constant\s+(\d+)')
    PTR_TYPE_PATTERN = re.compile(r'!tt\.ptr<([^>]+)>')
    
    def __init__(self):
        self.constants = {}  # Track constant values
    
    def parse(self, ir_content: str) -> KernelIRSummary:
        """
        Parse Triton TTIR content and extract tile semantics.
        
        Args:
            ir_content: Raw TTIR text content
            
        Returns:
            KernelIRSummary with extracted information
        """
        # Extract kernel name
        func_match = self.FUNC_PATTERN.search(ir_content)
        kernel_name = func_match.group(1) if func_match else "unknown"
        
        summary = KernelIRSummary(
            name=kernel_name,
            source="triton"
        )
        
        # Extract constants first for reference
        self._extract_constants(ir_content)
        
        # Parse line by line for operations
        lines = ir_content.split('\n')
        in_loop = False
        current_loop_ops = []
        
        for line in lines:
            line = line.strip()
            
            # Track tensor shapes
            self._extract_tile_shapes(line, summary)
            
            # Check for loop structure
            loop_info = self._parse_loop_start(line)
            if loop_info:
                in_loop = True
                current_loop_ops = []
                summary.loops.append(loop_info)
            
            # Check for loop end
            if 'scf.yield' in line and in_loop and summary.loops:
                summary.loops[-1].body_ops = current_loop_ops
                in_loop = False
            
            # Parse tile operations
            load_op = self._parse_load(line, ir_content)
            if load_op:
                summary.tile_loads.append(load_op)
                if in_loop:
                    current_loop_ops.append(load_op)
            
            store_op = self._parse_store(line, ir_content)
            if store_op:
                summary.tile_stores.append(store_op)
            
            dot_op = self._parse_dot(line)
            if dot_op:
                summary.tile_computes.append(dot_op)
                if in_loop:
                    current_loop_ops.append(dot_op)
            
            # Parse bounds checks
            if self.MASK_PATTERN.search(line) and ('cmpi slt' in line or 'andi' in line):
                if not any(bc.check_type == "mask" for bc in summary.bounds_checks):
                    summary.bounds_checks.append(BoundsCheck(
                        check_type="mask",
                        details="Explicit tensor masks for bounds checking"
                    ))
        
        # Extract input precision from dot operation
        self._extract_precision_info(ir_content, summary)
        
        return summary
    
    def _extract_constants(self, content: str):
        """Extract constant values from IR."""
        self.constants = {}
        for match in self.CONST_PATTERN.finditer(content):
            self.constants[match.group(1)] = int(match.group(2))
    
    def _extract_tile_shapes(self, line: str, summary: KernelIRSummary):
        """Extract tensor/tile shape declarations with category."""
        for match in self.TENSOR_SHAPE_PATTERN.finditer(line):
            rows, cols, dtype = int(match.group(1)), int(match.group(2)), match.group(3)
            # Filter out pointer types, only keep data tiles
            if 'ptr' not in dtype:
                # Categorize by dtype
                if 'i1' in dtype or 'bool' in dtype.lower():
                    category = "mask"
                elif 'i32' in dtype or 'i64' in dtype or 'index' in dtype.lower():
                    category = "index"
                else:
                    category = "data"
                
                shape = TileShape(rows=rows, cols=cols, dtype=dtype.strip(), category=category)
                if shape not in summary.tile_shapes:
                    summary.tile_shapes.append(shape)
    
    def _parse_loop_start(self, line: str) -> Optional[LoopInfo]:
        """Parse scf.for loop start."""
        match = self.SCF_FOR_PATTERN.search(line)
        if match:
            loop_var = match.group(1)
            start_var = match.group(2)
            end_var = match.group(3)
            step_var = match.group(4)
            
            # Try to resolve to concrete values
            start = str(self.constants.get(start_var, start_var))
            end = str(self.constants.get(end_var, end_var))
            step = str(self.constants.get(step_var, step_var))
            
            return LoopInfo(
                loop_var=loop_var,
                start=start,
                end=end,
                step=step,
                iterations_expr=f"ceil(K/{self.constants.get('%c32_i32', 32)})" if '%1' in end else end
            )
        return None
    
    def _parse_load(self, line: str, full_content: str) -> Optional[TileOperation]:
        """Parse tt.load operation."""
        if 'tt.load' not in line:
            return None
        
        # Extract tensor type from the load
        type_match = self.TENSOR_SHAPE_PATTERN.search(line)
        shape = None
        if type_match:
            dtype = type_match.group(3).replace('!tt.ptr<', '').replace('>', '')
            shape = TileShape(
                rows=int(type_match.group(1)),
                cols=int(type_match.group(2)),
                dtype=dtype
            )
        
        # Try to determine which array (A or B)
        array_name = "?"
        if '%a_ptrs' in line or '%a_mask' in line or '= tt.load %a_ptrs' in line:
            array_name = "A"
        elif '%b_ptrs' in line or '%b_mask' in line or '= tt.load %b_ptrs' in line:
            array_name = "B"
        else:
            # Check the variable being loaded from
            load_match = self.LOAD_PATTERN.search(line)
            if load_match:
                ptr_var = load_match.group(1)
                if 'a_' in ptr_var.lower():
                    array_name = "A"
                elif 'b_' in ptr_var.lower():
                    array_name = "B"
        
        return TileOperation(
            op_type="load",
            tile_shape=shape,
            array_name=array_name,
            source_info=line[:80] if len(line) > 80 else line
        )
    
    def _parse_store(self, line: str, full_content: str) -> Optional[TileOperation]:
        """Parse tt.store operation."""
        if 'tt.store' not in line:
            return None
        
        type_match = self.TENSOR_SHAPE_PATTERN.search(line)
        shape = None
        if type_match:
            dtype = type_match.group(3).replace('!tt.ptr<', '').replace('>', '')
            shape = TileShape(
                rows=int(type_match.group(1)),
                cols=int(type_match.group(2)),
                dtype=dtype
            )
        
        # Output is typically array C
        array_name = "C"
        if '%c_ptrs' in line or '%c_mask' in line:
            array_name = "C"
        
        return TileOperation(
            op_type="store",
            tile_shape=shape,
            array_name=array_name,
            source_info=line[:80] if len(line) > 80 else line
        )
    
    def _parse_dot(self, line: str) -> Optional[TileOperation]:
        """Parse tt.dot (matrix multiply) operation."""
        if 'tt.dot' not in line:
            return None
        
        match = self.DOT_PATTERN.search(line)
        if match:
            rows, cols = int(match.group(4)), int(match.group(5))
            dtype = match.group(6)
            shape = TileShape(rows=rows, cols=cols, dtype=dtype)
            
            # Check for precision hints
            precision_info = ""
            if 'inputPrecision = tf32' in line:
                precision_info = "tf32"
            elif 'inputPrecision = ieee' in line:
                precision_info = "ieee"
            
            return TileOperation(
                op_type="dot",
                tile_shape=shape,
                array_name="C",
                source_info=f"mma with {precision_info} precision" if precision_info else "mma"
            )
        
        # Fallback for simpler patterns
        if 'tt.dot' in line:
            return TileOperation(
                op_type="dot",
                tile_shape=None,
                source_info=line[:80]
            )
        
        return None
    
    def _extract_precision_info(self, content: str, summary: KernelIRSummary):
        """Extract input/output precision information."""
        if 'inputPrecision = tf32' in content:
            summary.metadata['input_precision'] = 'tf32'
        elif 'inputPrecision = ieee' in content:
            summary.metadata['input_precision'] = 'ieee'
        
        # Check for type conversions
        if 'arith.truncf' in content or 'arith.extf' in content:
            summary.type_conversions.append(TileOperation(
                op_type="astype",
                source_info="explicit precision conversion"
            ))


def parse_triton_ir(file_path: str) -> KernelIRSummary:
    """
    Parse Triton IR from a file.
    
    Args:
        file_path: Path to TTIR file
        
    Returns:
        KernelIRSummary with extracted information
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    parser = TritonIRParser()
    return parser.parse(content)
