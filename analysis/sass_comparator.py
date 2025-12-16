# SPDX-License-Identifier: Apache-2.0
"""
SASS comparison between Triton and cuTile compiled kernels.
"""

import os
import re
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class InstructionStats:
    """Statistics about SASS instructions."""
    total_instructions: int
    instruction_counts: Dict[str, int]
    memory_ops: int  # LDG, STG, LDS, STS, etc.
    compute_ops: int  # FADD, FMUL, FFMA, etc.
    control_ops: int  # BRA, EXIT, etc.
    
    def to_dict(self) -> Dict:
        return {
            'total_instructions': self.total_instructions,
            'instruction_counts': dict(self.instruction_counts),
            'memory_ops': self.memory_ops,
            'compute_ops': self.compute_ops,
            'control_ops': self.control_ops
        }


@dataclass
class SassComparison:
    """Comparison results between two SASS outputs."""
    backend_a: str
    backend_b: str
    stats_a: InstructionStats
    stats_b: InstructionStats
    diff_summary: str
    common_instructions: List[str]
    unique_to_a: List[str]
    unique_to_b: List[str]


class SassComparator:
    """Compare SASS output between backends."""
    
    # Common instruction patterns
    MEMORY_OPS = {'LDG', 'STG', 'LDS', 'STS', 'LD', 'ST', 'LDGSTS', 'LDSL', 'STSL'}
    COMPUTE_OPS = {'FADD', 'FMUL', 'FFMA', 'FMNMX', 'FSET', 'HADD2', 'HMUL2', 'HFMA2',
                   'DADD', 'DMUL', 'DFMA', 'IADD', 'IMUL', 'IMAD', 'IADD3'}
    CONTROL_OPS = {'BRA', 'EXIT', 'RET', 'CALL', 'SSY', 'SYNC', 'BAR', 'BSSY', 'BSYNC'}
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize SASS comparator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = self.config.get('output_dir', 'outputs')
    
    def parse_sass(self, sass_content: str) -> List[str]:
        """
        Parse SASS content and extract instruction mnemonics.
        
        Args:
            sass_content: Raw SASS code
            
        Returns:
            List of instruction mnemonics
        """
        instructions = []
        
        # SASS instruction pattern: starts with spaces, then hex address, then opcode
        # Example: /*0008*/  MOV R1, c[0x0][0x28] ;
        pattern = r'/\*[0-9a-f]+\*/\s+(\w+)'
        
        for match in re.finditer(pattern, sass_content, re.IGNORECASE):
            instr = match.group(1).upper()
            # Skip pseudo-instructions
            if not instr.startswith('.') and instr not in ['NOP']:
                instructions.append(instr)
        
        return instructions
    
    def compute_stats(self, instructions: List[str]) -> InstructionStats:
        """
        Compute statistics for a list of instructions.
        
        Args:
            instructions: List of instruction mnemonics
            
        Returns:
            InstructionStats object
        """
        counts = Counter(instructions)
        
        memory_ops = sum(counts.get(op, 0) for op in self.MEMORY_OPS)
        compute_ops = sum(counts.get(op, 0) for op in self.COMPUTE_OPS)
        control_ops = sum(counts.get(op, 0) for op in self.CONTROL_OPS)
        
        return InstructionStats(
            total_instructions=len(instructions),
            instruction_counts=dict(counts),
            memory_ops=memory_ops,
            compute_ops=compute_ops,
            control_ops=control_ops
        )
    
    def compare(
        self,
        sass_a: str,
        sass_b: str,
        name_a: str = 'backend_a',
        name_b: str = 'backend_b'
    ) -> SassComparison:
        """
        Compare two SASS outputs.
        
        Args:
            sass_a: SASS content from first backend
            sass_b: SASS content from second backend
            name_a: Name of first backend
            name_b: Name of second backend
            
        Returns:
            SassComparison with detailed comparison
        """
        instrs_a = self.parse_sass(sass_a)
        instrs_b = self.parse_sass(sass_b)
        
        stats_a = self.compute_stats(instrs_a)
        stats_b = self.compute_stats(instrs_b)
        
        # Find unique and common instructions
        set_a = set(stats_a.instruction_counts.keys())
        set_b = set(stats_b.instruction_counts.keys())
        
        common = sorted(set_a & set_b)
        unique_a = sorted(set_a - set_b)
        unique_b = sorted(set_b - set_a)
        
        # Generate summary
        diff_lines = []
        diff_lines.append(f"Comparison: {name_a} vs {name_b}")
        diff_lines.append("-" * 50)
        diff_lines.append(f"Total instructions: {stats_a.total_instructions} vs {stats_b.total_instructions}")
        diff_lines.append(f"Memory ops: {stats_a.memory_ops} vs {stats_b.memory_ops}")
        diff_lines.append(f"Compute ops: {stats_a.compute_ops} vs {stats_b.compute_ops}")
        diff_lines.append(f"Control ops: {stats_a.control_ops} vs {stats_b.control_ops}")
        diff_lines.append("")
        
        if unique_a:
            diff_lines.append(f"Instructions unique to {name_a}: {', '.join(unique_a)}")
        if unique_b:
            diff_lines.append(f"Instructions unique to {name_b}: {', '.join(unique_b)}")
        
        diff_lines.append("")
        diff_lines.append("Instruction count comparison:")
        for instr in sorted(common, key=lambda x: -(stats_a.instruction_counts.get(x, 0) + stats_b.instruction_counts.get(x, 0))):
            count_a = stats_a.instruction_counts.get(instr, 0)
            count_b = stats_b.instruction_counts.get(instr, 0)
            if count_a != count_b:
                diff_lines.append(f"  {instr}: {count_a} vs {count_b} (diff: {count_a - count_b:+d})")
        
        return SassComparison(
            backend_a=name_a,
            backend_b=name_b,
            stats_a=stats_a,
            stats_b=stats_b,
            diff_summary="\n".join(diff_lines),
            common_instructions=common,
            unique_to_a=unique_a,
            unique_to_b=unique_b
        )
    
    def compare_from_files(
        self,
        sass_path_a: str,
        sass_path_b: str,
        name_a: str = 'backend_a',
        name_b: str = 'backend_b'
    ) -> SassComparison:
        """
        Compare SASS from file paths.
        
        Args:
            sass_path_a: Path to first SASS file
            sass_path_b: Path to second SASS file
            name_a: Name of first backend
            name_b: Name of second backend
            
        Returns:
            SassComparison
        """
        with open(sass_path_a) as f:
            sass_a = f.read()
        with open(sass_path_b) as f:
            sass_b = f.read()
        
        return self.compare(sass_a, sass_b, name_a, name_b)
    
    def save_comparison(self, comparison: SassComparison, filename: Optional[str] = None) -> str:
        """
        Save comparison results to a file.
        
        Args:
            comparison: SassComparison to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"sass_comparison_{comparison.backend_a}_vs_{comparison.backend_b}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(comparison.diff_summary)
            f.write("\n\n")
            f.write("=" * 50)
            f.write(f"\n{comparison.backend_a} instruction counts:\n")
            for instr, count in sorted(comparison.stats_a.instruction_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {instr}: {count}\n")
            f.write("\n")
            f.write(f"{comparison.backend_b} instruction counts:\n")
            for instr, count in sorted(comparison.stats_b.instruction_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {instr}: {count}\n")
        
        print(f"Comparison saved to {filepath}")
        return filepath
    
    def run_comparison(self, kernel_name: str = 'add') -> Optional[SassComparison]:
        """
        Run full comparison between Triton and cuTile SASS.
        
        Args:
            kernel_name: Name of kernel to compare
            
        Returns:
            SassComparison or None if files not found
        """
        sass_config = self.config.get('sass_analysis', {})
        if not sass_config.get('enabled', False):
            print("SASS analysis disabled in config")
            return None
        
        compare_pairs = sass_config.get('compare_pairs', [['triton', 'cutile']])
        
        for pair in compare_pairs:
            backend_a, backend_b = pair
            
            target_sm = self.config.get('hardware', {}).get('target_sm', 100)
            sass_a_path = os.path.join(self.output_dir, f"{backend_a}_{kernel_name}_sm{target_sm}.sass")
            sass_b_path = os.path.join(self.output_dir, f"{backend_b}_{kernel_name}_sm{target_sm}.sass")
            
            if not os.path.exists(sass_a_path):
                print(f"SASS file not found: {sass_a_path}")
                continue
            if not os.path.exists(sass_b_path):
                print(f"SASS file not found: {sass_b_path}")
                continue
            
            comparison = self.compare_from_files(sass_a_path, sass_b_path, backend_a, backend_b)
            
            if sass_config.get('save_artifacts', True):
                self.save_comparison(comparison)
            
            print(comparison.diff_summary)
            return comparison
        
        return None
