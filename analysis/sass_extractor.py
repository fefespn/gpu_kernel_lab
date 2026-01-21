# SPDX-License-Identifier: Apache-2.0
"""
SASS extraction from Triton and cuTile compiled kernels.

# Compare add kernel (default)
python compare_sass.py --backends triton cutile

# Compare matmul kernel
python compare_sass.py --kernel matmul --backends triton cutile

# Compare ALL kernels (add + matmul)
python compare_sass.py --kernel all --backends triton cutile

# Extract only, no comparison
python compare_sass.py --kernel all --extract-only
"""

import os
import subprocess
import yaml
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class SassArtifacts:
    """Container for SASS extraction artifacts."""
    backend: str
    kernel_name: str
    ptx_path: Optional[str] = None
    cubin_path: Optional[str] = None
    sass_path: Optional[str] = None
    sass_content: Optional[str] = None


class SassExtractor:
    """Extract SASS from compiled GPU kernels."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize SASS extractor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = self.config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        hardware = self.config.get('hardware', {})
        self.cuobjdump_path = hardware.get('cuobjdump_path', '/usr/local/cuda/bin/cuobjdump')
        self.target_sm = hardware.get('target_sm', 100)
    
    def extract_triton(self, kernel_name: str = 'add') -> SassArtifacts:
        """
        Extract SASS from Triton kernel.
        
        Args:
            kernel_name: Name of the kernel to extract ('add' or 'matmul')
            
        Returns:
            SassArtifacts with paths to extracted files
        """
        # Dynamically import the correct kernel based on kernel_name
        if kernel_name == 'matmul':
            from kernels.matmul.triton_matmul import TritonMatmul
            kernel = TritonMatmul(self.config)
            backend_output_dir = os.path.join(self.output_dir, 'triton_matmul')
        else:
            from kernels.add.triton_add import TritonAdd
            kernel = TritonAdd(self.config)
            backend_output_dir = os.path.join(self.output_dir, 'triton')
        
        result = kernel.compile()
        
        artifacts = SassArtifacts(backend='triton', kernel_name=kernel_name)
        
        if 'ptx' in result['artifacts']:
            artifacts.ptx_path = result['artifacts']['ptx']
        
        if 'cubin' in result['artifacts']:
            artifacts.cubin_path = result['artifacts']['cubin']
        
        # SASS is now extracted during compile, just read it
        if 'sass' in result['artifacts']:
            artifacts.sass_path = result['artifacts']['sass']
            with open(artifacts.sass_path) as f:
                artifacts.sass_content = f.read()
        else:
            # Fallback: try to extract manually
            sass_path = os.path.join(backend_output_dir, f"{kernel_name}_sm{self.target_sm}.sass")
            
            if artifacts.cubin_path:
                try:
                    with open(sass_path, 'w') as f:
                        subprocess.run(
                            [self.cuobjdump_path, '-sass', artifacts.cubin_path],
                            stdout=f,
                            check=True
                        )
                    artifacts.sass_path = sass_path
                    with open(sass_path) as f:
                        artifacts.sass_content = f.read()
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Error extracting SASS: {e}")
            
            # Python-based extraction as last fallback (only for add kernel)
            if not artifacts.sass_content and kernel_name == 'add':
                sass_code = kernel.get_sass()
                if sass_code:
                    artifacts.sass_content = sass_code
                    with open(sass_path, 'w') as f:
                        f.write(sass_code)
                    artifacts.sass_path = sass_path
        
        return artifacts
    
    def extract_cutile(self, kernel_name: str = 'add') -> SassArtifacts:
        """
        Extract SASS from cuTile kernel.
        
        Args:
            kernel_name: Name of the kernel to extract ('add' or 'matmul')
            
        Returns:
            SassArtifacts with paths to extracted files
        """
        # Dynamically import the correct kernel based on kernel_name
        if kernel_name == 'matmul':
            from kernels.matmul.cutile_matmul import CutileMatmul
            kernel = CutileMatmul(self.config)
            backend_output_dir = os.path.join(self.output_dir, 'cutile_matmul')
        else:
            from kernels.add.cutile_add import CutileAdd
            kernel = CutileAdd(self.config)
            backend_output_dir = os.path.join(self.output_dir, 'cutile')
        
        result = kernel.compile()
        
        artifacts = SassArtifacts(backend='cutile', kernel_name=kernel_name)
        
        if 'ptx' in result['artifacts']:
            artifacts.ptx_path = result['artifacts']['ptx']
        
        if 'cubin' in result['artifacts']:
            artifacts.cubin_path = result['artifacts']['cubin']
        
        # SASS is now extracted during compile, just read it
        if 'sass' in result['artifacts']:
            artifacts.sass_path = result['artifacts']['sass']
            with open(artifacts.sass_path) as f:
                artifacts.sass_content = f.read()
        else:
            # Fallback: try to extract manually
            sass_path = os.path.join(backend_output_dir, f"{kernel_name}_sm{self.target_sm}.sass")
            
            if artifacts.cubin_path:
                try:
                    with open(sass_path, 'w') as f:
                        subprocess.run(
                            [self.cuobjdump_path, '-sass', artifacts.cubin_path],
                            stdout=f,
                            check=True
                        )
                    artifacts.sass_path = sass_path
                    with open(sass_path) as f:
                        artifacts.sass_content = f.read()
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Error extracting SASS: {e}")
        
        return artifacts
    
    def extract_all(self, kernel_name: str = 'add') -> Dict[str, SassArtifacts]:
        """
        Extract SASS from all backends that support it.
        
        Args:
            kernel_name: Name of the kernel
            
        Returns:
            Dict mapping backend name to SassArtifacts
        """
        results = {}
        
        print("Extracting Triton SASS...")
        try:
            results['triton'] = self.extract_triton(kernel_name)
            print(f"  PTX: {results['triton'].ptx_path}")
            print(f"  CUBIN: {results['triton'].cubin_path}")
            print(f"  SASS: {results['triton'].sass_path}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("Extracting cuTile SASS...")
        try:
            results['cutile'] = self.extract_cutile(kernel_name)
            print(f"  PTX: {results['cutile'].ptx_path}")
            print(f"  CUBIN: {results['cutile'].cubin_path}")
            print(f"  SASS: {results['cutile'].sass_path}")
        except Exception as e:
            print(f"  Error: {e}")
        
        return results
