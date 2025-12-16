# SPDX-License-Identifier: Apache-2.0
"""
SASS extraction from Triton and cuTile compiled kernels.
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
            kernel_name: Name of the kernel to extract
            
        Returns:
            SassArtifacts with paths to extracted files
        """
        from kernels.add.triton_add import TritonAdd
        
        kernel = TritonAdd(self.config)
        result = kernel.compile()
        
        artifacts = SassArtifacts(backend='triton', kernel_name=kernel_name)
        
        if 'ptx' in result['artifacts']:
            artifacts.ptx_path = result['artifacts']['ptx']
        
        if 'cubin' in result['artifacts']:
            artifacts.cubin_path = result['artifacts']['cubin']
            
            # Extract SASS using cuobjdump
            sass_path = os.path.join(
                self.output_dir,
                f"triton_{kernel_name}_sm{self.target_sm}.sass"
            )
            
            try:
                with open(sass_path, 'w') as f:
                    subprocess.run(
                        [self.cuobjdump_path, '-sass', artifacts.cubin_path],
                        stdout=f,
                        check=True
                    )
                artifacts.sass_path = sass_path
                
                # Read SASS content
                with open(sass_path) as f:
                    artifacts.sass_content = f.read()
                    
            except subprocess.CalledProcessError as e:
                print(f"Error extracting SASS: {e}")
            except FileNotFoundError:
                print(f"cuobjdump not found at {self.cuobjdump_path}")
        
        # Also try Python-based SASS extraction
        sass_code = kernel.get_sass()
        if sass_code and not artifacts.sass_content:
            artifacts.sass_content = sass_code
            sass_py_path = os.path.join(
                self.output_dir,
                f"triton_{kernel_name}_sm{self.target_sm}_python.sass"
            )
            with open(sass_py_path, 'w') as f:
                f.write(sass_code)
            artifacts.sass_path = sass_py_path
        
        return artifacts
    
    def extract_cutile(self, kernel_name: str = 'add') -> SassArtifacts:
        """
        Extract SASS from cuTile kernel.
        
        Args:
            kernel_name: Name of the kernel to extract
            
        Returns:
            SassArtifacts with paths to extracted files
        """
        from kernels.add.cutile_add import CutileAdd
        
        kernel = CutileAdd(self.config)
        result = kernel.compile()
        
        artifacts = SassArtifacts(backend='cutile', kernel_name=kernel_name)
        
        # cuTile stores artifacts in CUDA_TILE_TEMP_DIR
        if 'ptx' in result['artifacts']:
            artifacts.ptx_path = result['artifacts']['ptx']
        
        if 'cubin' in result['artifacts']:
            artifacts.cubin_path = result['artifacts']['cubin']
            
            # Extract SASS
            sass_path = os.path.join(
                self.output_dir,
                f"cutile_{kernel_name}_sm{self.target_sm}.sass"
            )
            
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
                    
            except subprocess.CalledProcessError as e:
                print(f"Error extracting SASS: {e}")
            except FileNotFoundError:
                print(f"cuobjdump not found at {self.cuobjdump_path}")
        
        # Look for artifacts in output directory
        if not artifacts.cubin_path:
            for f in os.listdir(self.output_dir):
                if f.endswith('.cubin') and 'cutile' not in f.lower():
                    # Likely a cuTile-generated cubin
                    cubin_path = os.path.join(self.output_dir, f)
                    artifacts.cubin_path = cubin_path
                    
                    sass_path = os.path.join(
                        self.output_dir,
                        f"cutile_{kernel_name}_sm{self.target_sm}.sass"
                    )
                    
                    try:
                        with open(sass_path, 'w') as out:
                            subprocess.run(
                                [self.cuobjdump_path, '-sass', cubin_path],
                                stdout=out,
                                check=True
                            )
                        artifacts.sass_path = sass_path
                        
                        with open(sass_path) as rf:
                            artifacts.sass_content = rf.read()
                    except Exception as e:
                        print(f"Error extracting cuTile SASS: {e}")
                    
                    break
        
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
