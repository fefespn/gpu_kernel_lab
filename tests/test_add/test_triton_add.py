# SPDX-License-Identifier: Apache-2.0
"""
Pytest tests for Triton vector addition kernel.
"""

import pytest
import torch
import numpy as np
import yaml
import os
from kernels.add.triton_add import TritonAdd


def load_config():
    """Load configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


class TestTritonAdd:
    """Test suite for Triton vector addition."""
    
    @pytest.fixture
    def config(self):
        """Load config."""
        return load_config()
    
    @pytest.fixture
    def kernel(self, config):
        """Create kernel instance."""
        return TritonAdd(config)
    
    @pytest.fixture
    def is_compile_only(self, config):
        """Check if in compile-only mode."""
        return config.get('hardware', {}).get('hardware_mode', 'native') == 'compile_only'
    
    @pytest.mark.parametrize("size", [1024, 4096, 16384, 65536])
    def test_correctness_float32(self, kernel, is_compile_only, size):
        """Test correctness with float32 inputs."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        a, b, c = kernel.create_inputs(size, dtype=torch.float32)
        
        # Run kernel
        kernel(a, b, c)
        torch.cuda.synchronize()
        
        # Compare with reference
        expected = kernel.reference(a, b)
        
        torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)
    
    @pytest.mark.parametrize("size", [1024, 4096, 16384])
    def test_correctness_float16(self, kernel, is_compile_only, size):
        """Test correctness with float16 inputs."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        a, b, c = kernel.create_inputs(size, dtype=torch.float16)
        
        # Run kernel
        kernel(a, b, c)
        torch.cuda.synchronize()
        
        # Compare with reference
        expected = kernel.reference(a, b)
        
        torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)
    
    def test_random_inputs(self, kernel, is_compile_only):
        """Test with random inputs."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        size = 8192
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)
        c = torch.zeros(size, device='cuda', dtype=torch.float32)
        
        kernel(a, b, c)
        torch.cuda.synchronize()
        
        expected = kernel.reference(a, b)
        torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)
    
    def test_compile_aot(self, kernel):
        """Test AOT compilation for sm_100."""
        result = kernel.compile()
        
        assert result['backend'] == 'triton'
        assert 'compiled' in result['status']
        assert result['target_sm'] == kernel.target_sm
        
        # Check artifacts were created
        if result['artifacts']:
            for artifact_type, path in result['artifacts'].items():
                assert os.path.exists(path), f"{artifact_type} artifact not found: {path}"
