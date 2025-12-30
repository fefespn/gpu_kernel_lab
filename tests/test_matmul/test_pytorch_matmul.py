# SPDX-License-Identifier: Apache-2.0
"""Tests for PyTorch matrix multiplication kernel."""

import pytest
import torch
import yaml
import os


def load_config():
    """Load test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


# Load config at module level
_CONFIG = load_config()

def _get_test_sizes():
    """Generate test sizes from M × N × K cartesian product."""
    from itertools import product
    tests_matmul = _CONFIG.get('tests_matmul', {})
    M_values = tests_matmul.get('M', [512, 1024])
    N_values = tests_matmul.get('N', [512, 1024])
    K_values = tests_matmul.get('K', [512, 1024])
    return list(product(M_values, N_values, K_values))

_TEST_SIZES = _get_test_sizes()


class TestPytorchMatmul:
    """Test suite for PyTorch matmul kernel."""
    
    @pytest.fixture
    def config(self):
        """Load configuration."""
        return load_config()
    
    @pytest.fixture
    def kernel(self, config):
        """Create kernel instance."""
        from kernels.matmul.pytorch_matmul import PytorchMatmul
        return PytorchMatmul(config)
    
    @pytest.fixture
    def is_compile_only(self, config):
        """Check if running in compile_only mode."""
        return config.get('hardware', {}).get('hardware_mode') == 'compile_only'
    
    @pytest.mark.parametrize("size", _TEST_SIZES)
    def test_correctness_float32(self, kernel, is_compile_only, size):
        """Test matmul correctness with float32."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        m, n, k = size
        A, B, C = kernel.create_inputs(m, n, k, dtype=torch.float32)
        kernel(A, B, C)
        
        expected = kernel.reference(A, B)
        torch.testing.assert_close(C, expected, rtol=1e-3, atol=1e-4)
    
    @pytest.mark.parametrize("size", [s for s in _TEST_SIZES if max(s) <= 1024])
    def test_correctness_float16(self, kernel, is_compile_only, size):
        """Test matmul correctness with float16."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        m, n, k = size
        A, B, C = kernel.create_inputs(m, n, k, dtype=torch.float16)
        kernel(A, B, C)
        
        expected = kernel.reference(A, B)
        # Looser tolerance for fp16
        torch.testing.assert_close(C, expected, rtol=1e-2, atol=1e-2)
    
    def test_random_inputs(self, kernel, is_compile_only):
        """Test with random matrix dimensions."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        m, n, k = 512, 768, 1024
        A, B, C = kernel.create_inputs(m, n, k)
        kernel(A, B, C)
        
        expected = kernel.reference(A, B)
        torch.testing.assert_close(C, expected, rtol=1e-3, atol=1e-4)
    
    def test_compile(self, kernel):
        """Test kernel compilation."""
        result = kernel.compile()
        assert result['backend'] == 'pytorch'
        assert 'status' in result
