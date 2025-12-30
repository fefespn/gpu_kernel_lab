# SPDX-License-Identifier: Apache-2.0
"""
Pytest tests for cuBLAS vector addition kernel.
"""

import pytest
import cupy as cp
import numpy as np
import yaml
import os
from kernels.add.cublas_add import CublasAdd


def load_config():
    """Load configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Load config at module level for parametrization
_CONFIG = load_config()
_TEST_SIZES = _CONFIG.get('tests', {}).get('sizes', [1024, 4096, 16384, 65536])
_TEST_SIZES_FP16 = [s for s in _TEST_SIZES if s <= 16384]  # Limit fp16 sizes


class TestCublasAdd:
    """Test suite for cuBLAS vector addition."""
    
    @pytest.fixture
    def kernel(self):
        """Create kernel instance."""
        return CublasAdd(load_config())
    
    @pytest.mark.parametrize("size", _TEST_SIZES)
    def test_correctness_float32(self, kernel, size):
        """Test correctness with float32 inputs."""
        a, b, c = kernel.create_inputs(size, dtype=cp.float32)
        
        # Run kernel
        kernel(a, b, c)
        cp.cuda.Stream.null.synchronize()
        
        # Compare with reference
        expected = kernel.reference(cp.asnumpy(a), cp.asnumpy(b))
        actual = cp.asnumpy(c)
        
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    
    @pytest.mark.parametrize("size", _TEST_SIZES_FP16)
    def test_correctness_float16(self, kernel, size):
        """Test correctness with float16 inputs."""
        a, b, c = kernel.create_inputs(size, dtype=cp.float16)
        
        # Run kernel
        kernel(a, b, c)
        cp.cuda.Stream.null.synchronize()
        
        # Compare with reference (lower tolerance for fp16)
        expected = kernel.reference(cp.asnumpy(a), cp.asnumpy(b))
        actual = cp.asnumpy(c)
        
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)
    
    def test_random_inputs(self, kernel):
        """Test with random inputs."""
        size = 8192
        a = cp.random.randn(size, dtype=cp.float32)
        b = cp.random.randn(size, dtype=cp.float32)
        c = cp.zeros(size, dtype=cp.float32)
        
        kernel(a, b, c)
        cp.cuda.Stream.null.synchronize()
        
        expected = kernel.reference(cp.asnumpy(a), cp.asnumpy(b))
        actual = cp.asnumpy(c)
        
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    
    def test_compile(self, kernel):
        """Test compilation and cubin extraction."""
        result = kernel.compile()
        assert result['backend'] == 'cublas'
        assert result['status'] == 'compiled'
