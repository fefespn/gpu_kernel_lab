# SPDX-License-Identifier: Apache-2.0
"""
Pytest tests for cuBLAS vector addition kernel.
"""

import pytest
import cupy as cp
import numpy as np
from kernels.add.cublas_add import CublasAdd


class TestCublasAdd:
    """Test suite for cuBLAS vector addition."""
    
    @pytest.fixture
    def kernel(self):
        """Create kernel instance."""
        return CublasAdd()
    
    @pytest.mark.parametrize("size", [1024, 4096, 16384, 65536])
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
    
    @pytest.mark.parametrize("size", [1024, 4096, 16384])
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
        """Test compilation (no-op for cuBLAS)."""
        result = kernel.compile()
        assert result['backend'] == 'cublas'
        assert result['status'] == 'ready'
