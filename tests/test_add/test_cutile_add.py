# SPDX-License-Identifier: Apache-2.0
"""
Pytest tests for cuTile vector addition kernel.
"""

import pytest
import numpy as np
import yaml
import os


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


class TestCutileAdd:
    """Test suite for cuTile vector addition."""
    
    @pytest.fixture
    def config(self):
        """Load config."""
        return load_config()
    
    @pytest.fixture
    def kernel(self, config):
        """Create kernel instance."""
        from kernels.add.cutile_add import CutileAdd
        return CutileAdd(config)
    
    @pytest.fixture
    def is_compile_only(self, config):
        """Check if in compile-only mode."""
        return config.get('hardware', {}).get('hardware_mode', 'native') == 'compile_only'
    
    @pytest.mark.parametrize("size", _TEST_SIZES)
    def test_correctness_float32(self, kernel, is_compile_only, size):
        """Test correctness with float32 inputs."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        import cupy as cp
        a, b, c = kernel.create_inputs(size, dtype=cp.float32)
        
        # Run kernel
        kernel(a, b, c)
        cp.cuda.Stream.null.synchronize()
        
        # Compare with reference
        expected = kernel.reference(a, b)
        
        cp.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)
    
    @pytest.mark.parametrize("size", _TEST_SIZES_FP16)
    def test_correctness_float16(self, kernel, is_compile_only, size):
        """Test correctness with float16 inputs."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        import cupy as cp
        a, b, c = kernel.create_inputs(size, dtype=cp.float16)
        
        # Run kernel
        kernel(a, b, c)
        cp.cuda.Stream.null.synchronize()
        
        # Compare with reference
        expected = kernel.reference(a, b)
        
        cp.testing.assert_allclose(c, expected, rtol=1e-2, atol=1e-2)
    
    def test_random_inputs(self, kernel, is_compile_only):
        """Test with random inputs."""
        if is_compile_only:
            pytest.skip("Skipping execution test in compile_only mode")
        
        import cupy as cp
        size = 8192
        a = cp.random.randn(size, dtype=cp.float32)
        b = cp.random.randn(size, dtype=cp.float32)
        c = cp.zeros(size, dtype=cp.float32)
        
        kernel(a, b, c)
        cp.cuda.Stream.null.synchronize()
        
        expected = kernel.reference(a, b)
        cp.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)
    
    def test_compile_sm100(self, kernel, is_compile_only):
        """Test compilation for sm_100."""
        result = kernel.compile()
        
        assert result['backend'] == 'cutile'
        assert 'compiled' in result['status']
        assert result['target_sm'] == kernel.target_sm
        
        # In compile_only mode, execution should fail but compilation succeeds
        if is_compile_only:
            assert 'compiled_only' in result['status']
