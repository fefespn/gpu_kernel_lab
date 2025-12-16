# SPDX-License-Identifier: Apache-2.0
"""
cuTile vector addition kernel with support for:
- Native mode: Normal execution on Blackwell (sm_100)
- Compile-only mode: Monkey-patch get_sm_arch() to compile for sm_100 on any GPU
"""

import os
from typing import Optional, Dict, Any

# Must set CUDA_TILE_TEMP_DIR before importing cuda.tile
_OUTPUT_DIR = None

def _setup_cutile_env(output_dir: str):
    """Setup environment for cuTile."""
    global _OUTPUT_DIR
    _OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.environ['CUDA_TILE_TEMP_DIR'] = output_dir


class CutileAdd:
    """cuTile-based vector addition kernel."""
    
    name = "cutile"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cuTile kernel.
        
        Args:
            config: Configuration dict with hardware settings
        """
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
        self.target_sm = self.config.get('hardware', {}).get('target_sm', 100)
        self.tile_size = self.config.get('kernels', {}).get('add', {}).get('tile_size', 16)
        self.output_dir = self.config.get('output_dir', 'outputs')
        
        self._kernel = None
        self._compiled = False
        self._ct = None
        self._cp = None
    
    def _import_and_patch(self):
        """Import cuTile and apply sm_100 monkey-patch if needed."""
        if self._ct is not None:
            return
        
        # Setup environment first
        _setup_cutile_env(self.output_dir)
        
        import cupy as cp
        import cuda.tile as ct
        
        self._cp = cp
        self._ct = ct
        
        # Monkey-patch for compile-only mode
        if self.hardware_mode == 'compile_only':
            import cuda.tile._compile as ct_compile
            ct_compile.get_sm_arch = lambda: f'sm_{self.target_sm}'
    
    def _create_kernel(self):
        """Create the cuTile kernel function."""
        if self._kernel is not None:
            return
        
        self._import_and_patch()
        ct = self._ct
        
        @ct.kernel
        def cutile_vector_add(a, b, c, tile_size: ct.Constant[int]):
            # Get the 1D pid
            pid = ct.bid(0)
            
            # Load input tiles
            a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
            b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
            
            # Perform elementwise addition
            result = a_tile + b_tile
            
            # Store result
            ct.store(c, index=(pid,), tile=result)
        
        self._kernel = cutile_vector_add
    
    def __call__(self, a, b, c) -> None:
        """
        Perform vector addition: c = a + b.
        
        Args:
            a: First input array (CuPy)
            b: Second input array (CuPy)
            c: Output array (will be modified in-place)
        """
        self._create_kernel()
        ct = self._ct
        cp = self._cp
        
        n_elements = a.size
        grid = (ct.cdiv(n_elements, self.tile_size), 1, 1)
        
        if self.hardware_mode == 'compile_only':
            # In compile-only mode, we trigger compilation but catch execution error
            try:
                ct.launch(
                    cp.cuda.get_current_stream(),
                    grid,
                    self._kernel,
                    (a, b, c, self.tile_size)
                )
            except Exception as e:
                # Expected to fail on non-Blackwell hardware
                raise RuntimeError(
                    f"Execution failed (expected in compile_only mode): {e}"
                )
        else:
            ct.launch(
                cp.cuda.get_current_stream(),
                grid,
                self._kernel,
                (a, b, c, self.tile_size)
            )
    
    def compile(self) -> Dict[str, Any]:
        """
        Compile the kernel for sm_100.
        In compile-only mode, triggers JIT compilation.
        
        Returns:
            Dict with compilation artifacts
        """
        self._create_kernel()
        ct = self._ct
        cp = self._cp
        
        # Create dummy inputs to trigger compilation
        size = self.tile_size * 4
        a = cp.ones(size, dtype=cp.float32)
        b = cp.ones(size, dtype=cp.float32)
        c = cp.zeros_like(a)
        
        grid = (ct.cdiv(size, self.tile_size), 1, 1)
        
        artifacts = {}
        
        try:
            ct.launch(
                cp.cuda.get_current_stream(),
                grid,
                self._kernel,
                (a, b, c, self.tile_size)
            )
            self._compiled = True
            status = 'compiled_and_executed'
        except Exception as e:
            # In compile-only mode, compilation happens but execution fails
            self._compiled = True
            status = f'compiled_only (exec error: {type(e).__name__})'
        
        # Check for artifacts in output directory
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.endswith('.ptx'):
                    artifacts['ptx'] = os.path.join(self.output_dir, f)
                elif f.endswith('.cubin'):
                    artifacts['cubin'] = os.path.join(self.output_dir, f)
        
        return {
            'backend': self.name,
            'status': status,
            'target_sm': self.target_sm,
            'artifacts': artifacts
        }
    
    @staticmethod
    def create_inputs(size: int, dtype=None):
        """
        Create input arrays for testing/benchmarking.
        
        Args:
            size: Number of elements
            dtype: Data type (default: float32)
            
        Returns:
            Tuple of (a, b, c) CuPy arrays
        """
        import cupy as cp
        if dtype is None:
            dtype = cp.float32
        
        a = cp.ones(size, dtype=dtype)
        b = cp.ones(size, dtype=dtype)
        c = cp.zeros(size, dtype=dtype)
        return a, b, c
    
    @staticmethod
    def reference(a, b):
        """Reference implementation."""
        return a + b
