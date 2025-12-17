# GPU Kernel Lab

A Python-based framework for benchmarking GPU kernel implementations across multiple backends (cuBLAS, Triton, cuTile, PyTorch), with support for correctness testing, performance metrics, and SASS analysis.

## Features

- **Multi-backend support**: Compare identical kernels across cuBLAS, Triton, cuTile, and PyTorch
- **Hardware flexibility**: Run on Blackwell (sm_100) or compile-only on older GPUs
- **Performance metrics**: Latency (µs), TFLOPS, memory bandwidth (GB/s)
- **SASS analysis**: Extract and compare SASS between Triton and cuTile
- **Config-driven**: YAML configuration for test/benchmark selection

## Project Structure

```
gpu_kernel_lab/
├── kernels/                    # Kernel implementations
│   └── add/                    # Vector addition kernel
│       ├── __init__.py         # Backend registry
│       ├── cublas_add.py       # cuBLAS implementation
│       ├── triton_add.py       # Triton implementation
│       ├── cutile_add.py       # cuTile implementation
│       └── pytorch_add.py      # PyTorch/Triton implementation
├── tests/                      # Pytest test suites
│   └── test_add/
│       ├── test_cublas_add.py
│       ├── test_triton_add.py
│       ├── test_cutile_add.py
│       └── test_pytorch_add.py
├── benchmarks/                 # Benchmarking tools
│   ├── metrics.py              # Latency, TFLOPS, bandwidth
│   └── benchmark_runner.py     # Config-driven benchmark runner
├── analysis/                   # SASS analysis tools
│   ├── sass_extractor.py       # Extract PTX/CUBIN/SASS
│   └── sass_comparator.py      # Compare SASS between backends
├── outputs/                    # Generated artifacts (organized by backend)
│   ├── triton/                 # Triton: add_sm100.cubin, .ptx, .sass
│   ├── cutile/                 # cuTile: add_sm100.cubin, .sass
│   └── sass_comparison_*.txt   # SASS comparison reports
├── config.yaml                 # Master configuration
├── run_tests.py                # Test runner CLI
├── run_benchmarks.py           # Benchmark runner CLI
├── compare_sass.py             # SASS comparison CLI (separate phase)
└── requirements.txt            # Python dependencies
```

## Installation

```bash
cd gpu_kernel_lab
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- PyTorch 2.x with CUDA
- Triton 3.0+
- CuPy (for cuBLAS)
- cuTile (cuda.tile)

## Configuration

Edit `config.yaml` to customize:

```yaml
hardware:
  # "native" for Blackwell, "compile_only" for older GPUs
  hardware_mode: compile_only
  target_sm: 100

benchmarks:
  enabled: true
  backends: [triton, cutile, pytorch, cublas]
  sizes: [65536, 262144, 1048576]
  warmup_iterations: 10
  benchmark_iterations: 100

sass_analysis:
  enabled: true
  compare_pairs: [[triton, cutile]]
```

## Usage

### Run Tests

```bash
# Run all enabled tests
python run_tests.py

# Run specific backend
python run_tests.py --backend triton

# Run compile-only tests (for non-Blackwell GPUs)
python run_tests.py --compile-only
```

### Run Benchmarks

```bash
# Run all benchmarks
python run_benchmarks.py

# Run specific backends and sizes
python run_benchmarks.py --backend triton cublas --sizes 65536,262144

# Compile-only mode (no execution)
python run_benchmarks.py --compile-only
```

### Compare SASS (Separate Phase)

SASS comparison is a separate step that doesn't require benchmark execution.
This is useful on non-Blackwell GPUs where you can compile but not run.

```bash
# Compare SASS between triton and cutile (default)
python compare_sass.py

# Specify backends to compare
python compare_sass.py --backends triton cutile

# Extract SASS only (no comparison)
python compare_sass.py --extract-only

# Verbose output with SASS line counts
python compare_sass.py --verbose
```

### Output

Results are saved to `outputs/`, organized by backend:

```
outputs/
├── triton/
│   ├── add_sm100.cubin          # Compiled kernel binary
│   ├── add_sm100.ptx            # PTX intermediate
│   └── add_sm100.sass           # SASS disassembly
├── cutile/
│   ├── add_sm100.cubin
│   └── add_sm100.sass
├── benchmark_results_*.json      # Detailed JSON results
├── benchmark_results_*.csv       # CSV for spreadsheet import
└── sass_comparison_*.txt         # SASS comparison reports
```

**Note:** SASS is automatically extracted during kernel compilation.

---

## Adding New Kernels

### Step 1: Create Kernel Folder

```bash
mkdir kernels/matmul
touch kernels/matmul/__init__.py
```

### Step 2: Implement Backend Classes

Create a file for each backend (e.g., `kernels/matmul/triton_matmul.py`):

```python
# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl
from typing import Optional, Dict, Any

@triton.jit
def _matmul_kernel(...):
    # Your Triton kernel here
    pass

class TritonMatmul:
    """Triton-based matrix multiplication."""
    
    name = "triton"  # Required: backend identifier
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_mode = self.config.get('hardware', {}).get('hardware_mode', 'native')
    
    def __call__(self, a, b, c) -> None:
        """Execute the kernel."""
        # Launch kernel here
        pass
    
    def compile(self) -> Dict[str, Any]:
        """Compile kernel (for compile_only mode)."""
        # AOT compilation logic
        return {'backend': self.name, 'status': 'compiled', 'artifacts': {}}
    
    @staticmethod
    def create_inputs(size: int, dtype=torch.float32):
        """Create test inputs."""
        # Return (a, b, c) tensors
        pass
    
    @staticmethod
    def reference(a, b):
        """Reference implementation for correctness check."""
        return a @ b
```

### Step 3: Register Backend

Edit `kernels/matmul/__init__.py`:

```python
from .triton_matmul import TritonMatmul
from .cublas_matmul import CublasMatmul
# ... other backends

BACKENDS = {
    'triton': TritonMatmul,
    'cublas': CublasMatmul,
}

def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    return BACKENDS[name]
```

### Step 4: Create Tests

Create `tests/test_matmul/test_triton_matmul.py`:

```python
import pytest
import torch
from kernels.matmul.triton_matmul import TritonMatmul

class TestTritonMatmul:
    @pytest.fixture
    def kernel(self):
        return TritonMatmul()
    
    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_correctness(self, kernel, size):
        a, b, c = kernel.create_inputs(size)
        kernel(a, b, c)
        torch.cuda.synchronize()
        
        expected = kernel.reference(a, b)
        torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)
    
    def test_compile(self, kernel):
        result = kernel.compile()
        assert result['status'] == 'compiled'
```

### Step 5: Update Config

Add the new kernel to `config.yaml`:

```yaml
kernels:
  add:
    enabled: true
    backends: [triton, cutile, pytorch, cublas]
  matmul:  # New kernel
    enabled: true
    backends: [triton, cublas]
    tile_size: 64
```

### Step 6: Update Metrics (Optional)

For custom FLOPs calculation, add to `benchmarks/metrics.py`:

```python
def get_matmul_metrics(m: int, n: int, k: int, dtype_bytes: int = 4) -> Dict[str, int]:
    """Metrics for matrix multiplication."""
    flops = 2 * m * n * k  # multiply-add
    bytes_transferred = (m * k + k * n + m * n) * dtype_bytes
    return {'flops': flops, 'bytes': bytes_transferred}
```

---

## Hardware Modes

| Mode | GPU | Execution | Use Case |
|------|-----|-----------|----------|
| `native` | Blackwell (sm_100) | Full | Production benchmarks |
| `compile_only` | Any (e.g., RTX 3060) | Compile only | Development, SASS analysis |

In `compile_only` mode:
- **Triton**: Uses AOT compilation with `GPUTarget("cuda", 100, 64)`
- **cuTile**: Monkey-patches `get_sm_arch()` to return `'sm_100'`
- Execution tests are skipped; only compile tests run

## License

Apache-2.0
