#!/usr/bin/env python3
"""
Modal.com deployment for running GPU Kernel Lab tests on B200 GPUs.

Usage:
    # Run the add kernel test
    modal run run_modal.py --test-path tests/test_add/test_cutile_add.py

    # Run all tests
    modal run run_modal.py

    # Deploy as a persistent service
    modal deploy run_modal.py
"""

import modal
import os

# Create the Modal app
app = modal.App("gpu-kernel-lab")

# Define the CUDA image with all dependencies
# Using CUDA 13.1 for Blackwell (sm_100) support
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu24.04",
        add_python="3.11"
    )
    # Create symlinks for CUDA 12.x compatibility (cupy-cuda12x expects libnvrtc.so.12)
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so.12",
        "ln -sf /usr/local/cuda/lib64/libnvrtc-builtins.so.13.1 /usr/local/cuda/lib64/libnvrtc-builtins.so.12",
        "ldconfig",
    )
    .apt_install(
        "git",
        "build-essential",
        "cmake",
    )
    .pip_install(
        # Core dependencies
        "cupy-cuda12x>=13.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
        # cuTile (cuda.tile)
        "cuda-tile",
        # PyTorch and Triton
        "torch",
        "triton",
    )
    
    # Add local project files to the image
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[
            ".git", "__pycache__", ".pytest_cache", "venv", "outputs",
            "*.pyc", "*.pyo", "*.egg-info"
        ]
    )
)


@app.function(
    gpu="B200",
    image=cuda_image,
    timeout=600,  # 10 minutes
)
def run_tests(test_path: str = None, verbose: bool = True):
    """
    Run GPU kernel tests on B200.

    Args:
        test_path: Specific test file or directory to run (relative to project root)
        verbose: Enable verbose output

    Returns:
        Dict with test results
    """
    import subprocess
    import os

    os.chdir("/app")

    # Update config to use native mode (we're on B200!)
    import yaml
    config_path = "/app/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure native mode for B200
    config['hardware']['hardware_mode'] = 'native'
    config['hardware']['target_sm'] = 100  # Blackwell

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Use run_tests.py which respects config.yaml (including name_filter)
    cmd = ["python", "run_tests.py"]

    if not verbose:
        cmd.append("--quiet")

    # If specific test path, pass extra pytest args
    if test_path:
        cmd.extend(["--", test_path])

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    # Run tests
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }


@app.function(
    gpu="B200",
    image=cuda_image,
    timeout=600,
)
def run_benchmarks(config_path: str = "config.yaml", backend: str = None, sizes: list = None):
    """
    Run GPU kernel benchmarks on B200.

    Args:
        config_path: Path to config file (default: config.yaml)
        backend: Specific backend to benchmark (triton, cutile, pytorch, cublas)
        sizes: List of sizes to benchmark

    Returns:
        Dict with benchmark results
    """
    import subprocess
    import os

    os.chdir("/app")

    # Update config to use native mode (we're on B200!)
    import yaml
    full_config_path = f"/app/{config_path}"
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure native mode for B200
    config['hardware']['hardware_mode'] = 'native'
    config['hardware']['target_sm'] = 100  # Blackwell

    with open(full_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Use run_benchmarks.py which respects config
    cmd = ["python", "run_benchmarks.py", "--config", config_path]

    # Override backend if specified via CLI
    if backend:
        cmd.extend(["--backend", backend])

    # Override sizes if specified
    if sizes:
        cmd.extend(["--sizes", ",".join(map(str, sizes))])

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }


@app.function(
    gpu="B200",
    image=cuda_image,
    timeout=300,
)
def run_sass_analysis():
    """
    Run SASS analysis comparing backends on B200.

    Returns:
        Dict with analysis results
    """
    import subprocess
    import os

    os.chdir("/app")

    cmd = ["python", "compare_sass.py"]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }


@app.local_entrypoint()
def main(
    test_path: str = None,
    benchmark: bool = False,
    sass: bool = False,
    config: str = "config.yaml",
    verbose: bool = True
):
    """
    Main entrypoint for Modal CLI.

    Args:
        test_path: Specific test file to run
        benchmark: Run benchmarks instead of tests
        sass: Run SASS analysis
        config: Path to config file (for benchmarks)
        verbose: Verbose output
    """
    print("=" * 60)
    print("GPU Kernel Lab - Running on Modal B200")
    print("=" * 60)

    if sass:
        print("\n📊 Running SASS analysis...")
        result = run_sass_analysis.remote()
    elif benchmark:
        print(f"\n⚡ Running benchmarks with config: {config}")
        result = run_benchmarks.remote(config_path=config)
    else:
        print(f"\n🧪 Running tests: {test_path or 'all tests'}")
        result = run_tests.remote(test_path=test_path, verbose=verbose)

    print("\n" + "=" * 60)
    if result["success"]:
        print("✅ Completed successfully!")
    else:
        print("❌ Failed!")
        print(f"Return code: {result['returncode']}")

    return result


# Alternative: Interactive GPU session
@app.function(
    gpu="B200",
    image=cuda_image,
    timeout=3600,  # 1 hour
)
def interactive_session():
    """
    Start an interactive session for debugging.
    Use with: modal shell run_modal.py::interactive_session
    """
    import os
    os.chdir("/app")

    print("Interactive session started!")
    print(f"Working directory: {os.getcwd()}")
    print(f"GPU available: checking...")

    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

    # Return shell context
    return {"status": "ready", "cwd": os.getcwd()}
