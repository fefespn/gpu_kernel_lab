#!/usr/bin/env python3
"""
Modal.com deployment for running GPU Kernel Lab tests on B200 GPUs.

Usage:
    # Run tests
    modal run run_modal.py --tests

    # Run benchmarks
    modal run run_modal.py --benchmark

    # Run SASS analysis
    modal run run_modal.py --sass

    # With custom config
    modal run run_modal.py --tests --config custom.yaml

"""

import modal
import os

# Create the Modal app
app = modal.App("gpu-kernel-lab-same-tile-strategy")

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
def run_tests(config_path: str = "config.yaml"):
    """
    Run GPU kernel tests on B200 for all enabled kernels in config.

    Args:
        config_path: Path to config file (default: config.yaml)

    Returns:
        Dict with test results
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

    # Use run_tests.py which runs all enabled kernels from config
    cmd = ["python", "run_tests.py", "--config", config_path]

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
    timeout=1800,  # 30 minutes for larger benchmark suites
)
def run_benchmarks(config_path: str = "config.yaml"):
    """
    Run GPU kernel benchmarks on B200 for all enabled kernels in config.

    Args:
        config_path: Path to config file (default: config.yaml)

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

    # Use run_benchmarks.py which runs all enabled kernels from config
    cmd = ["python", "run_benchmarks.py", "--config", config_path]

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
    tests: bool = False,
    benchmark: bool = False,
    sass: bool = False,
    config: str = "config.yaml"
):
    """
    Main entrypoint for Modal CLI.

    Args:
        tests: Run tests (all enabled kernels from config)
        benchmark: Run benchmarks (all enabled kernels from config)
        sass: Run SASS analysis
        config: Path to config file (controls which kernels are enabled)
    """
    print("=" * 60)
    print("GPU Kernel Lab - Running on Modal B200")
    print("=" * 60)

    if sass:
        print("\n📊 Running SASS analysis...")
        result = run_sass_analysis.remote()
    elif benchmark:
        print(f"\n⚡ Running benchmarks (all enabled kernels from {config})")
        result = run_benchmarks.remote(config_path=config)
    elif tests:
        print(f"\n🧪 Running tests (all enabled kernels from {config})")
        result = run_tests.remote(config_path=config)
    else:
        print("\n❌ Please specify --tests, --benchmark, or --sass")
        print("\nUsage:")
        print("  modal run run_modal.py --tests              # Run tests")
        print("  modal run run_modal.py --benchmark          # Run benchmarks")
        print("  modal run run_modal.py --sass               # Run SASS analysis")
        print("  modal run run_modal.py --tests --config custom.yaml")
        return {"success": False, "returncode": 1, "stdout": "", "stderr": "No action specified"}

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
