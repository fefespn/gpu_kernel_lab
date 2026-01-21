#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test runner script that reads config.yaml and runs pytest for all enabled kernels.

Usage:
    python run_tests.py                           # Run all enabled tests from config
    python run_tests.py --config custom.yaml      # Use custom config
    python run_tests.py --backend triton          # Override backends
    python run_tests.py --compile-only            # Only run compile tests
"""

import argparse
import sys
import os
import yaml
import subprocess
from typing import List, Optional, Dict, Any


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_test_files(
    kernel: str,
    backends: List[str],
    tests_dir: str = 'tests'
) -> List[str]:
    """Get list of test files to run."""
    test_files = []
    
    for backend in backends:
        test_file = os.path.join(tests_dir, f"test_{kernel}", f"test_{backend}_{kernel}.py")
        if os.path.exists(test_file):
            test_files.append(test_file)
        else:
            print(f"Warning: Test file not found: {test_file}")
    
    return test_files


def get_enabled_test_configs(config: Dict[str, Any]) -> List[tuple]:
    """
    Get all enabled test configurations from config.
    
    Returns:
        List of (kernel_name, test_config) tuples for enabled kernels
    """
    enabled = []
    
    # Check tests (add kernel)
    tests_config = config.get('tests', {})
    if tests_config.get('enabled', True):
        enabled.append(('add', tests_config))
    
    # Check tests_matmul
    tests_matmul_config = config.get('tests_matmul', {})
    if tests_matmul_config.get('enabled', True):
        enabled.append(('matmul', tests_matmul_config))
    
    return enabled


def run_kernel_tests(
    kernel: str,
    test_config: Dict[str, Any],
    hardware_mode: str,
    backends_override: Optional[List[str]] = None,
    compile_only: bool = False,
    verbose: bool = True,
    extra_pytest_args: Optional[List[str]] = None
) -> int:
    """
    Run tests for a single kernel.
    
    Returns:
        Exit code (0 for success)
    """
    # Determine backends from config or override
    backends = backends_override or test_config.get('backends', ['triton', 'cutile', 'pytorch'])
    
    print(f"\n{'='*60}")
    print(f"Running {kernel.upper()} tests in {hardware_mode} mode")
    print(f"Backends: {backends}")
    print(f"{'='*60}")
    
    # Build pytest command - use the same Python interpreter as this script
    pytest_args = [sys.executable, '-m', 'pytest']
    
    # Add test files
    test_files = get_test_files(kernel, backends)
    if not test_files:
        print(f"No test files found for {kernel}!")
        return 1
    
    pytest_args.extend(test_files)
    
    # Add options
    if verbose:
        pytest_args.append('-v')
    
    if compile_only:
        pytest_args.extend(['-k', 'compile'])
    else:
        # Check for name filter in config (like pytest -k)
        name_filter = test_config.get('name_filter')
        if name_filter:
            pytest_args.extend(['-k', name_filter])
    
    if extra_pytest_args:
        pytest_args.extend(extra_pytest_args)
    
    print(f"Running: {' '.join(pytest_args)}")
    print("-" * 50)
    
    # Run pytest
    result = subprocess.run(pytest_args, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    return result.returncode


def run_tests(
    config_path: str = 'config.yaml',
    backends: Optional[List[str]] = None,
    compile_only: bool = False,
    verbose: bool = True,
    extra_pytest_args: Optional[List[str]] = None
) -> int:
    """
    Run tests for all enabled kernels based on configuration.
    
    Args:
        config_path: Path to config file
        backends: Backend list override (applies to all kernels)
        compile_only: Only run compile tests
        verbose: Enable verbose output
        extra_pytest_args: Additional pytest arguments
        
    Returns:
        Exit code (0 for success, non-zero if any kernel failed)
    """
    config = load_config(config_path)
    hardware_mode = config.get('hardware', {}).get('hardware_mode', 'native')
    
    # Get all enabled test configs
    enabled_configs = get_enabled_test_configs(config)
    
    if not enabled_configs:
        print("No kernels enabled for testing in config")
        return 0
    
    print(f"Enabled kernels: {[k for k, _ in enabled_configs]}")
    
    # Run tests for each enabled kernel
    exit_codes = []
    for kernel, test_config in enabled_configs:
        exit_code = run_kernel_tests(
            kernel=kernel,
            test_config=test_config,
            hardware_mode=hardware_mode,
            backends_override=backends,
            compile_only=compile_only,
            verbose=verbose,
            extra_pytest_args=extra_pytest_args
        )
        exit_codes.append(exit_code)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for (kernel, _), code in zip(enabled_configs, exit_codes):
        status = "✅ PASSED" if code == 0 else "❌ FAILED"
        print(f"  {kernel}: {status}")
    print(f"{'='*60}")
    
    # Return 0 only if all tests passed
    return 0 if all(c == 0 for c in exit_codes) else 1


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU kernel tests for all enabled kernels in config'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--backend', '-b',
        action='append',
        dest='backends',
        help='Backend(s) to test (can be specified multiple times, overrides config)'
    )
    parser.add_argument(
        '--compile-only',
        action='store_true',
        help='Only run compilation tests (skip execution tests)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (less verbose)'
    )
    parser.add_argument(
        'pytest_args',
        nargs='*',
        help='Additional arguments to pass to pytest'
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        config_path=args.config,
        backends=args.backends,
        compile_only=args.compile_only,
        verbose=not args.quiet,
        extra_pytest_args=args.pytest_args if args.pytest_args else None
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
