#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test runner script that reads config.yaml and runs pytest for selected backends.

Usage:
    python run_tests.py                           # Run all tests based on config
    python run_tests.py --config custom.yaml      # Use custom config
    python run_tests.py --kernel add --backend triton  # Override config
    python run_tests.py --compile-only            # Only run compile tests
"""

import argparse
import sys
import os
import yaml
import subprocess
from typing import List, Optional


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


def run_tests(
    config_path: str = 'config.yaml',
    kernel: Optional[str] = None,
    backends: Optional[List[str]] = None,
    compile_only: bool = False,
    verbose: bool = True,
    extra_pytest_args: Optional[List[str]] = None
) -> int:
    """
    Run tests based on configuration.
    
    Args:
        config_path: Path to config file
        kernel: Kernel name override
        backends: Backend list override
        compile_only: Only run compile tests
        verbose: Enable verbose output
        extra_pytest_args: Additional pytest arguments
        
    Returns:
        Exit code (0 for success)
    """
    config = load_config(config_path)
    
    # Get test configuration
    test_config = config.get('tests', {})
    if not test_config.get('enabled', True):
        print("Tests disabled in config")
        return 0
    
    # Determine kernel and backends
    kernel = kernel or 'add'
    backends = backends or test_config.get('backends', ['triton', 'cutile', 'pytorch', 'cublas'])
    
    hardware_mode = config.get('hardware', {}).get('hardware_mode', 'native')
    
    print(f"Running tests in {hardware_mode} mode")
    print(f"Kernel: {kernel}")
    print(f"Backends: {backends}")
    print("-" * 50)
    
    # Build pytest command - use the same Python interpreter as this script
    pytest_args = [sys.executable, '-m', 'pytest']
    
    # Add test files
    test_files = get_test_files(kernel, backends)
    if not test_files:
        print("No test files found!")
        return 1
    
    pytest_args.extend(test_files)
    
    # Add options
    if verbose:
        pytest_args.append('-v')
    
    if compile_only:
        pytest_args.extend(['-k', 'compile'])
    
    if extra_pytest_args:
        pytest_args.extend(extra_pytest_args)
    
    print(f"Running: {' '.join(pytest_args)}")
    print("-" * 50)
    
    # Run pytest
    result = subprocess.run(pytest_args, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU kernel tests based on configuration'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--kernel', '-k',
        default=None,
        help='Kernel name to test (default: add)'
    )
    parser.add_argument(
        '--backend', '-b',
        action='append',
        dest='backends',
        help='Backend(s) to test (can be specified multiple times)'
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
        kernel=args.kernel,
        backends=args.backends,
        compile_only=args.compile_only,
        verbose=not args.quiet,
        extra_pytest_args=args.pytest_args if args.pytest_args else None
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
