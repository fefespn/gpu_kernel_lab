#!/usr/bin/env python3
"""
Test cuda-tile installation and basic functionality on Modal.
Run with: modal run modal_test_cutile.py
"""

import modal

app = modal.App("cutile-test")

# Image with cuda-tile - using CUDA 13.1 for Blackwell support
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
    .pip_install(
        "cuda-tile",
        "cupy-cuda12x>=13.0.0",
        "numpy>=1.24.0",
    )
)


@app.function(gpu="B200", image=cuda_image, timeout=300)
def test_cutile():
    """Test cuda-tile import and basic kernel."""
    import subprocess
    
    print("=" * 60)
    print("CUDA-TILE COMPATIBILITY TEST")
    print("=" * 60)
    
    # Check driver and GPU first
    print("\n📊 GPU Info:")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", 
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(result.stdout)
    
    # Try to import cuda.tile
    print("\n🔧 Testing cuda.tile import...")
    try:
        import cuda.tile as ct
        print("✅ cuda.tile imported successfully!")
        print(f"   Module: {ct}")
        
        # Try to get arch
        try:
            import cuda.tile._compile as ct_compile
            arch = ct_compile.get_sm_arch()
            print(f"   Detected SM arch: {arch}")
        except Exception as e:
            print(f"   Could not get SM arch: {e}")
        
    except Exception as e:
        print(f"❌ Failed to import cuda.tile: {e}")
        return {"status": "import_failed", "error": str(e)}
    
    # Try a simple kernel
    print("\n🧪 Testing simple cuTile kernel...")
    try:
        import cupy as cp
        
        @ct.kernel
        def add_kernel(a, b, c, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
            b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
            result = a_tile + b_tile
            ct.store(c, index=(pid,), tile=result)
        
        print("   Kernel defined ✅")
        
        # Create test arrays
        size = 1024
        tile_size = 16
        a = cp.ones(size, dtype=cp.float32)
        b = cp.ones(size, dtype=cp.float32)
        c = cp.zeros(size, dtype=cp.float32)
        
        print(f"   Test arrays created (size={size})")
        
        # Try to launch
        grid = (ct.cdiv(size, tile_size), 1, 1)
        ct.launch(
            cp.cuda.get_current_stream(),
            grid,
            add_kernel,
            (a, b, c, tile_size)
        )
        cp.cuda.Stream.null.synchronize()
        
        # Check result
        expected = a + b
        if cp.allclose(c, expected):
            print("   ✅ Kernel executed correctly! c = a + b verified")
            return {"status": "success", "result": "all_tests_passed"}
        else:
            print("   ❌ Kernel result incorrect")
            return {"status": "failed", "error": "incorrect_result"}
            
    except Exception as e:
        print(f"   ❌ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "kernel_failed", "error": str(e)}


@app.local_entrypoint()
def main():
    print("Testing cuda-tile on Modal B200...")
    result = test_cutile.remote()
    
    print("\n" + "=" * 60)
    print("RESULT:", result['status'].upper())
    if result['status'] == 'success':
        print("🎉 cuda-tile works on Modal B200!")
    else:
        print(f"⚠️  Error: {result.get('error', 'unknown')}")
