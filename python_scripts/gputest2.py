#!/usr/bin/env python3
"""
GPU Package Verification Script
Checks if all GPU-accelerated packages are properly installed
"""

import sys
import subprocess
import os

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def check_nvidia_gpu():
    """Check NVIDIA GPU and driver"""
    print("\n" + "="*60)
    print("CHECKING NVIDIA GPU AND DRIVERS")
    print("="*60)
    
    # Check nvidia-smi
    stdout, stderr, code = run_command("nvidia-smi")
    if code == 0:
        print("✓ NVIDIA GPU detected")
        print(stdout)
    else:
        print("✗ NVIDIA GPU not detected or drivers not installed")
        print("  Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        return False
    
    # Check CUDA version
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        print("✓ CUDA compiler detected")
        print(stdout)
    else:
        print("✗ CUDA toolkit not installed")
        print("  Install from: https://developer.nvidia.com/cuda-downloads")
    
    return True

def check_python_packages():
    """Check Python GPU packages"""
    print("\n" + "="*60)
    print("CHECKING PYTHON GPU PACKAGES")
    print("="*60)
    
    packages_status = {}
    
    # Check PyTorch with CUDA
    print("\n1. PyTorch CUDA:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"  CUDA available: {cuda_available}")
        print(f"  CUDA version: {cuda_version}")
        print(f"  GPU count: {device_count}")
        
        if cuda_available:
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        
        packages_status['pytorch'] = cuda_available
        
        if not cuda_available:
            print("\n  ⚠ PyTorch is CPU-only! Install CUDA version:")
            print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        packages_status['pytorch'] = False
    
    # Check CuPy
    print("\n2. CuPy:")
    try:
        import cupy as cp
        print(f"✓ CuPy installed: {cp.__version__}")
        
        # Test CuPy functionality
        try:
            test_array = cp.array([1, 2, 3])
            result = cp.sum(test_array)
            print(f"  CuPy test successful: sum([1,2,3]) = {result}")
            packages_status['cupy'] = True
        except Exception as e:
            print(f"  ✗ CuPy test failed: {e}")
            packages_status['cupy'] = False
            
    except ImportError:
        print("✗ CuPy not installed")
        print("  Detect CUDA version first:")
        stdout, _, _ = run_command("nvcc --version | grep release")
        print(f"  {stdout}")
        print("  Then install matching CuPy:")
        print("  CUDA 11.x: pip install cupy-cuda11x")
        print("  CUDA 12.x: pip install cupy-cuda12x")
        packages_status['cupy'] = False
    
    # Check OpenCV with CUDA
    print("\n3. OpenCV CUDA:")
    try:
        import cv2
        print(f"✓ OpenCV installed: {cv2.__version__}")
        
        # Check CUDA support
        cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"  CUDA devices: {cuda_enabled}")
        
        if cuda_enabled > 0:
            print("  ✓ OpenCV compiled with CUDA support")
            packages_status['opencv_cuda'] = True
        else:
            print("  ✗ OpenCV without CUDA support")
            print("  Rebuild OpenCV with CUDA or install:")
            print("  pip uninstall opencv-python opencv-contrib-python")
            print("  pip install opencv-contrib-python-headless")
            print("  (Note: Full CUDA support requires building from source)")
            packages_status['opencv_cuda'] = False
            
    except ImportError:
        print("✗ OpenCV not installed")
        packages_status['opencv_cuda'] = False
    
    # Check video decoding packages
    print("\n4. GPU Video Decoders:")
    
    # Decord
    try:
        from decord import VideoReader, gpu
        print("✓ Decord installed")
        
        # Test GPU context
        try:
            ctx = gpu(0)
            print("  ✓ Decord GPU context available")
            packages_status['decord'] = True
        except:
            print("  ✗ Decord GPU context failed")
            packages_status['decord'] = False
            
    except ImportError:
        print("✗ Decord not installed")
        print("  Install: pip install decord")
        packages_status['decord'] = False
    
    # PyAV (for video processing)
    try:
        import av
        print(f"✓ PyAV installed: {av.__version__}")
        packages_status['pyav'] = True
    except ImportError:
        print("✗ PyAV not installed")
        print("  Install: pip install av")
        packages_status['pyav'] = False
    
    # Check NVIDIA ML Python bindings
    print("\n5. NVIDIA Monitoring:")
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        print("✓ nvidia-ml-py3 installed")
        device_count = nvml.nvmlDeviceGetCount()
        print(f"  Devices: {device_count}")
        nvml.nvmlShutdown()
        packages_status['nvml'] = True
    except ImportError:
        print("✗ nvidia-ml-py3 not installed")
        print("  Install: pip install nvidia-ml-py3")
        packages_status['nvml'] = False
    except Exception as e:
        print(f"✗ NVML initialization failed: {e}")
        packages_status['nvml'] = False
    
    # Check RAPIDS (optional but useful)
    print("\n6. RAPIDS cuDF (optional):")
    try:
        import cudf
        print(f"✓ cuDF installed: {cudf.__version__}")
        packages_status['cudf'] = True
    except ImportError:
        print("✗ cuDF not installed (optional)")
        print("  For data processing on GPU: https://rapids.ai/start.html")
        packages_status['cudf'] = False
    
    return packages_status

def check_gpu_compute_capability():
    """Check GPU compute capability"""
    print("\n" + "="*60)
    print("GPU COMPUTE CAPABILITY")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"  Multi-processors: {props.multi_processor_count}")
                
                if props.major < 5:
                    print("  ⚠ Warning: Old GPU, some features may not work")
                elif props.major >= 7:
                    print("  ✓ Modern GPU with good support")
    except:
        pass

def test_gpu_operations():
    """Test basic GPU operations"""
    print("\n" + "="*60)
    print("TESTING GPU OPERATIONS")
    print("="*60)
    
    # Test PyTorch
    print("\n1. PyTorch GPU test:")
    try:
        import torch
        if torch.cuda.is_available():
            # Create tensors
            a = torch.randn(1000, 1000).cuda()
            b = torch.randn(1000, 1000).cuda()
            
            # Warmup
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Time operation
            import time
            start = time.time()
            for _ in range(100):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            gflops = (2 * 1000**3 * 100) / (elapsed * 1e9)
            print(f"  ✓ Matrix multiplication: {gflops:.1f} GFLOPS")
        else:
            print("  ✗ CUDA not available")
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
    
    # Test CuPy
    print("\n2. CuPy GPU test:")
    try:
        import cupy as cp
        
        # Create arrays
        a = cp.random.randn(1000, 1000, dtype=cp.float32)
        b = cp.random.randn(1000, 1000, dtype=cp.float32)
        
        # Time operation
        import time
        start = time.time()
        for _ in range(100):
            c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        
        gflops = (2 * 1000**3 * 100) / (elapsed * 1e9)
        print(f"  ✓ Matrix multiplication: {gflops:.1f} GFLOPS")
        
        # Memory info
        mempool = cp.get_default_memory_pool()
        print(f"  Memory used: {mempool.used_bytes() / 1e6:.1f} MB")
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")

def install_commands():
    """Print installation commands"""
    print("\n" + "="*60)
    print("INSTALLATION COMMANDS")
    print("="*60)
    
    print("\n# 1. Update NVIDIA drivers (Ubuntu/Debian):")
    print("sudo apt update")
    print("sudo apt install nvidia-driver-525  # or latest version")
    print("sudo reboot")
    
    print("\n# 2. Install CUDA Toolkit:")
    print("# Visit: https://developer.nvidia.com/cuda-downloads")
    print("# Or for Ubuntu:")
    print("wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb")
    print("sudo dpkg -i cuda-keyring_1.0-1_all.deb")
    print("sudo apt-get update")
    print("sudo apt-get -y install cuda")
    
    print("\n# 3. Install Python packages:")
    print("# Create fresh environment")
    print("conda create -n gpu-matcher python=3.10")
    print("conda activate gpu-matcher")
    
    print("\n# Install PyTorch with CUDA")
    print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n# Install CuPy (check CUDA version first)")
    print("nvcc --version  # Check CUDA version")
    print("pip install cupy-cuda11x  # For CUDA 11.x")
    print("pip install cupy-cuda12x  # For CUDA 12.x")
    
    print("\n# Install other packages")
    print("pip install opencv-contrib-python")
    print("pip install decord")
    print("pip install gpxpy pandas tqdm")
    print("pip install nvidia-ml-py3")
    print("pip install av")
    
    print("\n# Optional: Build OpenCV with CUDA (for maximum performance)")
    print("# See: https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html")

def main():
    print("GPU PACKAGE VERIFICATION TOOL")
    print("This will check if GPU packages are properly installed")
    
    # Check GPU
    has_gpu = check_nvidia_gpu()
    
    if not has_gpu:
        print("\nNo NVIDIA GPU detected. Exiting.")
        return
    
    # Check packages
    package_status = check_python_packages()
    
    # Check compute capability
    check_gpu_compute_capability()
    
    # Test operations
    test_gpu_operations()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    working = sum(1 for v in package_status.values() if v)
    total = len(package_status)
    
    print(f"\nPackages working: {working}/{total}")
    
    if working < total:
        print("\nMissing packages:")
        for pkg, status in package_status.items():
            if not status:
                print(f"  - {pkg}")
        
        print("\nRun installation commands above to fix issues.")
    else:
        print("\n✓ All packages properly installed!")
        print("You're ready to run the GPU-accelerated matcher.")
    
    # Installation help
    install_commands()

if __name__ == "__main__":
    main()