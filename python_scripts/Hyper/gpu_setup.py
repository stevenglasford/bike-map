#!/usr/bin/env python3
"""
GPU Environment Setup Script
============================

This script sets up the environment for GPU-accelerated video-GPX correlation.
It installs dependencies, checks GPU support, and validates the setup.

Usage:
    python setup_gpu_environment.py [--cuda-version 11.8] [--force]
"""

import sys
import os
import subprocess
import argparse
import platform
from pathlib import Path
import urllib.request
import json

def get_system_info():
    """Get system information"""
    info = {
        'os': platform.system(),
        'arch': platform.machine(),
        'python_version': sys.version,
        'platform': platform.platform()
    }
    
    print("System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return info

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("✅ NVIDIA driver installed")
        
        # Extract driver version
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version:' in line:
                driver_version = line.split('Driver Version:')[1].split()[0]
                print(f"  Driver Version: {driver_version}")
                break
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NVIDIA driver not found or nvidia-smi not available")
        print("Please install NVIDIA drivers first:")
        print("  Ubuntu/Debian: sudo apt install nvidia-driver-xxx")
        print("  Or download from: https://www.nvidia.com/drivers")
        return False

def check_cuda_installation():
    """Check CUDA installation"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        print("✅ CUDA toolkit installed")
        
        # Extract CUDA version
        lines = result.stdout.split('\n')
        for line in lines:
            if 'release' in line:
                cuda_version = line.split('release')[1].split(',')[0].strip()
                print(f"  CUDA Version: {cuda_version}")
                break
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ CUDA toolkit not found")
        print("Please install CUDA toolkit:")
        print("  Download from: https://developer.nvidia.com/cuda-downloads")
        return False

def install_pytorch_gpu(cuda_version="11.8"):
    """Install PyTorch with GPU support"""
    print(f"\nInstalling PyTorch with CUDA {cuda_version}...")
    
    # Determine appropriate PyTorch command
    if cuda_version.startswith("11.8"):
        torch_cmd = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif cuda_version.startswith("12.1"):
        torch_cmd = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        torch_cmd = "torch torchvision torchaudio"  # CPU version as fallback
    
    cmd = [sys.executable, '-m', 'pip', 'install'] + torch_cmd.split()
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch")
        return False

def install_cupy(cuda_version="11.8"):
    """Install CuPy with appropriate CUDA version"""
    print(f"\nInstalling CuPy for CUDA {cuda_version}...")
    
    # Determine CuPy package name
    if cuda_version.startswith("11.8"):
        cupy_package = "cupy-cuda11x"
    elif cuda_version.startswith("12"):
        cupy_package = "cupy-cuda12x"
    else:
        cupy_package = "cupy-cuda11x"  # Default fallback
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', cupy_package], check=True)
        print("✅ CuPy installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install CuPy")
        return False

def install_core_dependencies():
    """Install core dependencies"""
    print("\nInstalling core dependencies...")
    
    core_packages = [
        'numpy',
        'pandas', 
        'opencv-python',
        'scipy',
        'gpxpy',
        'tqdm',
        'aiofiles',
        'asyncio'
    ]
    
    for package in core_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def install_optional_dependencies():
    """Install optional dependencies"""
    print("\nInstalling optional dependencies...")
    
    optional_packages = [
        'av',  # PyAV for video processing
        'pynvml',  # NVIDIA management library
        'psutil',  # System monitoring
        'matplotlib',  # Plotting
        'plotly',  # Interactive plots
        'jupyter'  # Jupyter notebooks
    ]
    
    for package in optional_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install optional package {package}")

def check_ffmpeg_installation():
    """Check and suggest FFmpeg installation"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg installed")
        
        # Check for GPU support
        result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True, check=True)
        if 'cuda' in result.stdout:
            print("✅ FFmpeg CUDA support available")
        else:
            print("⚠️ FFmpeg CUDA support not detected")
            print("Consider installing FFmpeg with CUDA support:")
            print("  Ubuntu: sudo apt install ffmpeg")
            print("  Or build from source with --enable-cuda-nvcc")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found")
        print("Please install FFmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  macOS: brew install ffmpeg")
        return False

def validate_installation():
    """Validate the complete installation"""
    print("\nValidating installation...")
    
    validation_results = {}
    
    # Test PyTorch GPU
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
        validation_results['pytorch'] = torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch not properly installed")
        validation_results['pytorch'] = False
    
    # Test CuPy
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        print(f"CuPy CUDA available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"CuPy devices: {cp.cuda.runtime.getDeviceCount()}")
        validation_results['cupy'] = cp.cuda.is_available()
    except ImportError:
        print("❌ CuPy not properly installed")
        validation_results['cupy'] = False
    
    # Test basic operations
    if validation_results.get('pytorch') and validation_results.get('cupy'):
        try:
            # PyTorch test
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.mm(test_tensor, test_tensor)
            
            # CuPy test
            a = cp.random.randn(100, 100)
            b = cp.random.randn(100, 100) 
            c = cp.dot(a, b)
            
            print("✅ GPU operations test passed")
            validation_results['gpu_operations'] = True
            
            # Cleanup
            del test_tensor, result, a, b, c
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"❌ GPU operations test failed: {e}")
            validation_results['gpu_operations'] = False
    
    return validation_results

def create_test_script():
    """Create a simple test script"""
    test_script_content = '''#!/usr/bin/env python3
"""
Simple GPU Test Script
======================
Tests basic GPU functionality for the video-GPX correlation system.
"""

import torch
import cupy as cp
import time

def test_gpu():
    print("Testing GPU acceleration...")
    
    if not torch.cuda.is_available():
        print("❌ PyTorch CUDA not available")
        return False
    
    if not cp.cuda.is_available():
        print("❌ CuPy CUDA not available") 
        return False
    
    # Test PyTorch
    device = torch.device('cuda:0')
    print(f"Testing on {device}")
    
    # Matrix multiplication test
    size = 2000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start_time = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"PyTorch GPU matrix multiplication ({size}x{size}): {gpu_time:.3f}s")
    
    # Test CuPy
    a_cp = cp.random.randn(size, size)
    b_cp = cp.random.randn(size, size)
    
    start_time = time.time()
    c_cp = cp.dot(a_cp, b_cp)
    cp.cuda.Stream.null.synchronize()
    cupy_time = time.time() - start_time
    
    print(f"CuPy GPU matrix multiplication ({size}x{size}): {cupy_time:.3f}s")
    
    # CPU comparison
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU matrix multiplication ({size}x{size}): {cpu_time:.3f}s")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
    
    # Cleanup
    del a, b, c, a_cp, b_cp, c_cp, a_cpu, b_cpu, c_cpu
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    
    print("✅ GPU test completed successfully")
    return True

if __name__ == "__main__":
    test_gpu()
'''
    
    with open('simple_gpu_test.py', 'w') as f:
        f.write(test_script_content)
    
    print("✅ Created simple_gpu_test.py")

def main():
    parser = argparse.ArgumentParser(description="Setup GPU environment for video-GPX correlation")
    parser.add_argument("--cuda-version", default="11.8", help="CUDA version to target")
    parser.add_argument("--force", action='store_true', help="Force reinstallation")
    parser.add_argument("--skip-nvidia-check", action='store_true', help="Skip NVIDIA driver check")
    parser.add_argument("--skip-cuda-check", action='store_true', help="Skip CUDA toolkit check")
    
    args = parser.parse_args()
    
    print("GPU Environment Setup")
    print("=" * 50)
    
    # Get system info
    system_info = get_system_info()
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    if not args.skip_nvidia_check:
        nvidia_ok = check_nvidia_driver()
        if not nvidia_ok:
            print("⚠️ Continuing without NVIDIA driver verification")
    
    if not args.skip_cuda_check:
        cuda_ok = check_cuda_installation()
        if not cuda_ok:
            print("⚠️ Continuing without CUDA toolkit verification")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    # Core dependencies first
    install_core_dependencies()
    
    # GPU-specific packages
    install_pytorch_gpu(args.cuda_version)
    install_cupy(args.cuda_version)
    
    # Optional packages
    install_optional_dependencies()
    
    # Check FFmpeg
    check_ffmpeg_installation()
    
    # Validate installation
    validation_results = validate_installation()
    
    # Create test script
    create_test_script()
    
    # Summary
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    if all(validation_results.values()):
        print("✅ GPU environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python simple_gpu_test.py")
        print("2. Run: python run_gpu_tests.py --basic-test")
        print("3. Run: python run_gpu_tests.py --strict")
        print("4. Run: python gpu_optimized_matcher.py --help")
        return 0
    else:
        print("⚠️ Setup completed with some issues:")
        for component, status in validation_results.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
        
        print("\nYou may need to:")
        print("- Install NVIDIA drivers")
        print("- Install CUDA toolkit")
        print("- Check Python environment")
        return 1

if __name__ == "__main__":
    sys.exit(main())