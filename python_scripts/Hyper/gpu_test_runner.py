#!/usr/bin/env python3
"""
GPU Test Runner
===============

This script helps run the GPU tests with proper setup and validation.
It ensures all dependencies are available and provides clear feedback.

Usage:
    python run_gpu_tests.py [--gpu_ids 0 1] [--strict] [--setup]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import importlib.util

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = {
        'torch': 'PyTorch',
        'cupy': 'CuPy', 
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'gpxpy': 'GPXPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'aiofiles': 'aiofiles'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing")
            missing_packages.append(package)
    
    optional_packages = {
        'av': 'PyAV (for video processing)',
        'pynvml': 'pynvml (for GPU monitoring)'
    }
    
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️ {name} missing (optional)")
    
    return missing_packages

def check_gpu_availability():
    """Check if GPU acceleration is available"""
    try:
        import torch
        import cupy as cp
        
        print("\nGPU Availability Check:")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
        
        print(f"CuPy CUDA available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"CuPy device count: {cp.cuda.runtime.getDeviceCount()}")
        
        return torch.cuda.is_available() and cp.cuda.is_available()
        
    except ImportError as e:
        print(f"❌ Cannot check GPU availability: {e}")
        return False

def check_ffmpeg():
    """Check FFmpeg GPU support"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ FFmpeg available")
        
        # Check for GPU support
        hwaccel_result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                      capture_output=True, text=True, check=True)
        if 'cuda' in hwaccel_result.stdout:
            print(f"✅ FFmpeg CUDA support available")
            return True
        else:
            print(f"⚠️ FFmpeg CUDA support not found")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"❌ FFmpeg not available")
        return False

def create_gpu_matcher_file():
    """Ensure the GPU matcher file exists in the current directory"""
    current_dir = Path.cwd()
    matcher_file = current_dir / 'gpu_optimized_matcher.py'
    
    if not matcher_file.exists():
        print(f"⚠️ GPU matcher file not found at {matcher_file}")
        print("Please ensure gpu_optimized_matcher.py is in the current directory")
        return False
    
    print(f"✅ GPU matcher file found: {matcher_file}")
    return True

def run_basic_gpu_test():
    """Run a basic GPU test to verify functionality"""
    try:
        import torch
        import cupy as cp
        
        print("\nRunning basic GPU test...")
        
        # Test PyTorch
        device = torch.device('cuda:0')
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.mm(test_tensor, test_tensor)
        
        if result.device.type != 'cuda':
            raise RuntimeError("PyTorch GPU test failed!")
        
        print("✅ PyTorch GPU test passed")
        
        # Test CuPy
        a = cp.random.randn(100, 100)
        b = cp.random.randn(100, 100)
        c = cp.dot(a, b)
        
        if not isinstance(c, cp.ndarray):
            raise RuntimeError("CuPy GPU test failed!")
        
        print("✅ CuPy GPU test passed")
        
        # Cleanup
        del test_tensor, result, a, b, c
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        
        return True
        
    except Exception as e:
        print(f"❌ Basic GPU test failed: {e}")
        return False

def install_missing_packages(packages):
    """Install missing packages"""
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def main():
    parser = argparse.ArgumentParser(description="GPU Test Runner")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], 
                       help="GPU IDs to test")
    parser.add_argument("--strict", action='store_true', 
                       help="Strict mode - fail on any GPU fallback")
    parser.add_argument("--setup", action='store_true',
                       help="Setup mode - install missing dependencies")
    parser.add_argument("--basic-test", action='store_true',
                       help="Run basic GPU test only")
    
    args = parser.parse_args()
    
    print("GPU Test Runner")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        if args.setup:
            print(f"\nInstalling missing packages: {missing_packages}")
            install_missing_packages(missing_packages)
        else:
            print(f"\n❌ Missing required packages: {missing_packages}")
            print("Run with --setup to install them automatically")
            return 1
    
    # Check GPU availability
    if not check_gpu_availability():
        print("\n❌ GPU acceleration not available!")
        return 1
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        print("⚠️ FFmpeg GPU support issues detected")
    
    # Check for matcher file
    if not create_gpu_matcher_file():
        return 1
    
    # Run basic test if requested
    if args.basic_test:
        if run_basic_gpu_test():
            print("\n✅ Basic GPU test passed!")
            return 0
        else:
            print("\n❌ Basic GPU test failed!")
            return 1
    
    # Run comprehensive tests
    print("\nRunning comprehensive GPU tests...")
    
    # Construct command
    test_cmd = [sys.executable, 'gpu_test.py']
    
    if args.gpu_ids:
        test_cmd.extend(['--gpu_ids'] + [str(gpu_id) for gpu_id in args.gpu_ids])
    
    if args.strict:
        test_cmd.append('--strict')
    
    # Run the test
    try:
        result = subprocess.run(test_cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Failed to run GPU tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())