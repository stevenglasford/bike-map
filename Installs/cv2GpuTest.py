#!/usr/bin/env python3
import cv2
import numpy as np
import time

def test_opencv_gpu():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Build info file: {cv2.__file__}")
    
    # Check for CUDA modules
    cuda_modules = [x for x in dir(cv2) if 'cuda' in x.lower()]
    print(f"CUDA modules found: {len(cuda_modules)}")
    if cuda_modules:
        print("Available CUDA modules:", cuda_modules[:5], "..." if len(cuda_modules) > 5 else "")
    
    # Check CUDA device count
    if hasattr(cv2, 'cuda'):
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA enabled devices: {device_count}")
            
            if device_count > 0:
                # Test GPU operations
                print("\n=== Testing GPU Operations ===")
                
                # Create test image
                cpu_img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
                print(f"Test image shape: {cpu_img.shape}")
                
                # CPU processing time
                start_time = time.time()
                cpu_blur = cv2.GaussianBlur(cpu_img, (15, 15), 0)
                cpu_time = time.time() - start_time
                print(f"CPU processing time: {cpu_time:.4f} seconds")
                
                # GPU processing time
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_result = cv2.cuda_GpuMat()
                    
                    start_time = time.time()
                    gpu_img.upload(cpu_img)
                    cv2.cuda.bilateralFilter(gpu_img, -1, 15, 15, gpu_result)
                    result = gpu_result.download()
                    gpu_time = time.time() - start_time
                    
                    print(f"GPU processing time: {gpu_time:.4f} seconds")
                    print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
                    print("✅ GPU acceleration is working!")
                    
                except Exception as e:
                    print(f"❌ GPU operation failed: {e}")
            else:
                print("❌ No CUDA-capable devices found")
                
        except Exception as e:
            print(f"❌ CUDA check failed: {e}")
    else:
        print("❌ No cv2.cuda module found")
    
    # Check build info for CUDA
    build_info = cv2.getBuildInformation()
    if 'CUDA' in build_info and 'YES' in build_info:
        print("✅ OpenCV built with CUDA support")
    else:
        print("❌ OpenCV not built with CUDA support")

if __name__ == "__main__":
    test_opencv_gpu()