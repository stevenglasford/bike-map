#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def test_opencv_cuda():
    print("="*50)
    print("OpenCV CUDA Support Test")
    print("="*50)
    
    # Basic OpenCV info
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV build info:")
    print(f"  CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    
    # Check device count
    device_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA-enabled devices detected: {device_count}")
    
    if device_count == 0:
        print("❌ No CUDA devices detected!")
        print("This could mean:")
        print("  - OpenCV wasn't built with CUDA support")
        print("  - No NVIDIA GPUs found")
        print("  - CUDA drivers not installed")
        return False
    
    # Test each device
    for i in range(device_count):
        print(f"\n--- Testing Device {i} ---")
        try:
            # Set device
            cv2.cuda.setDevice(i)
            
            # Get device properties
            device_info = cv2.cuda.DeviceInfo(i)
            print(f"Device {i}: {device_info.name()}")
            print(f"  Compute capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
            print(f"  Total memory: {device_info.totalMemory() / (1024**3):.1f} GB")
            print(f"  Free memory: {device_info.freeMemory() / (1024**3):.1f} GB")
            
            # Test basic GPU operation
            try:
                # Create test matrices
                cpu_mat = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
                
                # Upload to GPU
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(cpu_mat)
                
                # Perform simple operation
                gpu_result = cv2.cuda_GpuMat()
                cv2.cuda.threshold(gpu_mat, gpu_result, 127, 255, cv2.THRESH_BINARY)
                
                # Download result
                cpu_result = gpu_result.download()
                
                print(f"  ✅ Basic GPU operations working")
                
                # Test more advanced operation
                gpu_blur = cv2.cuda_GpuMat()
                cv2.cuda.bilateralFilter(gpu_mat, gpu_blur, -1, 50, 50)
                
                print(f"  ✅ Advanced GPU operations working")
                
            except Exception as e:
                print(f"  ❌ GPU operations failed: {e}")
                print(f"  This usually means GPU memory is full or device is busy")
                return False
                
        except Exception as e:
            print(f"  ❌ Failed to access device {i}: {e}")
            return False
    
    print(f"\n✅ OpenCV CUDA support is working correctly!")
    return True

def check_system_gpu_usage():
    """Check system GPU usage using nvidia-smi if available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*50)
            print("Current GPU Usage (nvidia-smi)")
            print("="*50)
            print("GPU | Name | GPU% | Memory Used/Total")
            print("-" * 50)
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_id, name, util, mem_used, mem_total = parts[:5]
                    print(f"{gpu_id:3} | {name[:20]:20} | {util:4}% | {mem_used:>4}/{mem_total:<4} MB")
        else:
            print("\nnvidia-smi not available")
            
    except FileNotFoundError:
        print("\nnvidia-smi not found in PATH")
    except Exception as e:
        print(f"\nError running nvidia-smi: {e}")

if __name__ == "__main__":
    # Check system GPU usage first
    check_system_gpu_usage()
    
    # Test OpenCV CUDA
    success = test_opencv_cuda()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("  1. Check if other processes are using GPU memory:")
        print("     nvidia-smi")
        print("  2. Try freeing GPU memory:")
        print("     sudo fuser -v /dev/nvidia*")
        print("  3. Restart CUDA processes or reboot if needed")
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed! OpenCV CUDA is ready to use.")

