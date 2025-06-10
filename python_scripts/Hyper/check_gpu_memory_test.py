#!/usr/bin/env python3
"""
GPU Memory Checker
==================

Quick script to check GPU memory usage and clear memory if needed.

Usage:
    python check_gpu_memory.py [--clear] [--gpu_ids 0 1]
"""

import torch
import cupy as cp
import argparse
import subprocess
import time

def check_gpu_memory(gpu_ids=None):
    """Check GPU memory usage"""
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    print("GPU Memory Status:")
    print("=" * 50)
    
    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            print(f"GPU {gpu_id}: Not available")
            continue
        
        # PyTorch memory
        props = torch.cuda.get_device_properties(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)
        total = props.total_memory
        
        print(f"GPU {gpu_id} ({props.name}):")
        print(f"  Total Memory: {total // 1024**3:.1f} GB")
        print(f"  PyTorch Allocated: {allocated // 1024**2:.1f} MB")
        print(f"  PyTorch Reserved: {reserved // 1024**2:.1f} MB")
        print(f"  Available: {(total - reserved) // 1024**2:.1f} MB")
        
        # CuPy memory
        try:
            with cp.cuda.Device(gpu_id):
                cupy_used = cp.get_default_memory_pool().used_bytes()
                cupy_total = cp.get_default_memory_pool().total_bytes()
                print(f"  CuPy Used: {cupy_used // 1024**2:.1f} MB")
                print(f"  CuPy Total Pool: {cupy_total // 1024**2:.1f} MB")
        except Exception as e:
            print(f"  CuPy: Error - {e}")
        
        # NVIDIA-SMI info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits', f'--id={gpu_id}'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                used, total = result.stdout.strip().split(', ')
                print(f"  System Used: {used} MB / {total} MB")
                free_pct = (1 - int(used) / int(total)) * 100
                print(f"  Free: {free_pct:.1f}%")
        except:
            print("  System: Unable to query")
        
        print()

def clear_gpu_memory(gpu_ids=None):
    """Clear GPU memory"""
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    print("Clearing GPU Memory...")
    print("=" * 30)
    
    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            continue
        
        try:
            # Clear PyTorch
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            # Clear CuPy
            with cp.cuda.Device(gpu_id):
                cp.get_default_memory_pool().free_all_blocks()
            
            print(f"✅ Cleared GPU {gpu_id}")
            
        except Exception as e:
            print(f"❌ Failed to clear GPU {gpu_id}: {e}")

def test_gpu_allocation(gpu_id=0, size_mb=100):
    """Test GPU memory allocation"""
    print(f"\nTesting {size_mb}MB allocation on GPU {gpu_id}...")
    
    try:
        # Test PyTorch allocation
        device = torch.device(f'cuda:{gpu_id}')
        size_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        test_tensor = torch.randn(size_elements, device=device)
        
        print(f"✅ PyTorch allocation successful")
        
        # Test CuPy allocation
        with cp.cuda.Device(gpu_id):
            test_array = cp.random.randn(size_elements)
            print(f"✅ CuPy allocation successful")
        
        # Cleanup
        del test_tensor, test_array
        torch.cuda.empty_cache()
        with cp.cuda.Device(gpu_id):
            cp.get_default_memory_pool().free_all_blocks()
        
        print(f"✅ Memory test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def monitor_memory(gpu_ids=None, duration=30):
    """Monitor GPU memory usage over time"""
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    print(f"Monitoring GPU memory for {duration} seconds...")
    print("Press Ctrl+C to stop early")
    print()
    
    try:
        for i in range(duration):
            print(f"\rTime: {i+1:3d}s", end="")
            
            for gpu_id in gpu_ids:
                if gpu_id >= torch.cuda.device_count():
                    continue
                
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                                           '--format=csv,noheader,nounits', f'--id={gpu_id}'], 
                                           capture_output=True, text=True)
                    if result.returncode == 0:
                        used = result.stdout.strip()
                        print(f" | GPU{gpu_id}: {used}MB", end="")
                except:
                    print(f" | GPU{gpu_id}: Error", end="")
            
            time.sleep(1)
        
        print("\nMonitoring complete")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="GPU Memory Checker and Manager")
    parser.add_argument("--clear", action='store_true', help="Clear GPU memory")
    parser.add_argument("--test", type=int, metavar='SIZE_MB', help="Test allocation of SIZE_MB")
    parser.add_argument("--monitor", type=int, metavar='SECONDS', help="Monitor memory for SECONDS")
    parser.add_argument("--gpu_ids", nargs='+', type=int, help="GPU IDs to check")
    
    args = parser.parse_args()
    
    # Check basic CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return 1
    
    print(f"Found {torch.cuda.device_count()} CUDA device(s)")
    print()
    
    # Check memory status
    check_gpu_memory(args.gpu_ids)
    
    # Clear memory if requested
    if args.clear:
        clear_gpu_memory(args.gpu_ids)
        print()
        check_gpu_memory(args.gpu_ids)
    
    # Test allocation if requested
    if args.test:
        gpu_id = args.gpu_ids[0] if args.gpu_ids else 0
        test_gpu_allocation(gpu_id, args.test)
    
    # Monitor if requested
    if args.monitor:
        monitor_memory(args.gpu_ids, args.monitor)
    
    return 0

if __name__ == "__main__":
    exit(main())