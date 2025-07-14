#!/usr/bin/env python3
"""
GPU 1 DIAGNOSTIC SCRIPT
🔍 ISOLATE EXACTLY WHAT'S WRONG WITH GPU 1 🔍
"""

import cupy as cp
import torch
import numpy as np
import time
import multiprocessing as mp
import threading
import sys

def test_gpu_basic(gpu_id):
    """Basic GPU test"""
    print(f"🔍 Testing GPU {gpu_id} basic functionality...")
    
    try:
        # Set device
        cp.cuda.Device(gpu_id).use()
        current = cp.cuda.Device()
        print(f"✅ GPU {gpu_id}: Device set successfully (current={current.id})")
        
        # Memory info
        free, total = cp.cuda.Device().mem_info
        print(f"✅ GPU {gpu_id}: Memory: {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total")
        
        # Simple computation
        test_array = cp.random.rand(1000, 1000)
        result = cp.sum(test_array)
        print(f"✅ GPU {gpu_id}: Computation test: {float(result):.2f}")
        
        # Memory cleanup
        del test_array
        cp.get_default_memory_pool().free_all_blocks()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Basic test FAILED: {e}")
        return False

def test_gpu_heavy(gpu_id):
    """Heavy GPU test"""
    print(f"🔍 Testing GPU {gpu_id} heavy workload...")
    
    try:
        cp.cuda.Device(gpu_id).use()
        
        # Heavy computation
        for i in range(5):
            large_array = cp.random.rand(5000, 5000)
            result = cp.matmul(large_array, large_array.T)
            final = cp.sum(result)
            print(f"✅ GPU {gpu_id}: Heavy test {i+1}: {float(final):.2e}")
            del large_array, result
            time.sleep(0.5)
        
        cp.get_default_memory_pool().free_all_blocks()
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Heavy test FAILED: {e}")
        return False

def test_gpu_threading(gpu_id):
    """Test GPU in threading"""
    print(f"🔍 Testing GPU {gpu_id} in thread...")
    
    def thread_worker():
        try:
            cp.cuda.Device(gpu_id).use()
            current = cp.cuda.Device()
            
            if current.id != gpu_id:
                print(f"❌ GPU {gpu_id}: Thread context FAILED - got GPU {current.id}")
                return False
            
            # Do work
            for i in range(3):
                data = cp.random.rand(2000, 2000)
                result = cp.sum(data ** 2)
                print(f"✅ GPU {gpu_id}: Thread work {i+1}: {float(result):.2e}")
                del data
                time.sleep(0.2)
            
            cp.get_default_memory_pool().free_all_blocks()
            return True
            
        except Exception as e:
            print(f"❌ GPU {gpu_id}: Threading FAILED: {e}")
            return False
    
    thread = threading.Thread(target=thread_worker)
    thread.start()
    thread.join()

def test_gpu_multiprocessing(gpu_id):
    """Test GPU in multiprocessing"""
    print(f"🔍 Testing GPU {gpu_id} in separate process...")
    
    def process_worker(gpu_id, result_queue):
        try:
            cp.cuda.Device(gpu_id).use()
            current = cp.cuda.Device()
            
            if current.id != gpu_id:
                result_queue.put(f"❌ GPU {gpu_id}: Process context FAILED - got GPU {current.id}")
                return
            
            # Do work
            for i in range(3):
                data = cp.random.rand(3000, 3000)
                result = cp.sum(data ** 3)
                result_queue.put(f"✅ GPU {gpu_id}: Process work {i+1}: {float(result):.2e}")
                del data
                time.sleep(0.2)
            
            cp.get_default_memory_pool().free_all_blocks()
            result_queue.put(f"✅ GPU {gpu_id}: Process test COMPLETE")
            
        except Exception as e:
            result_queue.put(f"❌ GPU {gpu_id}: Multiprocessing FAILED: {e}")
    
    result_queue = mp.Queue()
    process = mp.Process(target=process_worker, args=(gpu_id, result_queue))
    process.start()
    
    # Collect results
    results = []
    while process.is_alive() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1)
            results.append(result)
            print(result)
        except:
            break
    
    process.join()

def test_gpu_pytorch(gpu_id):
    """Test GPU with PyTorch"""
    print(f"🔍 Testing GPU {gpu_id} with PyTorch...")
    
    try:
        device = f'cuda:{gpu_id}'
        
        # Test tensor operations
        tensor = torch.randn(2000, 2000, device=device)
        result = torch.matmul(tensor, tensor.T)
        final = torch.sum(result)
        print(f"✅ GPU {gpu_id}: PyTorch test: {float(final):.2e}")
        
        del tensor, result
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: PyTorch FAILED: {e}")
        return False

def test_gpu_concurrent():
    """Test both GPUs concurrently"""
    print(f"🔍 Testing BOTH GPUs concurrently...")
    
    def gpu_worker(gpu_id, results):
        try:
            cp.cuda.Device(gpu_id).use()
            
            for i in range(5):
                data = cp.random.rand(3000, 3000)
                result = cp.sum(data * data)
                results[gpu_id].append(f"GPU {gpu_id} iteration {i+1}: {float(result):.2e}")
                del data
                time.sleep(0.3)
                
        except Exception as e:
            results[gpu_id].append(f"GPU {gpu_id} ERROR: {e}")
    
    results = {0: [], 1: []}
    
    thread0 = threading.Thread(target=gpu_worker, args=(0, results))
    thread1 = threading.Thread(target=gpu_worker, args=(1, results))
    
    thread0.start()
    thread1.start()
    
    thread0.join()
    thread1.join()
    
    print("🔍 Concurrent results:")
    for gpu_id in [0, 1]:
        print(f"GPU {gpu_id}:")
        for result in results[gpu_id]:
            print(f"  {result}")

def main():
    print("🔍🔍🔍 GPU 1 DIAGNOSTIC SCRIPT 🔍🔍🔍")
    print("====================================")
    print("🎯 Finding exactly what's wrong with GPU 1")
    print("====================================")
    
    # Check basic GPU detection
    print(f"\n🔍 GPU Detection:")
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"✅ CuPy detected {gpu_count} GPUs")
        
        for gpu_id in [0, 1]:
            cp.cuda.Device(gpu_id).use()
            props = cp.cuda.Device().attributes
            print(f"✅ GPU {gpu_id}: Available")
            
    except Exception as e:
        print(f"❌ GPU detection FAILED: {e}")
        return
    
    # Test each GPU individually
    print(f"\n🔍 Individual GPU Tests:")
    print("=" * 40)
    
    for gpu_id in [0, 1]:
        print(f"\n--- Testing GPU {gpu_id} ---")
        
        # Basic test
        basic_ok = test_gpu_basic(gpu_id)
        
        if basic_ok:
            # Heavy test
            heavy_ok = test_gpu_heavy(gpu_id)
            
            # PyTorch test
            pytorch_ok = test_gpu_pytorch(gpu_id)
            
            print(f"GPU {gpu_id} Summary: Basic={basic_ok}, Heavy={heavy_ok}, PyTorch={pytorch_ok}")
        else:
            print(f"GPU {gpu_id} Summary: BASIC TEST FAILED - SKIPPING OTHER TESTS")
    
    # Test threading
    print(f"\n🔍 Threading Tests:")
    print("=" * 40)
    
    for gpu_id in [0, 1]:
        test_gpu_threading(gpu_id)
    
    # Test multiprocessing
    print(f"\n🔍 Multiprocessing Tests:")
    print("=" * 40)
    
    for gpu_id in [0, 1]:
        test_gpu_multiprocessing(gpu_id)
    
    # Test concurrent usage
    print(f"\n🔍 Concurrent GPU Test:")
    print("=" * 40)
    
    test_gpu_concurrent()
    
    print(f"\n🔍🔍🔍 DIAGNOSTIC COMPLETE 🔍🔍🔍")
    print("================================")
    print("📋 SUMMARY:")
    print("Look for patterns in the failures above.")
    print("If GPU 1 consistently fails, it's a hardware/driver issue.")
    print("If GPU 1 works alone but fails in concurrent tests,")
    print("it's a software/context issue.")

if __name__ == "__main__":
    main()