#!/usr/bin/env python3
"""
GPU Manager Diagnostic Tool
Identifies the root cause of GPU acquisition failures

This will help us understand why GPU acquisition is failing
even with minimal parallelization.
"""

import torch
import threading
import queue
import time
import logging
from typing import List, Optional
import sys
import traceback

# Setup logging for detailed diagnosis
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUDiagnostic:
    """Comprehensive GPU diagnostic tool"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.results = {}
        
    def run_full_diagnostic(self):
        """Run comprehensive GPU diagnostic"""
        print("🔍 GPU DIAGNOSTIC ANALYSIS")
        print("=" * 50)
        
        # Test 1: Basic CUDA availability
        self.test_cuda_basic()
        
        # Test 2: Individual GPU functionality
        self.test_individual_gpus()
        
        # Test 3: Concurrent GPU access
        self.test_concurrent_access()
        
        # Test 4: Queue-based GPU management
        self.test_queue_management()
        
        # Test 5: High-load simulation
        self.test_high_load_simulation()
        
        # Test 6: Memory allocation patterns
        self.test_memory_patterns()
        
        self.print_diagnostic_summary()
        
    def test_cuda_basic(self):
        """Test basic CUDA functionality"""
        print("\n🧪 Test 1: Basic CUDA Functionality")
        print("-" * 30)
        
        try:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            print(f"CUDA Available: {cuda_available}")
            print(f"Device Count: {device_count}")
            
            if cuda_available:
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    print(f"GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
            
            self.results['cuda_basic'] = {
                'available': cuda_available,
                'device_count': device_count,
                'status': 'PASS' if cuda_available and device_count > 0 else 'FAIL'
            }
            
        except Exception as e:
            print(f"❌ Basic CUDA test failed: {e}")
            self.results['cuda_basic'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_individual_gpus(self):
        """Test each GPU individually"""
        print("\n🧪 Test 2: Individual GPU Functionality")
        print("-" * 30)
        
        gpu_results = {}
        
        for gpu_id in self.gpu_ids:
            print(f"\nTesting GPU {gpu_id}...")
            
            try:
                # Test basic tensor operations
                with torch.cuda.device(gpu_id):
                    test_tensor = torch.zeros(1000, 1000, device=f'cuda:{gpu_id}')
                    result = torch.sum(test_tensor).item()
                    
                    # Test memory allocation/deallocation
                    for size in [100, 1000, 5000]:
                        temp = torch.zeros(size, size, device=f'cuda:{gpu_id}')
                        del temp
                    
                    torch.cuda.empty_cache()
                    
                    print(f"✅ GPU {gpu_id}: Basic operations successful")
                    gpu_results[gpu_id] = {'status': 'PASS', 'error': None}
                    
            except Exception as e:
                print(f"❌ GPU {gpu_id}: Failed - {e}")
                gpu_results[gpu_id] = {'status': 'FAIL', 'error': str(e)}
        
        self.results['individual_gpus'] = gpu_results
    
    def test_concurrent_access(self):
        """Test concurrent GPU access patterns"""
        print("\n🧪 Test 3: Concurrent GPU Access")
        print("-" * 30)
        
        def gpu_worker(gpu_id, worker_id, results_dict):
            """Worker function for concurrent GPU access"""
            try:
                thread_name = f"Worker-{worker_id}-GPU-{gpu_id}"
                with torch.cuda.device(gpu_id):
                    # Simulate typical processing
                    tensor = torch.randn(500, 500, device=f'cuda:{gpu_id}')
                    for _ in range(10):
                        tensor = tensor * 1.01
                    result = torch.sum(tensor).item()
                    
                    results_dict[thread_name] = {'status': 'PASS', 'result': result}
                    print(f"✅ {thread_name}: Success")
                    
            except Exception as e:
                results_dict[thread_name] = {'status': 'FAIL', 'error': str(e)}
                print(f"❌ {thread_name}: Failed - {e}")
        
        # Test concurrent access to same GPU
        print("Testing concurrent access to same GPU...")
        concurrent_results = {}
        threads = []
        
        for worker_id in range(3):  # 3 workers on same GPU
            thread = threading.Thread(
                target=gpu_worker, 
                args=(self.gpu_ids[0], worker_id, concurrent_results)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Test access across different GPUs
        if len(self.gpu_ids) > 1:
            print("Testing access across different GPUs...")
            for i, gpu_id in enumerate(self.gpu_ids):
                thread = threading.Thread(
                    target=gpu_worker, 
                    args=(gpu_id, f"multi-{i}", concurrent_results)
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads[-len(self.gpu_ids):]:
                thread.join()
        
        self.results['concurrent_access'] = concurrent_results
    
    def test_queue_management(self):
        """Test queue-based GPU management similar to matcher.py"""
        print("\n🧪 Test 4: Queue-Based GPU Management")
        print("-" * 30)
        
        class TestGPUManager:
            def __init__(self, gpu_ids):
                self.gpu_ids = gpu_ids
                self.gpu_queue = queue.Queue()
                for gpu_id in gpu_ids:
                    self.gpu_queue.put(gpu_id)
                self.gpu_usage = {gpu_id: 0 for gpu_id in gpu_ids}
                print(f"Initialized queue with {len(gpu_ids)} GPUs")
            
            def acquire_gpu(self, timeout=30):
                try:
                    gpu_id = self.gpu_queue.get(timeout=timeout)
                    self.gpu_usage[gpu_id] += 1
                    return gpu_id
                except queue.Empty:
                    return None
            
            def release_gpu(self, gpu_id):
                self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
                self.gpu_queue.put(gpu_id)
        
        def test_worker(manager, worker_id, results_dict):
            """Test worker that uses GPU manager"""
            try:
                # Acquire GPU
                start_time = time.time()
                gpu_id = manager.acquire_gpu(timeout=10)
                acquire_time = time.time() - start_time
                
                if gpu_id is None:
                    results_dict[worker_id] = {
                        'status': 'FAIL', 
                        'error': f'Could not acquire GPU in 10s'
                    }
                    return
                
                # Use GPU
                with torch.cuda.device(gpu_id):
                    tensor = torch.randn(200, 200, device=f'cuda:{gpu_id}')
                    time.sleep(0.5)  # Simulate work
                    result = torch.sum(tensor).item()
                
                # Release GPU
                manager.release_gpu(gpu_id)
                
                results_dict[worker_id] = {
                    'status': 'PASS',
                    'gpu_id': gpu_id,
                    'acquire_time': acquire_time,
                    'result': result
                }
                print(f"✅ Worker {worker_id}: Used GPU {gpu_id} (acquire: {acquire_time:.2f}s)")
                
            except Exception as e:
                results_dict[worker_id] = {'status': 'FAIL', 'error': str(e)}
                print(f"❌ Worker {worker_id}: Failed - {e}")
        
        # Test the queue management system
        manager = TestGPUManager(self.gpu_ids)
        queue_results = {}
        threads = []
        
        # Launch multiple workers
        num_workers = len(self.gpu_ids) * 2  # 2 workers per GPU
        print(f"Launching {num_workers} workers...")
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=test_worker, 
                args=(manager, worker_id, queue_results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all workers
        for thread in threads:
            thread.join()
        
        print(f"Queue test completed. Final GPU usage: {manager.gpu_usage}")
        self.results['queue_management'] = queue_results
    
    def test_high_load_simulation(self):
        """Simulate high load similar to chunked processing"""
        print("\n🧪 Test 5: High Load Simulation")
        print("-" * 30)
        
        def chunk_processor(chunk_id, gpu_queue, results_dict):
            """Simulate chunked processing"""
            try:
                # Acquire GPU (similar to chunked processor)
                start_time = time.time()
                
                try:
                    gpu_id = gpu_queue.get(timeout=5)  # Short timeout for stress test
                except queue.Empty:
                    results_dict[chunk_id] = {
                        'status': 'TIMEOUT',
                        'error': 'Could not acquire GPU in 5s'
                    }
                    return
                
                acquire_time = time.time() - start_time
                
                # Simulate chunk processing
                with torch.cuda.device(gpu_id):
                    # Simulate typical chunk processing load
                    frames = torch.randn(10, 3, 256, 256, device=f'cuda:{gpu_id}')
                    
                    # Simulate feature extraction
                    for i in range(5):
                        processed = torch.nn.functional.conv2d(
                            frames.view(-1, 3, 256, 256),
                            torch.randn(64, 3, 3, 3, device=f'cuda:{gpu_id}')
                        )
                    
                    result = torch.sum(processed).item()
                
                # Return GPU to queue
                gpu_queue.put(gpu_id)
                
                total_time = time.time() - start_time
                results_dict[chunk_id] = {
                    'status': 'SUCCESS',
                    'gpu_id': gpu_id,
                    'acquire_time': acquire_time,
                    'total_time': total_time,
                    'result': result
                }
                
                if chunk_id % 10 == 0:
                    print(f"✅ Chunk {chunk_id}: GPU {gpu_id} ({total_time:.2f}s)")
                
            except Exception as e:
                results_dict[chunk_id] = {'status': 'ERROR', 'error': str(e)}
                print(f"❌ Chunk {chunk_id}: Error - {e}")
                
                # Try to return GPU even on error
                try:
                    gpu_queue.put(gpu_id)
                except:
                    pass
        
        # Create GPU queue
        gpu_queue = queue.Queue()
        for gpu_id in self.gpu_ids:
            gpu_queue.put(gpu_id)
        
        # Simulate processing many chunks
        num_chunks = 50  # Simulate 50 chunks
        high_load_results = {}
        threads = []
        
        print(f"Simulating {num_chunks} chunks across {len(self.gpu_ids)} GPUs...")
        
        for chunk_id in range(num_chunks):
            thread = threading.Thread(
                target=chunk_processor,
                args=(chunk_id, gpu_queue, high_load_results)
            )
            threads.append(thread)
            thread.start()
            
            # Small delay to prevent overwhelming
            time.sleep(0.01)
        
        # Wait for all chunks
        for thread in threads:
            thread.join()
        
        self.results['high_load'] = high_load_results
    
    def test_memory_patterns(self):
        """Test memory allocation patterns that might cause issues"""
        print("\n🧪 Test 6: Memory Allocation Patterns")
        print("-" * 30)
        
        memory_results = {}
        
        for gpu_id in self.gpu_ids:
            print(f"Testing memory patterns on GPU {gpu_id}...")
            
            try:
                with torch.cuda.device(gpu_id):
                    # Test 1: Rapid allocation/deallocation
                    for i in range(100):
                        tensor = torch.randn(100, 100, device=f'cuda:{gpu_id}')
                        del tensor
                    
                    # Test 2: Large allocation
                    try:
                        large_tensor = torch.randn(2000, 2000, device=f'cuda:{gpu_id}')
                        del large_tensor
                    except torch.cuda.OutOfMemoryError:
                        print(f"⚠️ GPU {gpu_id}: Large allocation failed (expected)")
                    
                    # Test 3: Memory fragmentation
                    tensors = []
                    try:
                        for i in range(20):
                            tensors.append(torch.randn(200, 200, device=f'cuda:{gpu_id}'))
                    except torch.cuda.OutOfMemoryError:
                        print(f"⚠️ GPU {gpu_id}: Fragmentation test hit OOM")
                    finally:
                        del tensors
                    
                    torch.cuda.empty_cache()
                    
                    memory_results[gpu_id] = {'status': 'PASS'}
                    print(f"✅ GPU {gpu_id}: Memory patterns test passed")
                    
            except Exception as e:
                memory_results[gpu_id] = {'status': 'FAIL', 'error': str(e)}
                print(f"❌ GPU {gpu_id}: Memory test failed - {e}")
        
        self.results['memory_patterns'] = memory_results
    
    def print_diagnostic_summary(self):
        """Print comprehensive diagnostic summary"""
        print("\n" + "=" * 60)
        print("🎯 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Overall status
        all_tests = ['cuda_basic', 'individual_gpus', 'concurrent_access', 'queue_management', 'high_load', 'memory_patterns']
        passed_tests = 0
        total_tests = len(all_tests)
        
        for test_name in all_tests:
            if test_name in self.results:
                if test_name == 'individual_gpus':
                    # Special handling for individual GPU tests
                    gpu_passes = sum(1 for gpu_result in self.results[test_name].values() 
                                   if gpu_result.get('status') == 'PASS')
                    if gpu_passes == len(self.gpu_ids):
                        passed_tests += 1
                        print(f"✅ {test_name}: PASS ({gpu_passes}/{len(self.gpu_ids)} GPUs)")
                    else:
                        print(f"❌ {test_name}: PARTIAL ({gpu_passes}/{len(self.gpu_ids)} GPUs)")
                
                elif test_name in ['concurrent_access', 'queue_management', 'high_load']:
                    # Special handling for multi-result tests
                    results = self.results[test_name]
                    successes = sum(1 for r in results.values() 
                                  if r.get('status') in ['PASS', 'SUCCESS'])
                    total = len(results)
                    
                    if successes > total * 0.8:  # 80% success rate
                        passed_tests += 1
                        print(f"✅ {test_name}: PASS ({successes}/{total} successful)")
                    else:
                        print(f"❌ {test_name}: FAIL ({successes}/{total} successful)")
                        
                        # Show failures for debugging
                        failures = [k for k, v in results.items() 
                                  if v.get('status') not in ['PASS', 'SUCCESS']]
                        if failures:
                            print(f"   Failures: {failures[:5]}{'...' if len(failures) > 5 else ''}")
                
                else:
                    # Simple pass/fail tests
                    status = self.results[test_name].get('status', 'UNKNOWN')
                    if status == 'PASS':
                        passed_tests += 1
                        print(f"✅ {test_name}: PASS")
                    else:
                        print(f"❌ {test_name}: {status}")
        
        print(f"\n📊 Overall Score: {passed_tests}/{total_tests} tests passed")
        
        # Specific recommendations based on results
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! GPU system is healthy.")
            print("   The timeout issue is likely in the matcher.py GPU manager implementation.")
            print("   Recommend checking the GPU acquisition logic in the chunked processor.")
        
        elif 'queue_management' in self.results:
            queue_results = self.results['queue_management']
            timeouts = sum(1 for r in queue_results.values() 
                         if 'timeout' in r.get('error', '').lower())
            if timeouts > 0:
                print(f"🔍 Found {timeouts} timeout issues in queue management test")
                print("   This confirms the GPU queue system has problems")
                print("   Recommend fixing the GPU manager queue implementation")
        
        if 'high_load' in self.results:
            high_load_results = self.results['high_load']
            errors = sum(1 for r in high_load_results.values() 
                        if r.get('status') == 'ERROR')
            timeouts = sum(1 for r in high_load_results.values() 
                          if r.get('status') == 'TIMEOUT')
            
            if errors > 0:
                print(f"⚠️ Found {errors} errors under high load")
            if timeouts > 0:
                print(f"⚠️ Found {timeouts} timeouts under high load")
                print("   GPU queue is not handling concurrent access properly")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Review the detailed results above")
        print("2. Fix any failing GPU tests")
        print("3. Implement improved GPU manager if queue issues found")
        print("4. Re-test with fixed components")

def main():
    """Run GPU diagnostic"""
    gpu_ids = [0, 1]  # Adjust based on your system
    
    print("Starting comprehensive GPU diagnostic...")
    print(f"Testing GPUs: {gpu_ids}")
    
    diagnostic = GPUDiagnostic(gpu_ids)
    diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    main()