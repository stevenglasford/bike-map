#!/usr/bin/env python3
"""
GPU Debugging and Diagnostics
Add these debug functions to identify GPU utilization issues
"""

import cupy as cp
import threading
import time
import psutil
import logging

class GPUMonitor:
    """Real-time GPU monitoring to identify utilization issues"""
    
    def __init__(self, gpu_ids=[0, 1]):
        self.gpu_ids = gpu_ids
        self.monitoring = False
        self.stats = {gpu_id: [] for gpu_id in gpu_ids}
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start background GPU monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and print summary"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self._print_summary()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            for gpu_id in self.gpu_ids:
                try:
                    # Check GPU memory usage
                    with cp.cuda.Device(gpu_id):
                        mempool = cp.get_default_memory_pool()
                        used_bytes = mempool.used_bytes()
                        total_bytes = mempool.total_bytes()
                        
                        # Check if GPU is actually being used
                        device = cp.cuda.Device(gpu_id)
                        mem_info = device.mem_info
                        
                        self.stats[gpu_id].append({
                            'timestamp': time.time(),
                            'used_mb': used_bytes / (1024*1024),
                            'total_mb': total_bytes / (1024*1024),
                            'device_mem_used': (mem_info[1] - mem_info[0]) / (1024*1024),
                            'device_mem_total': mem_info[1] / (1024*1024)
                        })
                        
                except Exception as e:
                    logging.debug(f"GPU {gpu_id} monitoring error: {e}")
                    
            time.sleep(1)  # Monitor every second
    
    def _print_summary(self):
        """Print GPU utilization summary"""
        print("\n" + "="*80)
        print("🔍 GPU UTILIZATION ANALYSIS")
        print("="*80)
        
        for gpu_id in self.gpu_ids:
            stats = self.stats[gpu_id]
            if not stats:
                print(f"❌ GPU {gpu_id}: No data collected")
                continue
                
            # Calculate utilization metrics
            used_values = [s['used_mb'] for s in stats]
            device_used_values = [s['device_mem_used'] for s in stats]
            
            avg_used = sum(used_values) / len(used_values)
            max_used = max(used_values)
            active_time = sum(1 for u in used_values if u > 10) / len(used_values) * 100
            
            avg_device_used = sum(device_used_values) / len(device_used_values)
            max_device_used = max(device_used_values)
            
            print(f"🎮 GPU {gpu_id}:")
            print(f"   Memory Pool - Avg: {avg_used:.1f}MB, Max: {max_used:.1f}MB")
            print(f"   Device Memory - Avg: {avg_device_used:.1f}MB, Max: {max_device_used:.1f}MB")
            print(f"   Active Time: {active_time:.1f}%")
            print(f"   Samples: {len(stats)}")


def debug_gpu_initialization():
    """Debug GPU initialization issues"""
    print("\n🔍 DEBUGGING GPU INITIALIZATION")
    print("="*50)
    
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"✅ Detected {gpu_count} GPUs")
        
        for gpu_id in range(min(gpu_count, 2)):
            print(f"\n🎮 Testing GPU {gpu_id}:")
            
            try:
                # Test basic initialization
                original_device = cp.cuda.Device()
                cp.cuda.Device(gpu_id).use()
                
                # Test memory allocation
                test_array = cp.random.rand(1000, 1000)
                result = cp.sum(test_array)
                
                # Test memory pool
                mempool = cp.get_default_memory_pool()
                used_before = mempool.used_bytes()
                
                larger_array = cp.random.rand(5000, 5000)
                used_after = mempool.used_bytes()
                
                del test_array, larger_array
                
                print(f"   ✅ Basic operations: PASS")
                print(f"   ✅ Memory allocation: PASS ({(used_after-used_before)/(1024*1024):.1f}MB)")
                
                # Test concurrent access
                def gpu_worker():
                    try:
                        cp.cuda.Device(gpu_id).use()
                        arr = cp.random.rand(100, 100)
                        result = cp.sum(arr)
                        return True
                    except:
                        return False
                
                # Test with multiple threads
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(gpu_worker) for _ in range(3)]
                    results = [f.result() for f in futures]
                    
                success_count = sum(results)
                print(f"   ✅ Concurrent access: {success_count}/3 threads successful")
                
                # Restore original device
                original_device.use()
                
            except Exception as e:
                print(f"   ❌ GPU {gpu_id} test failed: {e}")
                
    except Exception as e:
        print(f"❌ GPU initialization debug failed: {e}")


def debug_worker_assignment():
    """Debug worker-to-GPU assignment"""
    print("\n🔍 DEBUGGING WORKER-GPU ASSIGNMENT")
    print("="*50)
    
    import queue
    import threading
    
    # Simulate the worker creation
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Add some dummy work
    for i in range(10):
        work_queue.put(f"work_item_{i}")
    
    class TestWorker:
        def __init__(self, worker_id, gpu_id):
            self.worker_id = worker_id
            self.gpu_id = gpu_id
            self.processed = 0
            
        def run(self):
            while True:
                try:
                    work_item = work_queue.get(timeout=2)
                    if work_item is None:
                        break
                        
                    # Test GPU assignment
                    cp.cuda.Device(self.gpu_id).use()
                    current_device = cp.cuda.Device()
                    
                    print(f"   Worker {self.worker_id}: Processing {work_item} on GPU {current_device.id}")
                    
                    # Simple GPU operation
                    arr = cp.random.rand(100, 100)
                    result = cp.sum(arr)
                    
                    self.processed += 1
                    result_queue.put((self.worker_id, work_item, current_device.id))
                    work_queue.task_done()
                    
                    time.sleep(0.1)  # Simulate processing time
                    
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"   ❌ Worker {self.worker_id} error: {e}")
                    work_queue.task_done()
    
    # Create test workers
    workers = []
    threads = []
    
    for gpu_id in [0, 1]:
        for worker_idx in range(2):
            worker_id = f"GPU{gpu_id}_W{worker_idx}"
            worker = TestWorker(worker_id, gpu_id)
            workers.append(worker)
            
            thread = threading.Thread(target=worker.run)
            threads.append(thread)
            thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Analyze results
    gpu0_count = 0
    gpu1_count = 0
    
    while not result_queue.empty():
        worker_id, work_item, actual_gpu = result_queue.get()
        if actual_gpu == 0:
            gpu0_count += 1
        elif actual_gpu == 1:
            gpu1_count += 1
            
    print(f"   📊 GPU 0 processed: {gpu0_count} items")
    print(f"   📊 GPU 1 processed: {gpu1_count} items")
    
    if gpu1_count == 0:
        print("   ❌ GPU 1 did not process any items - PROBLEM IDENTIFIED!")
    elif abs(gpu0_count - gpu1_count) > 2:
        print("   ⚠️  Significant load imbalance detected")
    else:
        print("   ✅ Work distribution looks good")


def debug_cupy_context():
    """Debug CuPy context switching issues"""
    print("\n🔍 DEBUGGING CUPY CONTEXT SWITCHING")
    print("="*50)
    
    try:
        # Test context switching in single thread
        print("   Testing single-thread context switching:")
        
        for gpu_id in [0, 1]:
            cp.cuda.Device(gpu_id).use()
            current = cp.cuda.Device()
            arr = cp.random.rand(100, 100)
            result = cp.sum(arr)
            print(f"     GPU {gpu_id}: Set to {gpu_id}, actual {current.id}, result {result:.2f}")
        
        # Test context switching in multiple threads
        print("   Testing multi-thread context switching:")
        
        def thread_worker(gpu_id, results_dict):
            try:
                cp.cuda.Device(gpu_id).use()
                current = cp.cuda.Device()
                
                # Do some work
                arr = cp.random.rand(1000, 1000)
                result = cp.sum(arr)
                
                results_dict[f"thread_gpu_{gpu_id}"] = {
                    'requested_gpu': gpu_id,
                    'actual_gpu': current.id,
                    'result': float(result),
                    'success': True
                }
            except Exception as e:
                results_dict[f"thread_gpu_{gpu_id}"] = {
                    'requested_gpu': gpu_id,
                    'actual_gpu': -1,
                    'result': 0,
                    'success': False,
                    'error': str(e)
                }
        
        import threading
        results = {}
        threads = []
        
        for gpu_id in [0, 1]:
            thread = threading.Thread(target=thread_worker, args=(gpu_id, results))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        for key, result in results.items():
            if result['success']:
                print(f"     {key}: Requested GPU {result['requested_gpu']}, "
                      f"got GPU {result['actual_gpu']}, result {result['result']:.2f}")
            else:
                print(f"     {key}: FAILED - {result['error']}")
                
    except Exception as e:
        print(f"   ❌ Context switching test failed: {e}")


# Add this to your main worker class for detailed logging
class EnhancedDualGPUWorker:
    """Enhanced worker with detailed GPU debugging"""
    
    def __init__(self, worker_id: str, gpu_id: int, work_queue, result_queue, gpu_memory_gb=0):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.errors = 0
        self.logger = logging.getLogger(f'Worker_{worker_id}')
        self.is_running = True
        
        # GPU debugging info
        self.gpu_operations_count = 0
        self.context_switches = 0
        
    def run(self):
        """Enhanced run with GPU debugging"""
        
        self.logger.info(f"🔥 Worker {self.worker_id} starting on GPU {self.gpu_id}")
        
        # Enhanced GPU initialization with verification
        if not self._initialize_gpu_enhanced():
            self.logger.error(f"💀 Worker {self.worker_id}: Enhanced GPU {self.gpu_id} init failed")
            return
        
        # Main processing loop with GPU monitoring
        while self.is_running:
            try:
                work_item = self.work_queue.get(timeout=5)
                if work_item is None:
                    break
                
                # Verify GPU context before processing
                actual_gpu = self._verify_gpu_context()
                if actual_gpu != self.gpu_id:
                    self.logger.warning(f"⚠️  Worker {self.worker_id}: Expected GPU {self.gpu_id}, "
                                      f"but on GPU {actual_gpu}")
                    self._force_gpu_context()
                
                # Process with enhanced monitoring
                result = self._process_with_monitoring(*work_item)
                self.result_queue.put(result)
                self.processed += 1
                
                # Enhanced progress logging
                if self.processed % 3 == 0:
                    gpu_mem = GPUManager.get_gpu_memory_usage(self.gpu_id)
                    self.logger.info(f"🔥 Worker {self.worker_id}: {self.processed} processed, "
                                   f"{self.gpu_operations_count} GPU ops, "
                                   f"{self.context_switches} context switches, "
                                   f"{gpu_mem:.1f}MB GPU mem")
                
                self.work_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"💀 Worker {self.worker_id}: Error: {e}")
                self.errors += 1
                self.work_queue.task_done()
        
        self.logger.info(f"🔥 Worker {self.worker_id} finished: {self.processed} processed, "
                        f"{self.errors} errors, {self.gpu_operations_count} GPU operations")
    
    def _initialize_gpu_enhanced(self) -> bool:
        """Enhanced GPU initialization with detailed verification"""
        try:
            # Set device and verify
            cp.cuda.Device(self.gpu_id).use()
            actual_device = cp.cuda.Device()
            
            if actual_device.id != self.gpu_id:
                self.logger.error(f"GPU context mismatch: requested {self.gpu_id}, got {actual_device.id}")
                return False
            
            # Test basic operations
            test_array = cp.random.rand(100, 100)
            result = cp.sum(test_array)
            cp.cuda.Device(self.gpu_id).synchronize()
            
            if not (0 < result < 10000):
                self.logger.error(f"GPU computation test failed: result {result}")
                return False
            
            # Set memory limit if specified
            if self.gpu_memory_gb > 0:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.gpu_memory_gb * 1024**3))
                self.logger.info(f"Set GPU {self.gpu_id} memory limit: {self.gpu_memory_gb}GB")
            
            del test_array
            self.logger.info(f"✅ Enhanced GPU {self.gpu_id} initialization successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced GPU {self.gpu_id} initialization failed: {e}")
            return False
    
    def _verify_gpu_context(self) -> int:
        """Verify current GPU context"""
        try:
            current_device = cp.cuda.Device()
            return current_device.id
        except:
            return -1
    
    def _force_gpu_context(self):
        """Force GPU context switch"""
        try:
            cp.cuda.Device(self.gpu_id).use()
            self.context_switches += 1
            
            # Verify the switch worked
            actual_gpu = self._verify_gpu_context()
            if actual_gpu != self.gpu_id:
                self.logger.error(f"Force context switch failed: still on GPU {actual_gpu}")
            else:
                self.logger.info(f"✅ Forced context switch to GPU {self.gpu_id}")
        except Exception as e:
            self.logger.error(f"Force context switch error: {e}")
    
    def _process_with_monitoring(self, video_path, gpx_path, match):
        """Process match with GPU operation monitoring"""
        
        # Count this as a GPU operation
        self.gpu_operations_count += 1
        
        # Ensure we're on the right GPU
        cp.cuda.Device(self.gpu_id).use()
        
        # Your existing process_match logic here
        # ... (copy from your DualGPUWorker.process_match method)
        
        return match  # Placeholder


if __name__ == "__main__":
    print("🔍 GPU DEBUGGING SUITE")
    print("="*80)
    
    debug_gpu_initialization()
    debug_cupy_context()
    debug_worker_assignment()
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Run this script to identify specific GPU issues")
    print("2. Add GPUMonitor to your main processing loop")
    print("3. Replace DualGPUWorker with EnhancedDualGPUWorker")
    print("4. Check the output for specific failure patterns")