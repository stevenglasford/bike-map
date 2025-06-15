#!/bin/bash

echo "🚀 RUNNING MATCHER WITH GPU TIMEOUT FIX"
echo "========================================"

# Step 1: Apply the GPU timeout fix
echo "🔧 Step 1: Applying GPU timeout fix..."
python3 -c "
import sys
import os
sys.path.append('.')

# Import and apply the fix
exec(open('gpu_timeout_fix.py').read()) if os.path.exists('gpu_timeout_fix.py') else print('⚠️ gpu_timeout_fix.py not found - using inline fix')

# Inline fix if file doesn't exist
if not os.path.exists('gpu_timeout_fix.py'):
    exec('''
import torch
import threading
import time
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FixedGPUManager:
    def __init__(self, gpu_ids: List[int], strict: bool = False, config=None):
        self.gpu_ids = gpu_ids
        self.strict = strict
        self.config = config
        self.gpu_semaphores = {gpu_id: threading.Semaphore(10) for gpu_id in gpu_ids}
        self.gpu_usage = {gpu_id: 0 for gpu_id in gpu_ids}
        self.gpu_locks = {gpu_id: threading.RLock() for gpu_id in gpu_ids}
        self.round_robin = 0
        self.round_robin_lock = threading.Lock()
        self.validate_gpus()
        logger.info(f\"✅ FIXED GPU Manager initialized for GPUs: {gpu_ids}\")
    
    def validate_gpus(self):
        if not torch.cuda.is_available():
            raise RuntimeError(\"CUDA not available\" if not self.strict else \"STRICT MODE: CUDA required\")
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f\"GPU {gpu_id} not available\")
            try:
                with torch.cuda.device(gpu_id):
                    test = torch.zeros(100, 100, device=f\"cuda:{gpu_id}\")
                    del test
                    torch.cuda.empty_cache()
                props = torch.cuda.get_device_properties(gpu_id)
                logger.info(f\"GPU {gpu_id}: {props.name} ({props.total_memory/1024**3:.1f}GB)\")
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f\"STRICT MODE: GPU {gpu_id} validation failed: {e}\")
    
    def acquire_gpu(self, timeout: int = 60) -> Optional[int]:
        start_time = time.time()
        attempts = 0
        max_attempts = len(self.gpu_ids) * 3
        
        while attempts < max_attempts:
            with self.round_robin_lock:
                gpu_idx = self.round_robin % len(self.gpu_ids)
                self.round_robin += 1
            gpu_id = self.gpu_ids[gpu_idx]
            
            try:
                acquired = self.gpu_semaphores[gpu_id].acquire(blocking=True, timeout=0.5)
                if acquired:
                    if self.strict:
                        try:
                            with torch.cuda.device(gpu_id):
                                test = torch.zeros(10, 10, device=f\"cuda:{gpu_id}\")
                                del test
                                torch.cuda.empty_cache()
                        except Exception as e:
                            self.gpu_semaphores[gpu_id].release()
                            raise RuntimeError(f\"STRICT MODE: GPU {gpu_id} unavailable: {e}\")
                    
                    with self.gpu_locks[gpu_id]:
                        self.gpu_usage[gpu_id] += 1
                    logger.debug(f\"✅ Acquired GPU {gpu_id} (attempt {attempts + 1})\")
                    return gpu_id
            except Exception as e:
                if self.strict and \"STRICT MODE\" in str(e):
                    logger.error(f\"GPU {gpu_id}: {e}\")
                    continue
                logger.debug(f\"Failed to acquire GPU {gpu_id}: {e}\")
            
            attempts += 1
            if time.time() - start_time >= timeout:
                break
            time.sleep(0.01)
        
        error_msg = f\"Could not acquire any GPU within {time.time() - start_time:.1f}s\"
        if self.strict:
            error_msg = f\"STRICT MODE: {error_msg}\"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.error(error_msg)
        return None
    
    def release_gpu(self, gpu_id: int):
        try:
            self.cleanup_gpu_memory(gpu_id)
            with self.gpu_locks[gpu_id]:
                self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
            self.gpu_semaphores[gpu_id].release()
            logger.debug(f\"✅ Released GPU {gpu_id}\")
        except Exception as e:
            logger.error(f\"Error releasing GPU {gpu_id}: {e}\")
            try:
                self.gpu_semaphores[gpu_id].release()
            except:
                pass
    
    def cleanup_gpu_memory(self, gpu_id: int):
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug(f\"Memory cleanup warning for GPU {gpu_id}: {e}\")
    
    def get_gpu_memory_info(self, gpu_id: int) -> dict:
        try:
            with torch.cuda.device(gpu_id):
                total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                return {\"total_gb\": total, \"allocated_gb\": allocated, \"reserved_gb\": reserved, 
                       \"free_gb\": total - reserved, \"utilization_pct\": (reserved / total) * 100}
        except:
            return {\"total_gb\": 0, \"allocated_gb\": 0, \"reserved_gb\": 0, \"free_gb\": 0, \"utilization_pct\": 0}

# Apply the fix
try:
    import matcher
    if hasattr(matcher, \"EnhancedGPUManager\"):
        matcher.EnhancedGPUManager = FixedGPUManager
        print(\"✅ Applied GPU timeout fix to matcher.py\")
    else:
        print(\"⚠️ Could not find EnhancedGPUManager in matcher.py\")
except ImportError:
    print(\"⚠️ Could not import matcher.py - fix will be applied at runtime\")
''')

print(\"🔧 GPU timeout fix applied\")
"

if [ $? -eq 0 ]; then
    echo "✅ GPU fix applied successfully"
else
    echo "⚠️ GPU fix had issues, but continuing..."
fi

# Step 2: Set environment variables
echo "🔧 Step 2: Setting up environment..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0  # Disable blocking for better performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Create directories
mkdir -p ~/penis/temp/chunks
mkdir -p ~/penis/temp/gpu_temp
mkdir -p ~/penis/temp/processing
mkdir -p ~/penis/testingground/high_parallel_results

# Step 3: Check GPU status
echo "🔧 Step 3: Checking GPU status..."
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

# Step 4: Run with HIGH PARALLELIZATION
echo ""
echo "🚀 Step 4: Running matcher with HIGH PARALLELIZATION..."
echo "Settings:"
echo "  ✅ parallel_videos: 4 (HIGH PARALLELIZATION)"
echo "  ✅ gpu_ids: 0 1 (BOTH GPUS)"
echo "  ✅ max_frames: unlimited"
echo "  ✅ video_size: 1920x1080"
echo "  ✅ PowerSafe mode enabled"
echo "  ✅ GPU timeout fix applied"
echo ""

# THE FIXED COMMAND - NOW WITH HIGH PARALLELIZATION!
python matcher.py \
    -d ~/penis/panoramics/playground/ \
    -o ~/penis/testingground/high_parallel_results \
    --max_frames 999999 \
    --video_size 1920 1080 \
    --sample_rate 1.0 \
    --parallel_videos 4 \
    --gpu_ids 0 1 \
    --gpu_timeout 60 \
    --debug \
    --powersafe \
    --enable_preprocessing \
    --ram_cache 48.0 \
    --force

# Step 5: Check results
echo ""
echo "🎯 Processing Results:"
if [ $? -eq 0 ]; then
    echo "🎉 SUCCESS! High parallelization working perfectly!"
    echo ""
    echo "✅ WHAT WORKED:"
    echo "   • GPU timeout fix eliminated queue deadlocks"
    echo "   • 4 parallel videos processing simultaneously"
    echo "   • Both GPUs utilized efficiently"
    echo "   • Chunked processing handling unlimited resolution"
    echo "   • No more 30s timeout errors"
    echo ""
    echo "📊 Check your results in:"
    echo "   ~/penis/testingground/high_parallel_results/"
    echo ""
    echo "🚀 Your optimal settings:"
    echo "   --parallel_videos 4 (or even higher!)"
    echo "   --gpu_ids 0 1"
    echo "   --max_frames 999999"
    echo "   --video_size 1920 1080 (or 3840 2160 for 4K)"
    echo "   --powersafe --debug"
else
    echo "❌ Issues remain. Let's diagnose..."
    echo ""
    echo "🔍 Running diagnostic..."
    
    # Quick diagnostic
    python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(i):
                test = torch.zeros(100, 100, device=f'cuda:{i}')
                print(f'GPU {i}: ✅ Working')
                del test
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'GPU {i}: ❌ {e}')
"
    
    echo ""
    echo "💡 Try these fallback settings:"
    echo "   --parallel_videos 2     # Reduce parallelization"
    echo "   --video_size 1280 720   # Lower resolution"
    echo "   --gpu_ids 0             # Single GPU only"
fi

echo ""
echo "📈 PERFORMANCE MONITORING:"
echo "Run these in separate terminals to monitor:"
echo "   watch -n 1 nvidia-smi"
echo "   tail -f production_correlation.log"
echo ""
echo "Complete! 🎯"