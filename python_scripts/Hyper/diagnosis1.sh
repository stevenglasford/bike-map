#!/bin/bash

# Diagnostic script to find why turbo mode isn't working

echo "=== TURBO MODE DIAGNOSTIC ==="
echo "Date: $(date)"
echo ""

echo "=== GPU HARDWARE CHECK ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "=== PYTHON ENVIRONMENT CHECK ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

echo "=== RUNNING MATCHER WITH DEBUG ==="
python -c "
import argparse
import sys
sys.path.append('.')

# Simulate your exact arguments
class MockArgs:
    def __init__(self):
        self.turbo_mode = True
        self.gpu_batch_size = 32
        self.correlation_batch_size = 2000
        self.max_cpu_workers = 0
        self.max_gpu_memory = 8.0
        self.parallel_videos = 8
        self.cuda_streams = True
        self.vectorized_ops = True
        self.gpu_ids = [0, 1]

# Create config-like object
args = MockArgs()

print('=== CONFIG VERIFICATION ===')
print(f'turbo_mode: {args.turbo_mode}')
print(f'gpu_batch_size: {args.gpu_batch_size}')
print(f'Condition (turbo_mode and gpu_batch_size > 1): {args.turbo_mode and args.gpu_batch_size > 1}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test GPU manager initialization
try:
    import torch
    gpu_manager_test = torch.cuda.is_available()
    print(f'GPU manager test: {gpu_manager_test}')
    
    if gpu_manager_test:
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            test_tensor = torch.randn(10, 10).to(device)
            print(f'GPU {i} test: SUCCESS')
except Exception as e:
    print(f'GPU manager test FAILED: {e}')

print('')
"

echo "=== DEPENDENCY CHECK ==="
python -c "
import sys
modules = ['torch', 'torchvision', 'cv2', 'numpy', 'gpxpy', 'pandas', 'tqdm', 'numba']
for module in modules:
    try:
        __import__(module)
        print(f'{module}: ✅ OK')
    except ImportError as e:
        print(f'{module}: ❌ MISSING - {e}')

# Check CuPy specifically
try:
    import cupy as cp
    print(f'cupy: ✅ OK (version {cp.__version__})')
    print(f'CuPy CUDA available: {cp.cuda.is_available()}')
except ImportError:
    print('cupy: ⚠️  NOT INSTALLED')
except Exception as e:
    print(f'cupy: ❌ ERROR - {e}')
"

echo ""
echo "=== MEMORY CHECK ==="
free -h
echo ""
df -h /tmp
echo ""

echo "=== PROCESS CHECK ==="
ps aux | grep python | grep -v grep
echo ""

echo "=== LOG RECOMMENDED FIXES ==="
echo "1. If CUDA not available: Fix PyTorch CUDA installation"
echo "2. If GPU test fails: Check nvidia-smi and reboot"
echo "3. If CuPy missing: pip install cupy-cuda11x (or cupy-cuda12x)"
echo "4. If memory low: Close other processes"
echo "5. If all OK: Check for syntax errors in matcher50.py"