#!/usr/bin/env python3
"""
Quick test to see what's preventing turbo mode from activating
"""

import torch

# Test the exact conditions from your matcher50.py
class MockConfig:
    def __init__(self):
        self.turbo_mode = True
        self.gpu_batch_size = 32

config = MockConfig()

print("=== TURBO MODE ACTIVATION TEST ===")
print(f"config.turbo_mode: {config.turbo_mode}")
print(f"config.gpu_batch_size: {config.gpu_batch_size}")
print(f"config.gpu_batch_size > 1: {config.gpu_batch_size > 1}")
print(f"Full condition: {config.turbo_mode and config.gpu_batch_size > 1}")
print("")

print("=== PYTORCH/CUDA CHECK ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        try:
            # Test basic GPU operations
            device = torch.device(f'cuda:{i}')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"  - GPU {i} computation test: ✅ PASSED")
        except Exception as e:
            print(f"  - GPU {i} computation test: ❌ FAILED - {e}")
print("")

print("=== EXPECTED BEHAVIOR ===")
if config.turbo_mode and config.gpu_batch_size > 1:
    print("✅ Should see: '🚀 Initializing GPU batch correlation engine for maximum performance...'")
    if torch.cuda.is_available():
        print("✅ All conditions met for GPU batch processing")
    else:
        print("❌ CUDA not available - this is likely the problem!")
else:
    print("❌ Should see: '⚡ Initializing enhanced similarity engine with RAM cache...'")
    print("❌ This means turbo condition failed")

print("")
print("=== TROUBLESHOOTING ===")
if not torch.cuda.is_available():
    print("🔧 FIX: Install PyTorch with CUDA support:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
elif torch.cuda.device_count() == 0:
    print("🔧 FIX: No CUDA devices detected. Check nvidia-smi")
else:
    print("🔧 Hardware OK - check matcher50.py for syntax errors or missing imports")