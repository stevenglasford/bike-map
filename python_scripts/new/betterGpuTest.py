#!/usr/bin/env python3

import subprocess
import sys

def check_gpu_capabilities():
try:
# Get GPU info
result = subprocess.run([‘nvidia-smi’, ‘–query-gpu=name,driver_version,compute_cap’, ‘–format=csv,noheader,nounits’],
capture_output=True, text=True)

```
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            parts = line.split(', ')
            gpu_name = parts[0]
            driver_version = parts[1]
            compute_cap = parts[2]
            
            print(f"GPU {i}: {gpu_name}")
            print(f"Driver Version: {driver_version}")
            print(f"Compute Capability: {compute_cap}")
            
            # Check NVDEC support (rough estimate based on GPU generation)
            major_cap = int(compute_cap.split('.')[0])
            if major_cap >= 6:  # Maxwell generation and newer typically support NVDEC
                print("✅ Likely supports NVDEC")
            else:
                print("❌ May not support NVDEC")
            print("-" * 40)
    else:
        print("❌ Could not get GPU information")
        
except Exception as e:
    print(f"❌ Error checking GPU: {e}")
```

def check_nvdec_libraries():
import os

```
print("Checking NVDEC libraries:")

# Common locations for NVDEC libraries
nvdec_paths = [
    "/usr/lib/x86_64-linux-gnu/libnvcuvid.so",
    "/usr/local/cuda/lib64/libnvcuvid.so",
    "/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1",
]

for path in nvdec_paths:
    if os.path.exists(path):
        print(f"✅ Found: {path}")
    else:
        print(f"❌ Missing: {path}")
```

if **name** == “**main**”:
check_gpu_capabilities()
print()
check_nvdec_libraries()