import os
import sys
import subprocess
import torch

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

safe_print("========== CUDA ENVIRONMENT CHECK ==========")

# 1. PyTorch CUDA info
try:
    safe_print(f"PyTorch version: {torch.__version__}")
    safe_print(f"PyTorch CUDA version: {torch.version.cuda}")
    safe_print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        safe_print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        safe_print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
except Exception as e:
    safe_print(f"Error checking PyTorch CUDA info: {e}")

# 2. nvcc version
try:
    nvcc_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
    safe_print("\n--- nvcc version ---")
    safe_print(nvcc_output)
except FileNotFoundError:
    safe_print("nvcc not found (not installed or not in PATH)")
except Exception as e:
    safe_print(f"Error checking nvcc: {e}")

# 3. Environment variables
safe_print("\n--- Environment ---")
for var in ["CUDA_HOME", "CUDA_PATH", "PATH", "LD_LIBRARY_PATH"]:
    val = os.environ.get(var, "(not set)")
    safe_print(f"{var}: {val}")

# 4. Local CUDA installations
try:
    output = subprocess.check_output(["ls", "/usr/local/"]).decode()
    dirs = [line for line in output.splitlines() if "cuda" in line]
    safe_print("\n--- /usr/local/ contents ---")
    for d in dirs:
        safe_print(f"Found: /usr/local/{d}")
except Exception as e:
    safe_print(f"Error listing /usr/local/: {e}")

# 5. Optional OpenCV CUDA check
try:
    import cv2
    safe_print(f"\nOpenCV version: {cv2.__version__}")
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        safe_print("OpenCV CUDA: ENABLED")
    else:
        safe_print("OpenCV CUDA: NOT ENABLED")
except ImportError:
    safe_print("OpenCV not installed")
except Exception as e:
    safe_print(f"Error checking OpenCV: {e}")