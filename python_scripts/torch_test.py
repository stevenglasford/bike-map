import torch

# Check if CUDA (NVIDIA GPU support) is available
print("CUDA available:", torch.cuda.is_available())

# Check the name of the GPU
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
    print("Device count:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)
else:
    print("No CUDA-compatible GPU detected.")
	
x = torch.rand(3, 3).cuda()  # Moves tensor to GPU
y = torch.rand(3, 3).cuda()
z = x + y  # Operation on GPU
print("Tensor on GPU:", z)
print("Device of result:", z.device)