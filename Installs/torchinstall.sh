cd ~
rm -rf pytorch

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py clean

# Environment setup
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-"$(dirname $(which conda))/../"}"
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print(f'{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}')")

# Important for cuFile support
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=1
export USE_CUFILE=1
export USE_FAST_MATH=1

# Ensure CMake finds cuFile
export CMAKE_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib
export CMAKE_INCLUDE_PATH=/usr/local/cuda-12.9/include
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

export LIBRARY_PATH=$CMAKE_LIBRARY_PATH


# Optional: check that cuFile exists
if [ ! -f /usr/local/cuda-12.9/targets/x86_64-linux/lib/libcufile.so ]; then
    echo "❌ cuFile not found in expected directory"
    exit 1
fi

# Clean old builds
python setup.py clean

# Compile PyTorch
MAX_JOBS=$(nproc) python setup.py develop 2>&1 | tee ~/torchinstallerrors.txt
