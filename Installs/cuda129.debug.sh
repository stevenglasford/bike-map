#!/bin/bash
echo "🔍 CUDA Diagnostic Report"
echo "=========================="

echo -e "\n📍 Environment Variables:"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo -e "\n📍 Which CUDA tools are being found:"
echo "which nvcc: $(which nvcc)"
echo "which nvidia-smi: $(which nvidia-smi)"

echo -e "\n📍 NVCC version:"
nvcc --version

echo -e "\n📍 All CUDA installations on system:"
find /usr -name "nvcc" 2>/dev/null
find /usr/local -name "nvcc" 2>/dev/null
ls -la /usr/local/ | grep cuda

echo -e "\n📍 CUDA libraries being found:"
find /usr -name "libcuda*" 2>/dev/null | head -10
find /usr/local -name "libcuda*" 2>/dev/null | head -10

echo -e "\n📍 PKG_CONFIG_PATH:"
echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
pkg-config --list-all | grep -i cuda 2>/dev/null || echo "No CUDA pkg-config found"

echo -e "\n📍 CMake CUDA detection test:"
cat > /tmp/test_cuda.cmake << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(TestCUDA LANGUAGES CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
EOF

cd /tmp && cmake -S . -B build_test 2>&1 | grep -E "(CUDA|Found)"

echo -e "\n📍 System package CUDA versions:"
dpkg -l | grep -i cuda || echo "No CUDA packages via apt"

echo -e "\n📍 Current symlinks:"
ls -la /usr/local/cuda* 2>/dev/null