#!/bin/bash

set -e

echo "🚿 Cleaning previous build..."
rm -rf opencv_build
mkdir opencv_build
cd opencv_build 

#echo "📥 Cloning OpenCV and contrib (4.x latest)..."
#git clone --branch 4.12.0 https://github.com/opencv/opencv.git
#git clone --branch 4.12.0 https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
git checkout 4.12.0
cd ..

cd opencv_contrib
git checkout 4.12.0
cd ..


cd opencv
rm -rf build
mkdir build
cd build

# Use gcc-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10

# CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

echo "📦 CUDA version:"
nvcc --version
echo ""
echo "🧠 GPU info:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

# Set CUDA architecture manually to 12.9
CUDA_ARCH_BIN="12"

# Optional: NVIDIA Video Codec SDK
VIDEO_CODEC_SDK_PATH="/home/preston/bike-map/python_scripts/new/Nvidia-shit/Video_Codec_SDK_13.0.19"
if [ -d "$VIDEO_CODEC_SDK_PATH" ]; then
    echo "✅ Found Video Codec SDK: $VIDEO_CODEC_SDK_PATH"
    NVCUVID_FLAGS="-D WITH_NVCUVID=ON -D NVCUVID_ROOT=$VIDEO_CODEC_SDK_PATH"
    NVCUVENC_FLAGS="-D WITH_NVCUVENC=ON -D NVCUVENC_ROOT=$VIDEO_CODEC_SDK_PATH"
else
    echo "⚠️  Video Codec SDK not found. Skipping..."
    NVCUVID_FLAGS="-D WITH_NVCUVID=OFF"
    NVCUVENC_FLAGS="-D WITH_NVCUVENC=OFF"
fi

echo "🛠 Running CMake..."
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=$(pwd)/../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D WITH_CUBLAS=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D OPENCV_DNN_CUDA=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CUDA_ARCH_BIN="12.0"\
      -D CUDA_ARCH_PTX="12.0" \
      $NVCUVID_FLAGS \
      $NVCUVENC_FLAGS \
      -D BUILD_opencv_python3=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_world=OFF \
      -D BUILD_opencv_sfm=ON \
      -D WITH_OPENGL=ON \
      -D WITH_TBB=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_FFMPEG=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      -D CMAKE_C_COMPILER=/usr/bin/gcc-10 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-10 \
      -D CUDA_HOST_COMPILER=/usr/bin/g++-10 \
      ..
      #-D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
      

echo "✅ CMake configuration completed."

echo "🧱 Building OpenCV with $(nproc) cores..."
make -j$(nproc)

echo "📦 Installing..."
sudo make install
sudo ldconfig

echo "✅ Installation complete."

echo "🔍 Verifying CUDA support..."
python3 -c "import cv2; print('OpenCV version:', cv2.__version__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"