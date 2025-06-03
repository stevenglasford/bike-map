rm -rf opencv_build
mkdir opencv_build
cd opencv_build

git clone -b 4.11.0 https://github.com/opencv/opencv.git
git clone -b 4.11.0 https://github.com/opencv/opencv_contrib.git

cd ~/opencv_build/opencv
mkdir build
cd build

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D BUILD_opencv_sfm=ON \
      -D BUILD_opencv_viz=ON \
      -D BUILD_opencv_ovis=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D OPENCV_DNN_CUDA=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_EXAMPLES=OFF \
      #-D CUDA_ARCH_BIN="12.0" \
      #-D CUDA_ARCH_PTX="12.0" \
      -D BUILD_opencv_world=ON \
      -D BUILD_TESTS=OFF \
      -D WITH_FFMPEG=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D GLOG_INCLUDE_DIR=/usr/include/glog \
      -D GLOG_LIBRARY=/usr/lib/x86_64-linux-gnu/libglog.so \
      -D BUILD_opencv_sfm=ON \
      -D BUILD_opencv_world=OFF \
      -D CMAKE_C_COMPILER=/usr/bin/gcc-10 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-10 \
      -D CUDA_HOST_COMPILER=/usr/bin/g++-10 \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      ..
      
make -j$(nproc)
sudo make install