#!/bin/bash

echo "=== Setting up Decord with GPU Video Decoding ==="

# Create the CUDA lib64 directory structure that CMake expects
mkdir -p /usr/local/cuda/lib64
mkdir -p /usr/local/cuda/include

# Copy NVDEC library to where CMake expects it
if [ -f /tmp/libnvcuvid.so ]; then
    cp /tmp/libnvcuvid.so /usr/local/cuda/lib64/libnvcuvid.so
    chmod 755 /usr/local/cuda/lib64/libnvcuvid.so
    
    # Also copy to system lib directory
    cp /tmp/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so
    chmod 755 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
    
    # Create version symlinks
    ln -sf /usr/local/cuda/lib64/libnvcuvid.so /usr/local/cuda/lib64/libnvcuvid.so.1
    ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
    
    echo "✅ NVDEC library installed"
else
    echo "❌ NVDEC library not found at /tmp/libnvcuvid.so"
    exit 1
fi

# Create symlinks for CUDA toolkit
ln -sf /usr/bin/nvcc /usr/local/cuda/bin/nvcc 2>/dev/null || true
ln -sf /usr/include/cuda* /usr/local/cuda/include/ 2>/dev/null || true

ldconfig

# Install PyCUDA
echo "Installing PyCUDA..."
pip install pycuda

# Build Decord with NVDEC enabled
echo "Building Decord with GPU video decoding..."
cd /opt/decord
rm -rf build
mkdir -p build && cd build

# Configure with explicit CUDA paths and NVDEC enabled
cmake .. \
    -DUSE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DFFMPEG_DIR=/usr \
    -DCUDA_NVDEC=ON \
    -DCUDA_NVCUVID_LIBRARY=/usr/local/cuda/lib64/libnvcuvid.so \
    -DAVFORMAT_LIBRARIES=/usr/lib/x86_64-linux-gnu/libavformat.so \
    -DAVCODEC_LIBRARIES=/usr/lib/x86_64-linux-gnu/libavcodec.so \
    -DAVFILTER_LIBRARIES=/usr/lib/x86_64-linux-gnu/libavfilter.so \
    -DAVUTIL_LIBRARIES=/usr/lib/x86_64-linux-gnu/libavutil.so \
    -DAVDEVICE_LIBRARIES=/usr/lib/x86_64-linux-gnu/libavdevice.so \
    -DSWRESAMPLE_LIBRARIES=/usr/lib/x86_64-linux-gnu/libswresample.so \
    -DSWSCALE_LIBRARIES=/usr/lib/x86_64-linux-gnu/libswscale.so

if [ $? -eq 0 ]; then
    echo "CMake configuration successful, building..."
    make -j$(nproc)
    
    if [ $? -eq 0 ]; then
        make install
        cp libdecord.so /usr/local/lib/
        ldconfig
        
        # Install Python bindings
        echo "Installing Python bindings..."
        cd ../python
        python3 setup.py install
        
        echo "✅ Decord with GPU video decoding setup complete!"
    else
        echo "❌ Decord build failed"
        exit 1
    fi
else
    echo "❌ CMake configuration failed"
    exit 1
fi
