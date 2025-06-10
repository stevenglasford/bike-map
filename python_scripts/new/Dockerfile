FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

COPY libnvcuvid.so.1 /usr/local/cuda/lib64/
COPY libnvcuvid.so /usr/local/cuda/lib64/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    python3-pip \
    git \
    cmake \
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libavdevice-dev \
    libswresample-dev \
    libswscale-dev \
    libtbb-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# -----------------------------
# Environment setup
# -----------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# -----------------------------
# Python dependencies
# -----------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel numpy cython pycuda

# -----------------------------
# Install Decord from source with GPU
# -----------------------------
WORKDIR /opt
RUN git clone --recursive https://github.com/dmlc/decord.git

# Optional: Add libnvcuvid manually (assumes you've added it to context under Nvidia-shit/)
COPY Nvidia-shit/libnvcuvid.so /usr/local/cuda/lib64/libnvcuvid.so
RUN chmod 755 /usr/local/cuda/lib64/libnvcuvid.so && \
    ln -sf /usr/local/cuda/lib64/libnvcuvid.so /usr/local/cuda/lib64/libnvcuvid.so.1 && \
    ln -sf /usr/local/cuda/lib64/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/local/cuda/lib64/libnvcuvid.so /tmp/libnvcuvid.so && \
    ldconfig

# Optional: symlinks to make FFmpeg easier to find
RUN ln -sf /usr/lib/x86_64-linux-gnu/libavformat.so /usr/lib/libavformat.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libavcodec.so /usr/lib/libavcodec.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libavfilter.so /usr/lib/libavfilter.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libavutil.so /usr/lib/libavutil.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libavdevice.so /usr/lib/libavdevice.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libswresample.so /usr/lib/libswresample.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libswscale.so /usr/lib/libswscale.so

# -----------------------------
# Build Decord with CUDA
# -----------------------------
WORKDIR /opt/decord
RUN mkdir -p build && cd build && \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cp libdecord.so /usr/local/lib/ && \
    ldconfig

# Install Python bindings
WORKDIR /opt/decord/python
RUN python3 setup.py install

# -----------------------------
# Default runtime
# -----------------------------
WORKDIR /workspace
CMD ["/bin/bash"]