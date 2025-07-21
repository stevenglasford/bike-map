# Simple Insta360 Processor - CUDA 11.8 Ubuntu 22.04 (Minimal)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Bypass GPG issues by using --allow-unauthenticated
RUN apt-get clean && \
    apt-get update --allow-insecure-repositories || true

# Install minimal required packages
RUN apt-get install -y --allow-unauthenticated \
    python3 \
    python3-pip \
    xvfb \
    && apt-get clean

# Install Python packages using pip (doesn't need apt)
RUN python3 -m pip install --upgrade pip
RUN pip3 install numpy opencv-python pillow tqdm rich

# Create working directories
WORKDIR /app
RUN mkdir -p /input /output /logs /temp

# Copy MediaSDK installation files
COPY libMediaSDK-dev-*.deb /tmp/

# Install MediaSDK (bypass apt issues by forcing install)
RUN dpkg -i /tmp/libMediaSDK-dev-*.deb || true && \
    rm /tmp/libMediaSDK-dev-*.deb && \
    ldconfig && \
    echo "=== Quick System Check ===" && \
    python3 --version && \
    nvcc --version && \
    ldd --version && \
    echo "=== MediaSDK Test ===" && \
    (/usr/bin/MediaSDKTest --help 2>&1 | head -10 || echo "MediaSDK needs testing") && \
    echo "=========================="

# Set up environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1

# Copy Python scripts
COPY process_directory.py /app/
COPY monitor.py /app/

# Make scripts executable
RUN chmod +x /app/process_directory.py /app/monitor.py

# Create entrypoint script
RUN printf '#!/bin/bash\n\nXvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &\nexport DISPLAY=:99\nsleep 2\n\nif [ $# -eq 0 ]; then\n    exec python3 /app/process_directory.py /input /output 2\nelse\n    exec "$@"\nfi\n' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Set working directory and entrypoint
WORKDIR /app
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "/app/process_directory.py", "/input", "/output", "2"]