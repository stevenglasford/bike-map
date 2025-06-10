#!/bin/bash

# Copy test script to workspace

cp simple_debug_test.py ~/penis/panoramics/playground/

# Run Docker container with GPU and volume

docker run --rm --gpus all --runtime=nvidia   \
-v /home/preston/penis/panoramics/playground:/workspace   \
decord-gpu-test python3 simple_debug_test.py