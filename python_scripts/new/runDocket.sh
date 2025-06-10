#!/bin/bash

docker run --rm --gpus all --runtime=nvidia \
  -v /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1:/usr/local/cuda/lib64/libnvcuvid.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libnvcuvid.so:/usr/local/cuda/lib64/libnvcuvid.so \
  -v /home/preston/penis/panoramics/playground:/workspace \
  decord-gpu-test