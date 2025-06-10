# Copy test script to workspace
cp test_gpu_video.py ~/penis/panoramics/playground/

# Run the comprehensive test
echo "Running comprehensive GPU video decoding test..."
docker run --rm --gpus all --runtime=nvidia \
  --privileged \
  -v /usr/lib/x86_64-linux-gnu:/host-libs:ro \
  -v /home/preston/penis/panoramics/playground:/workspace \
  -e LD_LIBRARY_PATH="/host-libs:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
  decord-gpu-test python3 /workspace/test_gpu_video.py
  