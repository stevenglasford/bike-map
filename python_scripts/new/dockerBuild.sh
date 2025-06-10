#!/bin/bash

# Clean up any existing containers/images if needed

echo “Building Decord GPU Docker image…”

# Build Docker image

docker build -t decord-gpu-test .

# Check build success

if [ $? -eq 0 ]; then
echo “✅ Docker image built successfully!”
echo “Run with: ./runDocket.sh”
else
echo “❌ Docker build failed!”
exit 1
fi