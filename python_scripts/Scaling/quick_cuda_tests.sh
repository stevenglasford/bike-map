#!/bin/bash
# Quick tests to find working CUDA scaling

echo "🔍 QUICK CUDA SCALING TESTS"
echo "================================="

# Test 1: Check available CUDA filters
echo "1️⃣ Available CUDA filters:"
ffmpeg -filters 2>/dev/null | grep -i cuda

echo -e "\n2️⃣ Available NPP filters:"
ffmpeg -filters 2>/dev/null | grep -i npp

echo -e "\n3️⃣ Available scale filters:"
ffmpeg -filters 2>/dev/null | grep -i scale

# Test 2: Basic CUDA scaling tests
echo -e "\n🧪 TESTING CUDA SCALING METHODS..."

echo "Test 2.1: scale_cuda"
ffmpeg -hwaccel cuda -f lavfi -i testsrc=duration=1:size=1280x720:rate=1 \
       -vf scale_cuda=640:360 -f null - 2>&1 | head -5

echo -e "\nTest 2.2: scale_npp"  
ffmpeg -hwaccel cuda -f lavfi -i testsrc=duration=1:size=1280x720:rate=1 \
       -vf scale_npp=640:360 -f null - 2>&1 | head -5

echo -e "\nTest 2.3: hwupload + scale_cuda + hwdownload"
ffmpeg -hwaccel cuda -f lavfi -i testsrc=duration=1:size=1280x720:rate=1 \
       -vf hwupload_cuda,scale_cuda=640:360,hwdownload -f null - 2>&1 | head -5

echo -e "\nTest 2.4: Alternative scale_cuda syntax"
ffmpeg -hwaccel cuda -f lavfi -i testsrc=duration=1:size=1280x720:rate=1 \
       -vf scale_cuda=w=640:h=360 -f null - 2>&1 | head -5

echo -e "\nTest 2.5: nvresize filter"
ffmpeg -hwaccel cuda -f lavfi -i testsrc=duration=1:size=1280x720:rate=1 \
       -vf nvresize=640:360 -f null - 2>&1 | head -5

# Test 3: Check your specific video
if [ "$1" ]; then
    echo -e "\n🎬 TESTING WITH YOUR VIDEO: $1"
    
    echo "Test 3.1: CUDA + regular scale (known working)"
    time ffmpeg -hwaccel cuda -i "$1" -vf scale=640:360 -r 2 -f null - -t 5 2>/dev/null
    
    echo -e "\nTest 3.2: Try scale_cuda on real video"
    ffmpeg -hwaccel cuda -i "$1" -vf scale_cuda=640:360 -r 2 -f null - -t 2 2>&1 | head -10
    
    echo -e "\nTest 3.3: Try hwupload method on real video"
    ffmpeg -hwaccel cuda -i "$1" -vf hwupload_cuda,scale_cuda=640:360,hwdownload -r 2 -f null - -t 2 2>&1 | head -10
fi

echo -e "\n================================="
echo "Look for SUCCESS messages above!"