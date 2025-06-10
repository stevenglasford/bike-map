#!/bin/bash

# 🎯 WORKING CUDA SCALING EXAMPLES
# The key is using compatible pixel formats and proper filter chains

echo "🔧 WORKING CUDA SCALING METHODS"
echo "================================"

# Method 1: Hardware decode + CUDA scale + Hardware encode
echo "Method 1: Full GPU pipeline (recommended)"
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
       -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 \
       -vf "scale_cuda=1920:1080" \
       -c:v h264_nvenc -preset fast \
       output_cuda_scaled.mp4

# Method 2: CPU decode + Upload to GPU + CUDA scale + Download
echo "Method 2: CPU to GPU pipeline"
ffmpeg -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 \
       -vf "format=nv12,hwupload_cuda,scale_cuda=1920:1080,hwdownload,format=nv12" \
       -c:v libx264 -preset fast \
       output_hybrid.mp4

# Method 3: Hardware decode + NPP scale
echo "Method 3: Using NPP (NVIDIA Performance Primitives)"
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
       -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 \
       -vf "scale_npp=1920:1080" \
       -c:v h264_nvenc -preset fast \
       output_npp_scaled.mp4

# Method 4: Test with synthetic input (proper format)
echo "Method 4: Synthetic test with correct format"
ffmpeg -f lavfi -i "testsrc2=size=3840x2160:duration=5:rate=30" \
       -pix_fmt nv12 \
       -vf "hwupload_cuda,scale_cuda=1920:1080,hwdownload" \
       -t 5 -y test_cuda_scale.mp4

# Method 5: Complex filter chain
echo "Method 5: Complex GPU processing"
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
       -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 \
       -vf "scale_cuda=1920:1080,hwdownload,format=nv12" \
       -c:v h264_nvenc -preset fast -profile:v high \
       -b:v 5M -maxrate 8M -bufsize 10M \
       output_complex.mp4

# 🧪 TESTING SCRIPT
echo ""
echo "🧪 QUICK TEST COMMANDS"
echo "======================"

# Test 1: Check if CUDA device is available
echo "Test 1: CUDA device detection"
ffmpeg -f lavfi -i color=red:size=1920x1080:duration=1 \
       -vf "hwupload_cuda,hwdownload" \
       -frames:v 1 -f null - 2>&1 | grep -E "(cuda|device)"

# Test 2: Working CUDA scale test
echo "Test 2: Simple working CUDA scale"
ffmpeg -f lavfi -i "testsrc2=size=1920x1080:duration=1:rate=1" \
       -pix_fmt yuv420p \
       -vf "hwupload_cuda,scale_cuda=640:360,hwdownload" \
       -frames:v 1 -f null - -v quiet && echo "✅ CUDA scaling WORKS!" || echo "❌ Still failing"

# Test 3: NPP scale test
echo "Test 3: NPP scale test"
ffmpeg -f lavfi -i "testsrc2=size=1920x1080:duration=1:rate=1" \
       -pix_fmt yuv420p \
       -vf "hwupload_cuda,scale_npp=640:360,hwdownload" \
       -frames:v 1 -f null - -v quiet && echo "✅ NPP scaling WORKS!" || echo "❌ NPP failing"

# 📊 PERFORMANCE COMPARISON
echo ""
echo "📊 PERFORMANCE COMPARISON SCRIPT"
echo "================================="

# Compare CPU vs CUDA vs NPP scaling
create_perf_test() {
    local input_file="$1"
    local output_size="$2"
    
    echo "Testing with: $input_file -> $output_size"
    
    # CPU scaling
    echo "CPU scaling:"
    time ffmpeg -i "$input_file" -vf "scale=$output_size" -f null - -v quiet
    
    # CUDA scaling
    echo "CUDA scaling:"
    time ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
                -i "$input_file" -vf "scale_cuda=$output_size" -f null - -v quiet
    
    # NPP scaling
    echo "NPP scaling:"
    time ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
                -i "$input_file" -vf "scale_npp=$output_size" -f null - -v quiet
}

# Usage: create_perf_test "/home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4" "1920:1080"

# 🎛️ QUALITY SETTINGS FOR CUDA SCALING
echo ""
echo "🎛️ QUALITY SETTINGS"
echo "==================="

# High quality CUDA scaling with different algorithms
echo "High quality bicubic:"
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
       -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 \
       -vf "scale_cuda=1920:1080:interp_algo=bicubic" \
       -c:v h264_nvenc -preset slow -crf 18 \
       output_hq.mp4

echo "Fast bilinear (default):"
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
       -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 \
       -vf "scale_cuda=1920:1080:interp_algo=bilinear" \
       -c:v h264_nvenc -preset fast \
       output_fast.mp4

# Available scaling algorithms for scale_cuda:
# - bilinear (default, fastest)
# - bicubic (higher quality, slower)
# - nearest (pixelated, very fast)
# - area (good for downscaling)

echo ""
echo "✅ Use these working examples with your actual video files!"