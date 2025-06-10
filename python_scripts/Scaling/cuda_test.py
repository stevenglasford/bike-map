#!/usr/bin/env python3
"""
Fixed CUDA Scaling Tester - with proper pixel format handling
"""

import subprocess
import sys

def run_ffmpeg_test(description, cmd):
    """Run ffmpeg command and return success status"""
    print(f"\n🧪 Testing: {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   ✅ SUCCESS!")
            return True
        else:
            print(f"   ❌ FAILED: {result.stderr.strip().split(chr(10))[-2] if result.stderr else 'Unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"   💥 ERROR: {e}")
        return False

def main():
    print("🔍 FIXED CUDA SCALING TESTS")
    print("=" * 50)
    
    # Check if we have a video file argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"📁 Using input file: {input_file}")
    else:
        input_file = None
        print("📁 Using synthetic test input")
    
    success_count = 0
    total_tests = 0
    
    # Test configurations that should work
    tests = [
        {
            "name": "CUDA scale with YUV420P input",
            "cmd": [
                "ffmpeg", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:duration=1:rate=1",
                "-pix_fmt", "yuv420p",
                "-vf", "hwupload_cuda,scale_cuda=640:360,hwdownload",
                "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
            ]
        },
        {
            "name": "NPP scale with YUV420P input",
            "cmd": [
                "ffmpeg", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:duration=1:rate=1",
                "-pix_fmt", "yuv420p",
                "-vf", "hwupload_cuda,scale_npp=640:360,hwdownload",
                "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
            ]
        },
        {
            "name": "CUDA scale with NV12 format",
            "cmd": [
                "ffmpeg", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:duration=1:rate=1",
                "-pix_fmt", "nv12",
                "-vf", "hwupload_cuda,scale_cuda=640:360,hwdownload",
                "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
            ]
        },
        {
            "name": "Format conversion + CUDA scale",
            "cmd": [
                "ffmpeg", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:duration=1:rate=1",
                "-vf", "format=nv12,hwupload_cuda,scale_cuda=640:360,hwdownload,format=nv12",
                "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
            ]
        },
        {
            "name": "CUDA device initialization test",
            "cmd": [
                "ffmpeg", "-f", "lavfi", "-i", "color=red:size=320x240:duration=1",
                "-vf", "format=nv12,hwupload_cuda,hwdownload",
                "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
            ]
        }
    ]
    
    # Add real file tests if input provided
    if input_file:
        real_file_tests = [
            {
                "name": "Real file - Hardware decode + CUDA scale",
                "cmd": [
                    "ffmpeg", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-i", input_file,
                    "-vf", "scale_cuda=640:360",
                    "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
                ]
            },
            {
                "name": "Real file - CPU decode + CUDA scale",
                "cmd": [
                    "ffmpeg", "-i", input_file,
                    "-vf", "format=nv12,hwupload_cuda,scale_cuda=640:360,hwdownload",
                    "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
                ]
            },
            {
                "name": "Real file - NPP scaling",
                "cmd": [
                    "ffmpeg", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-i", input_file,
                    "-vf", "scale_npp=640:360",
                    "-frames:v", "1", "-f", "null", "-", "-v", "quiet"
                ]
            }
        ]
        tests.extend(real_file_tests)
    
    # Run all tests
    for test in tests:
        total_tests += 1
        if run_ffmpeg_test(test["name"], test["cmd"]):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 CUDA SCALING TEST RESULTS")
    print("=" * 50)
    print(f"✅ Successful tests: {success_count}/{total_tests}")
    
    if success_count > 0:
        print("🎉 CUDA scaling is working!")
        print("\n💡 Working patterns:")
        print("   1. Use -pix_fmt yuv420p or nv12 for synthetic inputs")
        print("   2. Use format=nv12,hwupload_cuda,scale_cuda,hwdownload")
        print("   3. For real files: -hwaccel cuda -hwaccel_output_format cuda")
        print("\n🚀 Example working command:")
        print("   ffmpeg -hwaccel cuda -hwaccel_output_format cuda \\")
        print("          -i input.mp4 -vf 'scale_cuda=1920:1080' \\")
        print("          -c:v h264_nvenc output.mp4")
    else:
        print("❌ CUDA scaling still not working")
        print("💡 Try checking:")
        print("   1. NVIDIA driver version: nvidia-smi")
        print("   2. CUDA installation: nvcc --version")
        print("   3. FFmpeg CUDA support: ffmpeg -hwaccels")
    
    print(f"\n📊 Usage: python {sys.argv[0]} /path/to/video.mp4")
    print("   (to test with actual video file)")

if __name__ == "__main__":
    main()