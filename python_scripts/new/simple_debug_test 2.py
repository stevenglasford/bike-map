#!/usr/bin/env python3

import os
import sys

def main():
    print("=== Decord GPU Debug Test ===")
    
    # Check if video file exists
    VIDEO_PATH = "/workspace/temp_video_257734894283657216.MP4"
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found: {VIDEO_PATH}")
        print("Available files in /workspace:")
        try:
            files = os.listdir("/workspace")
            for f in files:
                print(f"  - {f}")
        except:
            print("  Could not list files")
        return

    print(f"Video file found: {VIDEO_PATH}")

    # Test 1: Basic Decord import
    try:
        import decord
        print(f"Decord imported successfully (version: {decord.__version__})")
    except Exception as e:
        print(f"Failed to import decord: {e}")
        return

    # Test 2: CPU decoding first
    try:
        from decord import VideoReader
        print("Testing CPU decoding...")
        vr = VideoReader(VIDEO_PATH)
        frame = vr[0]
        print(f"CPU decoding works. Frame shape: {frame.shape}")
    except Exception as e:
        print(f"CPU decoding failed: {e}")
        return

    # Test 3: Check GPU availability  
    try:
        from decord import gpu
        print("Testing GPU context creation...")
        ctx = gpu(0)
        print(f"GPU context created: {ctx}")
    except Exception as e:
        print(f"GPU context creation failed: {e}")
        return

    # Test 4: GPU decoding
    try:
        print("Testing GPU decoding...")
        vr_gpu = VideoReader(VIDEO_PATH, ctx=ctx)
        print("VideoReader with GPU context created")
        
        frame_gpu = vr_gpu[0]
        print(f"GPU decoding successful! Frame shape: {frame_gpu.shape}")
        
    except Exception as e:
        print(f"GPU decoding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()