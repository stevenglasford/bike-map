#!/usr/bin/env python3

import os
import sys

def main():
    print("=== Testing GPU Decord ===")
    
    # Test Decord import
    try:
        import decord
        from decord import gpu, VideoReader
        print(f"✅ Decord version: {decord.__version__}")
    except Exception as e:
        print(f"❌ Decord import failed: {e}")
        return False
    
    # Test GPU context
    try:
        ctx = gpu(0)
        print(f"✅ GPU context: {ctx}")
    except Exception as e:
        print(f"❌ GPU context failed: {e}")
        return False
    
    # Test video decoding
    VIDEO_PATH = "/workspace/temp_video_257734894283657216.MP4"
    if os.path.exists(VIDEO_PATH):
        try:
            vr = VideoReader(VIDEO_PATH, ctx=ctx)
            frame = vr[0]
            print(f"🎉 SUCCESS! GPU video decoding works!")
            print(f"Frame shape: {frame.shape}")
            return True
        except Exception as e:
            print(f"❌ Video decoding failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"✅ GPU context works! (Video file not found for full test)")
        return True

if __name__ == "__main__":
    success = main()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
