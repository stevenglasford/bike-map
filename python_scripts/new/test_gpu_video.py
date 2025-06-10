import os
import sys
import time

def test_gpu_video_decoding():
    print("=== GPU Video Decoding Test ===")
    
    VIDEO_PATH = "/workspace/temp_video_257734894283657216.MP4"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        print("Available files in /workspace:")
        for f in os.listdir("/workspace"):
            print(f"  - {f}")
        return False
    
    try:
        import decord
        from decord import VideoReader, gpu, cpu
        print(f"✅ Decord version: {decord.__version__}")
        
        # Test CPU baseline first
        print("\n--- CPU Baseline Test ---")
        start_time = time.time()
        vr_cpu = VideoReader(VIDEO_PATH, ctx=cpu(0))
        cpu_frame = vr_cpu[0]
        cpu_time = time.time() - start_time
        print(f"✅ CPU decoding: {cpu_frame.shape} in {cpu_time:.4f}s")
        
        # Test GPU decoding
        print("\n--- GPU Decoding Test ---")
        start_time = time.time()
        ctx = gpu(0)
        vr_gpu = VideoReader(VIDEO_PATH, ctx=ctx)
        gpu_frame = vr_gpu[0]
        gpu_time = time.time() - start_time
        print(f"✅ GPU decoding: {gpu_frame.shape} in {gpu_time:.4f}s")
        
        # Performance comparison
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n--- Performance Results ---")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Test multiple frames for throughput
        print("\n--- Throughput Test (10 frames) ---")
        start_time = time.time()
        frames = []
        for i in range(min(10, len(vr_gpu))):
            frames.append(vr_gpu[i])
        batch_time = time.time() - start_time
        fps = 10 / batch_time
        print(f"✅ GPU batch decoding: 10 frames in {batch_time:.4f}s ({fps:.1f} fps)")
        
        print(f"\n🎉 SUCCESS! GPU video decoding is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_video_decoding()
    sys.exit(0 if success else 1)