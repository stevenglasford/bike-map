from simple_nvidia_decoder import SimpleNVIDIADecoder
import time
import sys

def test_video(video_path):
    print(f"Testing Simple NVIDIA decoder with: {video_path}")
    print("=" * 60)
    
    decoder = SimpleNVIDIADecoder(gpu_ids=[0])
    
    print("\nStarting video decode test...")
    start = time.time()
    
    frames, fps, duration, indices = decoder.decode_video_batch(
        video_path, sample_rate=2.0, target_size=(640, 360)
    )
    
    elapsed = time.time() - start
    
    print("=" * 60)
    if frames is not None:
        print(f"✓ SUCCESS!")
        print(f"  Video duration: {duration:.1f}s")
        print(f"  Original FPS: {fps:.1f}")
        print(f"  Processing time: {elapsed:.2f}s")
        print(f"  Speed: {duration/elapsed:.1f}x real-time")
        print(f"  Frames extracted: {len(frames)}")
        print(f"  Frame tensor shape: {frames.shape}")
        
        # Performance assessment
        if elapsed < duration:
            speedup = duration / elapsed
            print(f"  🚀 EXCELLENT: {speedup:.1f}x faster than real-time!")
        else:
            print(f"  ⚠️  Processing slower than real-time")
            
    else:
        print(f"✗ FAILED to decode video")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_video(sys.argv[1])
    else:
        print("Usage: python test_simple.py /path/to/video.mp4")
        print("\nOr try with a video from current directory:")
        import glob
        videos = glob.glob("*.mp4") + glob.glob("*.MP4")
        if videos:
            print(f"Found video: {videos[0]}")
            test_video(videos[0])
        else:
            print("No videos found in current directory")