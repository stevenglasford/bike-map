from nvidia_decoder import NVIDIAVideoDecoder
import time

def test_video(video_path):
    decoder = NVIDIAVideoDecoder(gpu_ids=[0])
    
    print(f"Testing: {video_path}")
    start = time.time()
    
    frames, fps, duration, indices = decoder.decode_video_batch(
        video_path, sample_rate=2.0, target_size=(640, 360)
    )
    
    elapsed = time.time() - start
    
    if frames is not None:
        print(f"✓ SUCCESS: {duration:.1f}s video processed in {elapsed:.2f}s")
        print(f"  Speed: {duration/elapsed:.1f}x real-time")
        print(f"  Frames: {len(frames)}")
    else:
        print("✗ FAILED")

# Test with your video
test_video("/home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4")
