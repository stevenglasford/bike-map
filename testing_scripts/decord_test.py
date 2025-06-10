from decord import VideoReader, gpu
import time

video_path = "/home/preston/penis/panoramics/playground/temp_video_257743398239211520.MP4"  # use a valid local video file

start = time.time()
try:
    vr = VideoReader(video_path, ctx=gpu(0))
    print("✅ GPU decoding is working")
    print(f"Video has {len(vr)} frames")

    frame = vr[0]  # Force decode a frame
    print("Successfully decoded first frame on GPU")

except Exception as e:
    print("❌ GPU decoding failed:", e)

end = time.time()
print(f"Decode test took {end - start:.2f} seconds")
