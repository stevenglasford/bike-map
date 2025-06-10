from decord import VideoReader, gpu
import time

VIDEO_PATH = "/workspace/temp_video_257734894283657216.MP4"

try:
    ctx = gpu(0)
    print("✅ GPU context initialized:", ctx)

    vr = VideoReader(VIDEO_PATH, ctx=ctx)
    print("✅ Video loaded using GPU context")

    start = time.time()
    frame = vr[0]  # Decode first frame
    end = time.time()

    print("✅ First frame type:", type(frame))
    print("✅ Decoding time:", round(end - start, 4), "seconds")

except Exception as e:
    print("❌ GPU acceleration test failed:", str(e))