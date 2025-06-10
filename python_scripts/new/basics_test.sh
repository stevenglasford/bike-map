ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -c:v libx264 -f mp4 /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 -y
ffmpeg -hwaccel cuda -c:v h264_cuvid -i /home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4 -f null -

# Test 4: Check what's available
ffmpeg -hwaccels
ffmpeg -decoders | grep cuvid