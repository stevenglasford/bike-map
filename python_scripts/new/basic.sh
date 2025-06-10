ffmpeg -hwaccel cuda -i "/home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4" -vf scale_cuda=640:360 -r 2 -f rawvideo -pix_fmt rgb24 -t 5 test_output.raw

# Check if it created output
ls -la test_output.raw

# Test 2: See if we can get frame count
ffmpeg -hwaccel cuda -i "/home/preston/penis/panoramics/playground/temp_video_257734894283657216.MP4" -vf scale_cuda=640:360 -r 2 -f null - -t 10