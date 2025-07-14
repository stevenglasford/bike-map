export OPENCV_FFMPEG_CAPTURE_OPTIONS="buffer_size;8388608;tcp_nodelay;1"
export OPENCV_FFMPEG_READ_ATTEMPTS=4096
python -c "
import cv2
cap = cv2.VideoCapture('~/penis/panoramics/playground/temp_video_258120568170090496(1).MP4', cv2.CAP_FFMPEG)
print('FFmpeg backend available:', cap.isOpened())
cap.release()
"