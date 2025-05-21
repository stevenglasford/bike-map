from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from strongsort.strong_sort import StrongSORT
import torch

print("Torch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Test YOLOv11x model loading
model = YOLO("yolo11x.pt")
model.to("cuda")
print("YOLOv11x loaded on GPU.")

# Dummy args for ByteTrack
class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    min_box_area = 10
    mot20 = False

bytetrack_tracker = BYTETracker(Args())
print("ByteTrack OK.")

# StrongSORT needs model_weights, device, fp16
# You can provide a real model path later; this just checks it loads
strongsort = StrongSORT(model_weights=None, device="cuda", fp16=False)
print("StrongSORT OK.")