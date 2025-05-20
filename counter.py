#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import os
import sys
import glob
from collections import defaultdict
from ultralytics import YOLO
from norfair import Detection, Tracker

# Constants
MOVEMENT_THRESHOLD = 20  # pixels/sec
FPS_SMOOTH_WINDOW = 30   # for averaging
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
model = YOLO("yolo11x.pt").to(device)

# Tracker setup
def create_tracker(threshold):
    return Tracker(
        distance_function=lambda a, b: np.linalg.norm(a.points - b.points),
        distance_threshold=threshold
    )

trackers = {
    'person': create_tracker(30),
    'bicycle': create_tracker(30),
    'car': create_tracker(40),
    'bus': create_tracker(40)
}

history = {
    'person': defaultdict(list),
    'bicycle': defaultdict(list),
    'car': defaultdict(list),
    'bus': defaultdict(list)
}

gps_data = {}  # second → (lat, lon)

# Load camera GPS
def load_gps_csv(gps_path):
    df = pd.read_csv(gps_path)
    return {int(row['second']): (row['lat'], row['lon']) for _, row in df.iterrows()}

# Helper to extract detections
def extract_detections(results, class_id):
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == class_id:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append(Detection(points=np.array([[cx.item(), cy.item()]]), scores=np.array([conf.item()])))
    return detections

# Main video processing
def process_video(video_path):
    global gps_data
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    gps_path = os.path.join(os.path.dirname(video_path), f"{base_name}_merged_output.csv")
    if not os.path.exists(gps_path):
        print(f"[WARN] Missing GPS file for {video_path}")
        return

    gps_data = load_gps_csv(gps_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    object_class_ids = {'person': 0, 'bicycle': 1, 'car': 2, 'bus': 5}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        second = int(frame_count / fps)
        cam_lat, cam_lon = gps_data.get(second, (None, None))

        results = model(frame)[0]
        for label, class_id in object_class_ids.items():
            dets = extract_detections(results, class_id)
            tracked = trackers[label].update(dets)
            for t in tracked:
                oid = t.id
                cx, cy = t.estimate[0]
                history[label][oid].append((frame_count, cx, cy, cam_lat, cam_lon))

        frame_count += 1

    cap.release()
    return generate_csv(base_name)

# Motion classifier
def compute_motion(entries):
    if len(entries) < 2:
        return "idle"
    dists = [np.linalg.norm(np.array(entry[1:3]) - np.array(prev[1:3])) for entry, prev in zip(entries[1:], entries[:-1])]
    avg_motion = np.mean(dists)
    return "moving" if avg_motion > MOVEMENT_THRESHOLD else "idle"

# CSV writer
def generate_csv(base_name):
    summary = []
    for label in history:
        for oid, entries in history[label].items():
            motion_status = compute_motion(entries)
            seen_frames = len(entries)
            last_lat, last_lon = entries[-1][3], entries[-1][4]  # last known location
            summary.append({
                "object_id": f"{label}_{oid}",
                "type": label,
                "motion": motion_status,
                "frames_seen": seen_frames,
                "last_lat": last_lat,
                "last_lon": last_lon
            })

    df = pd.DataFrame(summary)
    output_csv = f"{base_name}_object_counts.csv"
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved object counts to {output_csv}")
    return output_csv

# Batch runner
def run_batch(video_paths):
    for path in video_paths:
        process_video(path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python object_counter.py /path/to/videos/*.mp4")
    else:
        video_files = []
        for pattern in sys.argv[1:]:
            video_files.extend(glob.glob(pattern))
        run_batch(video_files)
