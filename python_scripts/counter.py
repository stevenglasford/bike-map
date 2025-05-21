#!/usr/bin/env python3

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import defaultdict

# Constants
MOVEMENT_THRESHOLD = 20  # Pixels/sec
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
model = YOLO("yolo11x.pt").to(device)

# Tracker setup with vectorized Euclidean distance
def create_tracker(threshold):
    return Tracker(
        distance_function="euclidean",
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

def load_merged_output_csv(csv_path):
    df = pd.read_csv(csv_path)
    return {int(row['second']): (row['lat'], row['long']) for _, row in df.iterrows()}

def extract_detections(results, class_id):
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == class_id:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append(Detection(
                points=np.array([[cx.item(), cy.item()]]),
                scores=np.array([conf.item()])
            ))
    return detections

def resolve_video_path(input_path):
    if os.path.isdir(input_path):
        mp4_files = [f for f in os.listdir(input_path) if f.lower().endswith('.mp4')]
        if not mp4_files:
            raise FileNotFoundError(f"No .mp4 file found in directory: {input_path}")
        return os.path.join(input_path, mp4_files[0])
    elif os.path.isfile(input_path) and input_path.lower().endswith('.mp4'):
        return input_path
    else:
        raise FileNotFoundError(f"Invalid input path or unsupported format: {input_path}")

def process_video(input_path):
    global gps_data
    try:
        video_path = resolve_video_path(input_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(video_dir, "merged_output.csv")

    if not os.path.exists(csv_path):
        print(f"[WARN] Missing merged_output.csv in {video_dir}")
        return

    gps_data = load_merged_output_csv(csv_path)
    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    object_class_ids = {'person': 0, 'bicycle': 1, 'car': 2, 'bus': 5}

    while True:
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
                cx, cy = t.estimate[0]
                history[label][t.id].append((frame_count, cx, cy, cam_lat, cam_lon))

        frame_count += 1

    cap.release()
    return generate_csv(video_name, video_dir)

def compute_motion(entries):
    if len(entries) < 2:
        return "idle"
    dists = [
        np.linalg.norm(np.array(entry[1:3]) - np.array(prev[1:3]))
        for entry, prev in zip(entries[1:], entries[:-1])
    ]
    return "moving" if np.mean(dists) > MOVEMENT_THRESHOLD else "idle"

def generate_csv(base_name, output_dir):
    summary = []
    for label, objs in history.items():
        for oid, entries in objs.items():
            motion = compute_motion(entries)
            seen_frames = len(entries)
            last_lat, last_lon = entries[-1][3], entries[-1][4]
            summary.append({
                "object_id": f"{label}_{oid}",
                "type": label,
                "motion": motion,
                "frames_seen": seen_frames,
                "last_lat": last_lat,
                "last_lon": last_lon
            })

    df = pd.DataFrame(summary)
    out_csv = os.path.join(output_dir, f"{base_name}_object_counts.csv")
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved counts to {out_csv}")
    return out_csv

def run_batch(paths):
    for p in paths:
        process_video(p)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python object_counter_with_merged_csv.py <video-or-folder> [...]")
    else:
        run_batch(sys.argv[1:])