#!/usr/bin/env python3

import argparse
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from geopy.distance import distance as geopy_distance
from geopy import Point
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_merged_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index("second")
    return df

def get_gps_data(df, sec):
    try:
        row = df.loc[sec]
        return row['lat'], row['long'], row['speed_mph'], row['gpx_time']
    except KeyError:
        return None, None, None, None

def compute_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def compute_bearing(prev, curr):
    delta = curr - prev
    angle_rad = np.arctan2(delta[1], delta[0])
    angle_deg = (np.degrees(angle_rad) + 360) % 360
    return angle_deg

def estimate_distance(bbox_height, frame_height, camera_height_m=1.7):
    if bbox_height <= 0:
        return None
    relative_size = bbox_height / frame_height
    approx_distance = camera_height_m / relative_size
    return round(approx_distance, 2)

def offset_latlon(base_lat, base_lon, bearing_deg, dist_meters):
    try:
        origin = Point(base_lat, base_lon)
        destination = geopy_distance(meters=dist_meters).destination(origin, bearing_deg)
        return destination.latitude, destination.longitude
    except:
        return None, None

def compute_object_speed(pixel_dist, ref_speed, ref_pixels=100):
    """Compute speed based on reference GPS speed and relative movement."""
    if ref_pixels == 0:
        return 0.0
    return (pixel_dist / ref_pixels) * ref_speed

def process_group(directory, model_path):
    dir_path = Path(directory)
    video_file = next((f for f in dir_path.iterdir() if f.suffix.lower() == ".mp4"), None)
    merged_csv = dir_path / "merged_output.csv"

    if not video_file or not merged_csv.exists():
        print(f"Missing .mp4 or merged_output.csv in {dir_path}")
        return

    gps_df = load_merged_csv(merged_csv)
    print(f"GPS second range: {gps_df.index.min()} to {gps_df.index.max()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    previous_centroids = defaultdict(lambda: None)
    output_rows = []

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        second = int(frame_idx // fps)
        lat, lon, ref_speed, timestamp = get_gps_data(gps_df, second)
        if None in (lat, lon, ref_speed, timestamp):
            print(f"Skipping frame {frame_idx} — missing GPS data at second {second}")
            continue

        print(f"Frame {frame_idx}: GPS ref_speed={ref_speed:.2f}, GPS=({lat},{lon})")

        results = model.track(source=frame, persist=True, verbose=False, conf=0.1)[0]
        if results is None or results.boxes is None or len(results.boxes) == 0:
            print(f"Frame {frame_idx}: 0 detections")
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None
        classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else None

        if ids is None or len(ids) != len(boxes) or classes is None:
            print(f"Frame {frame_idx}: Tracker returned incomplete data")
            continue

        print(f"Frame {frame_idx}: {len(boxes)} tracked detections")

        for i, bbox in enumerate(boxes):
            try:
                obj_id = int(ids[i])
                cls = classes[i]
                obj_type = model.names[cls] if cls < len(model.names) else "unknown"

                x1, y1, x2, y2 = bbox
                centroid = compute_centroid((x1, y1, x2, y2))
                height = y2 - y1

                prev = previous_centroids[obj_id]
                previous_centroids[obj_id] = centroid

                pixel_dist = np.linalg.norm(centroid - prev) if prev is not None else 0

                # Estimate object speed (mph)
                # Assume 100 pixels == 1 GPS mph as scaling factor, adjust if needed
                object_speed = compute_object_speed(pixel_dist, ref_speed, ref_pixels=100)

                if prev is not None:
                    bearing = compute_bearing(prev, centroid)
                else:
                    bearing = None

                distance = estimate_distance(height, frame.shape[0])

                if bearing is not None and distance is not None:
                    obj_lat, obj_lon = offset_latlon(lat, lon, bearing, distance)
                else:
                    obj_lat, obj_lon = None, None

                output_rows.append({
                    "Frame": frame_idx,
                    "Seconds": second,
                    "Object ID": obj_id,
                    "Object Type": obj_type,
                    "Object Speed (mph)": round(object_speed, 2),
                    "Bearing (deg)": round(bearing, 2) if bearing is not None else None,
                    "Distance (m)": distance,
                    "Object Lat": obj_lat,
                    "Object Lon": obj_lon,
                    "Cam Lat": lat,
                    "Cam Lon": lon,
                    "Time": timestamp
                })

                print(f"Row: Frame {frame_idx}, Obj {obj_id}, Type {obj_type}, Speed {object_speed:.2f} mph, Bearing {bearing}, Dist {distance}, GPS ({lat}, {lon})")

            except Exception as e:
                print(f"Error processing track at frame {frame_idx}, index {i}: {e}")
                continue

        print(f"Processed frame {frame_idx + 1}/{frame_count}", end='\r')

    cap.release()
    df_out = pd.DataFrame(output_rows)
    output_file = dir_path / f"enhanced_output_{video_file.stem}.csv"
    df_out.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file} ({len(output_rows)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, help="Directory with .mp4 and merged_output.csv")
    parser.add_argument("-m", "--model", required=False, default="yolo11x.pt", help="Path to YOLOv11x model")
    args = parser.parse_args()

    process_group(args.directory, args.model)

