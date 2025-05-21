#!/usr/bin/env python3

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from math import atan2, radians, degrees, cos, sin

def detect_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 80) & (v > 100)
    hue = h[mask]
    if hue.size == 0:
        return 'unknown'
    hist = cv2.calcHist([hue.astype(np.uint8)], [0], None, [180], [0, 180])
    dominant = int(hist.argmax())
    if 0 <= dominant <= 10 or 160 <= dominant <= 180:
        return 'red'
    elif 15 <= dominant <= 35:
        return 'yellow'
    elif 36 <= dominant <= 89:
        return 'green'
    return 'unknown'

def angle_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = radians(lon2 - lon1)
    y = sin(dLon) * cos(radians(lat2))
    x = cos(radians(lat1)) * sin(radians(lat2)) - sin(radians(lat1)) * cos(radians(lat2)) * cos(dLon)
    brng = atan2(y, x)
    brng = degrees(brng)
    return (brng + 360) % 360

def add_bearing_column(df):
    bearings = []
    prev_lat = prev_lon = None
    for i, row in df.iterrows():
        lat = row["lat"]
        lon = row["long"]  # updated to match your column name
        if prev_lat is not None and prev_lon is not None:
            bearing = calculate_bearing(prev_lat, prev_lon, lat, lon)
        else:
            bearing = np.nan
        bearings.append(bearing)
        prev_lat, prev_lon = lat, lon
    df["bearing"] = bearings
    return df

def process_video(video_path, merged_df):
    model = YOLO("yolo11x.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = 0
    light_records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        second = frame_count // fps
        if second not in merged_df.index:
            frame_count += 1
            continue

        row = merged_df.loc[second]
        cam_bearing = row.get("bearing", None)
        if pd.isna(cam_bearing):
            frame_count += 1
            continue

        results = model(frame)[0]
        current_light = 'unknown'

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 9:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                rel_angle = ((cx / width) * 360) % 360

                if angle_diff(cam_bearing, rel_angle) <= 20:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        current_light = detect_light_color(roi)
                        break

        light_records.append({
            "stoplight": current_light,
            "second": second,
            "lat": row["lat"],
            "long": row["long"],  # matches your CSV
            "gpx_time": row["gpx_time"]
        })

        frame_count += 1

    cap.release()
    return light_records

def main(directory):
    video_files = glob.glob(os.path.join(directory, "*.mp4")) + glob.glob(os.path.join(directory, "*.MP4"))
    if not video_files:
        print("No video files found.")
        return

    merged_csv_path = os.path.join(directory, "merged_output.csv")
    if not os.path.exists(merged_csv_path):
        print(f"Missing merged_output.csv in {directory}")
        return

    merged_df = pd.read_csv(merged_csv_path)
    required_columns = {"second", "lat", "long", "gpx_time"}
    if not required_columns.issubset(merged_df.columns):
        print("merged_output.csv must contain: second, lat, long, gpx_time")
        return

    merged_df = add_bearing_column(merged_df)
    merged_df = merged_df.set_index("second")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing: {video_name}")
        records = process_video(video_path, merged_df)

        if records:
            df_out = pd.DataFrame(records)
            output_csv = os.path.join(directory, f"stoplights_{video_name}.csv")
            df_out.to_csv(output_csv, index=False)
            print(f"Saved to {output_csv}")
        else:
            print(f"No stoplight data found for {video_name}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stoplight_detection.py /path/to/video_directory")
    else:
        main(sys.argv[1])