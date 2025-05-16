import os
import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from norfair import Detection, Tracker
import gpxpy
import gpxpy.gpx
import math


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def load_gpx(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
    return [(pt.time, pt.latitude, pt.longitude)
            for track in gpx.tracks
            for seg in track.segments
            for pt in seg.points]


def map_gps_to_seconds(csv_df, gpx_points):
    gps_map = {}
    for _, row in csv_df.iterrows():
        sec = int(row['seconds'])
        target_time = datetime.utcfromtimestamp(row['seconds'])
        closest = min(gpx_points, key=lambda x: abs((x[0] - target_time).total_seconds()))
        gps_map[sec] = {"lat": closest[1], "lon": closest[2], "time": closest[0].isoformat()}
    return gps_map


def process_group(group_path, model_path="yolo11x.pt"):
    group_path = Path(group_path)
    video_file = next(group_path.glob("*.mp4"))
    gpx_file = next(group_path.glob("*.gpx"))
    csv_file = next(group_path.glob("*.csv"))
    video_name = video_file.stem

    gps_df = pd.read_csv(csv_file)
    gps_map = map_gps_to_seconds(gps_df, load_gpx(gpx_file))

    model = YOLO(model_path)
    deepsort = DeepSort()
    bytetrack = Tracker(distance_function='euclidean', distance_threshold=30)

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    output_rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        sec = int(frame_idx / fps)
        if sec not in gps_map:
            frame_idx += 1
            continue

        yolo_out = model(frame)[0]
        bboxes, confs = [], []

        for det in yolo_out.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            confs.append(float(det.conf[0]))
            bboxes.append([x1, y1, x2 - x1, y2 - y1])

        tracks = deepsort.update_tracks(bboxes, confs, frame=frame)

        norfair_detections = []
        for box in bboxes:
            xc, yc = box[0] + box[2] / 2, box[1] + box[3] / 2
            norfair_detections.append(Detection(points=np.array([[xc, yc]])))

        tracked_objs = bytetrack.update(detections=norfair_detections)

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            speed = 0.0
            for obj in tracked_objs:
                if obj.id == tid and obj.vel is not None:
                    speed = obj.vel
                    break

            row = {
                "Seconds": sec,
                "Object id": tid,
                "Object speed": round(speed, 2),
                "Lat": gps_map[sec]['lat'],
                "Long": gps_map[sec]['lon'],
                "Time": gps_map[sec]['time']
            }
            output_rows.append(row)

        frame_idx += 1

    cap.release()
    out_csv = group_path / f"speed_table_{video_name}.csv"
    pd.DataFrame(output_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and analyze object speeds using video, GPX, and metadata.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory containing .mp4, .gpx, and .csv")
    parser.add_argument("-m", "--model", type=str, default="yolo11x.pt", help="YOLO model path (default: yolo11x.pt)")
    args = parser.parse_args()

    process_group(args.directory, args.model)