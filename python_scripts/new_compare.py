import os
import cv2
import torch
import numpy as np
import gpxpy
import math
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from dtaidistance import dtw
from geopy.distance import geodesic


def load_video_features(video_path, yolo_model):
    cap = cv2.VideoCapture(str(video_path))
    prev_gray = None
    optical_speeds = []
    bearings = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx = np.median(flow[..., 0])
            dy = np.median(flow[..., 1])
            speed = math.sqrt(dx ** 2 + dy ** 2)
            bearing = math.degrees(math.atan2(dy, dx)) % 360

            optical_speeds.append(speed)
            bearings.append(bearing)
            timestamps.append(t)

        prev_gray = gray

    cap.release()
    return {
        "timestamps": timestamps,
        "speeds": optical_speeds,
        "bearings": bearings,
    }


def parse_gpx_features(gpx_path):
    with open(gpx_path, "r") as f:
        gpx = gpxpy.parse(f)

    speeds = []
    bearings = []
    timestamps = []
    latlons = []

    for track in gpx.tracks:
        for segment in track.segments:
            prev_point = None
            for point in segment.points:
                latlons.append((point.latitude, point.longitude))
                if prev_point:
                    distance_m = geodesic(
                        (prev_point.latitude, prev_point.longitude),
                        (point.latitude, point.longitude),
                    ).meters
                    time_delta = (point.time - prev_point.time).total_seconds()
                    if time_delta <= 0:
                        continue
                    speed = distance_m / time_delta
                    bearing = calculate_bearing(
                        prev_point.latitude,
                        prev_point.longitude,
                        point.latitude,
                        point.longitude,
                    )

                    speeds.append(speed)
                    bearings.append(bearing)
                    timestamps.append(point.time.timestamp())

                prev_point = point

    return {
        "timestamps": timestamps,
        "speeds": speeds,
        "bearings": bearings,
        "latlons": latlons,
    }


def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(dLon)
    brng = math.atan2(y, x)
    return (math.degrees(brng) + 360) % 360


def compute_similarity(video_feats, gpx_feats):
    if len(video_feats["speeds"]) < 2 or len(gpx_feats["speeds"]) < 2:
        return float("inf")

    v_speeds = np.array(video_feats["speeds"])
    v_bearings = np.array(video_feats["bearings"])
    g_speeds = np.array(gpx_feats["speeds"])
    g_bearings = np.array(gpx_feats["bearings"])

    v_combined = (v_speeds - v_speeds.mean()) / (v_speeds.std() + 1e-6) +                  (v_bearings - v_bearings.mean()) / (v_bearings.std() + 1e-6)
    g_combined = (g_speeds - g_speeds.mean()) / (g_speeds.std() + 1e-6) +                  (g_bearings - g_bearings.mean()) / (g_bearings.std() + 1e-6)

    v_combined = np.nan_to_num(v_combined)
    g_combined = np.nan_to_num(g_combined)

    dist = dtw.distance(v_combined.tolist(), g_combined.tolist())
    return dist


def match_video_to_all_gpx(video_path, gpx_folder, output_path="comparison_all_results.json", yolo_weights="yolo11x.pt"):
    yolo_model = YOLO(yolo_weights)
    video_feats = load_video_features(video_path, yolo_model)

    gpx_files = list(Path(gpx_folder).rglob("*.gpx"))
    matches = []

    for gpx_file in gpx_files:
        try:
            gpx_feats = parse_gpx_features(gpx_file)
            score = compute_similarity(video_feats, gpx_feats)
            matches.append((str(gpx_file), score))
        except Exception as e:
            print(f"Failed to process {gpx_file}: {e}")
            matches.append((str(gpx_file), float("inf")))

    matches.sort(key=lambda x: x[1])
    output = {
        "video": str(video_path),
        "matches": [{"gpx": gpx, "score": score} for gpx, score in matches]
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Done. {len(matches)} files compared. Best match: {matches[0] if matches else 'None'}")


# Example usage
if __name__ == "__main__":
    video_path = "../dtemp_video_249173973948960768/test/temp_video_257743256610148352.MP4"
    gpx_folder = "../dtemp_video_249173973948960768/test/gps"
    output_path = "match_results.json"
    match_video_to_all_gpx(video_path, gpx_folder)