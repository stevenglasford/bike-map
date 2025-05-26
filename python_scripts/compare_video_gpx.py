
import os
import cv2
import torch
import numpy as np
import gpxpy
import math
import json
import folium
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
    yolo_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Landmark detection
        results = yolo_model(frame, verbose=False)
        objects = results[0].boxes.cls.tolist() if results else []
        yolo_detections.append((t, objects))

        # Motion estimation
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
        "yolo": yolo_detections,
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
        return float("inf"), []

    v_speeds = np.array(video_feats["speeds"])
    v_bearings = np.array(video_feats["bearings"])
    g_speeds = np.array(gpx_feats["speeds"])
    g_bearings = np.array(gpx_feats["bearings"])

    # Normalize sequences
    v_combined = (v_speeds - v_speeds.mean()) / (v_speeds.std() + 1e-6) +                  (v_bearings - v_bearings.mean()) / (v_bearings.std() + 1e-6)
    g_combined = (g_speeds - g_speeds.mean()) / (g_speeds.std() + 1e-6) +                  (g_bearings - g_bearings.mean()) / (g_bearings.std() + 1e-6)

    dist = dtw.distance(v_combined.tolist(), g_combined.tolist())
    return dist, []


def visualize_match(gpx_feats, score, output_html="map_match_debug.html"):
    latlons = gpx_feats["latlons"]
    if not latlons:
        return
    m = folium.Map(location=latlons[0], zoom_start=15)
    folium.PolyLine(locations=latlons, color='blue', weight=5, tooltip=f"Match Score: {score:.2f}").add_to(m)
    folium.Marker(latlons[0], tooltip="Start").add_to(m)
    folium.Marker(latlons[-1], tooltip="End").add_to(m)
    m.save(output_html)


def compare_video_to_gpx(video_path, gpx_path, output_json="comparison_result.json", output_map="map_match_debug.html", yolo_weights="yolov11x.pt"):
    yolo_model = YOLO(yolo_weights)
    video_feats = load_video_features(video_path, yolo_model)
    gpx_feats = parse_gpx_features(gpx_path)
    score, _ = compute_similarity(video_feats, gpx_feats)

    result = {
        "video": str(video_path),
        "gpx": str(gpx_path),
        "score": score
    }

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    visualize_match(gpx_feats, score, output_map)


# Example usage
if __name__ == "__main__":
    video_path = "example.mp4"
    gpx_path = "example.gpx"
    compare_video_to_gpx(video_path, gpx_path)
