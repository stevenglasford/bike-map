import cv2
import numpy as np
import gpxpy
import pandas as pd
import math
from scipy.signal import correlate
from datetime import timedelta
import argparse
import os

def dtw_align(signal1, signal2):
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    distance, path = fastdtw(signal1, signal2, dist=euclidean)
    if not path:
        return 0, 0.0
    offsets = [j - i for i, j in path]
    offset = int(np.median(offsets))
    score = 1 / (1 + distance / len(path))  # Higher is better
    return offset, score

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def parse_gpx(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.time.replace(tzinfo=None), pt.latitude, pt.longitude))

    df = pd.DataFrame(points, columns=["timestamp", "lat", "lon"])
    df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    df['speed'] = [0.0] + [
        haversine(df['lat'][i-1], df['lon'][i-1], df['lat'][i], df['lon'][i]) /
        ((df['seconds'][i] - df['seconds'][i-1]) / 3600)
        for i in range(1, len(df))
    ]
    df['accel'] = [0.0] + list(np.diff(df['speed']))
    return df

def compute_motion_accel(video_path):
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"Using CUDA: {use_cuda}")

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)

    motion_energy = []
    prev_gray = None

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            if use_cuda:
                gpu_prev = cv2.cuda_GpuMat()
                gpu_curr = cv2.cuda_GpuMat()
                gpu_prev.upload(prev_gray)
                gpu_curr.upload(gray)
                flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
                flow = flow_gpu.calc(gpu_prev, gpu_curr, None).download()
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.linalg.norm(flow, axis=2)
            motion_energy.append(np.mean(magnitude))

        prev_gray = gray

    accel_series = [0.0] + list(np.diff(motion_energy))
    return accel_series

def align_by_accel(video_accel, gpx_accel):
    n = min(len(video_accel), len(gpx_accel))
    v = np.array(video_accel[:n])
    g = np.array(gpx_accel[:n])
    corr = correlate(v - v.mean(), g - g.mean(), mode='full')
    offset = np.argmax(corr) - (n - 1)
    return offset

def export_csv(video_accel, gpx_df, shift_index, output_path):
    rows = []
    for i, a in enumerate(video_accel):
        gpx_i = i + shift_index
        if 0 <= gpx_i < len(gpx_df):
            g = gpx_df.iloc[gpx_i]
            rows.append({
                "second": i,
                "lat": g["lat"],
                "lon": g["lon"],
                "speed_mph": g["speed"]
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"CSV saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align a video with a GPX file based on acceleration.")
    parser.add_argument("--video", "-v", required=True, help="Path to the panoramic video file.")
    parser.add_argument("--gpx", "-g", required=True, help="Path to the GPX file.")
    parser.add_argument("--output", "-o", default="aligned_output.csv", help="Path to output CSV file.")

    args = parser.parse_args()

    video_path = args.video
    gpx_path = args.gpx
    output_csv = args.output

    print("[1] Parsing GPX...")
    gpx_df = parse_gpx(gpx_path)

    print("[2] Analyzing video...")
    video_accel = compute_motion_accel(video_path)

    print("[3] Aligning...")
    offset = align_by_accel(video_accel, list(gpx_df['accel']))
    print(f"Best offset: {offset} seconds")

    print("[4] Exporting...")
    export_csv(video_accel, gpx_df, offset, output_csv)
