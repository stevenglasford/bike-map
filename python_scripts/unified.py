#!/usr/bin/env python3
import os
import cv2
import gpxpy
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from scipy.signal import correlate
from ultralytics import YOLO
from collections import defaultdict
from geopy.distance import distance as geopy_distance
from geopy import Point
from multiprocessing import Process, Lock

# --------------------- Utility Functions ---------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def parse_gpx(gpx_path):
    """Parse GPX file and return DataFrame with timestamp, lat, lon, seconds, speed, accel."""
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.time.replace(tzinfo=None), pt.latitude, pt.longitude))
    df = pd.DataFrame(points, columns=["gpx_time", "lat", "lon"])
    df['seconds'] = (df['gpx_time'] - df['gpx_time'].iloc[0]).dt.total_seconds()
    df['speed'] = [0.0] + [
        haversine(df['lat'].iloc[i-1], df['lon'].iloc[i-1],
                  df['lat'].iloc[i], df['lon'].iloc[i]) /
        ((df['seconds'].iloc[i] - df['seconds'].iloc[i-1]) / 3600)
        for i in range(1, len(df))
    ]
    df['accel'] = [0.0] + list(np.diff(df['speed']))
    return df

def compute_motion_accel(video_path):
    """Compute a 1-second-sampled acceleration series from video motion using optical flow."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    motion_energy = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    for sec in range(1, duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_cuda:
            gpu_prev = cv2.cuda_GpuMat()
            gpu_curr = cv2.cuda_GpuMat()
            gpu_prev.upload(prev_gray)
            gpu_curr.upload(gray)
            flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(
                5, 0.5, False, 15, 3, 5, 1.2, 0)
            flow = flow_gpu.calc(gpu_prev, gpu_curr, None).download()
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        motion_energy.append(np.mean(mag))
        prev_gray = gray
    cap.release()
    accel = [0.0] + list(np.diff(motion_energy))
    return accel

def align_by_accel(video_accel, gpx_accel):
    """Align video accel with GPX accel by cross-correlation. Returns best offset (seconds)."""
    n = min(len(video_accel), len(gpx_accel))
    v = np.array(video_accel[:n]) - np.mean(video_accel[:n])
    g = np.array(gpx_accel[:n]) - np.mean(gpx_accel[:n])
    corr = correlate(v, g, mode='full')
    offset = np.argmax(corr) - (n - 1)
    return offset

def offset_latlon(base_lat, base_lon, bearing_deg, dist_meters):
    """Compute lat/lon offset from a point, given bearing and distance."""
    try:
        origin = Point(base_lat, base_lon)
        dest = geopy_distance(meters=dist_meters).destination(origin, bearing_deg)
        return dest.latitude, dest.longitude
    except:
        return None, None

def compute_centroid(bbox):
    """Compute centroid of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def compute_bearing(prev, curr):
    """Compute bearing in degrees from prev point to curr point."""
    delta = curr - prev
    angle_rad = np.arctan2(delta[1], delta[0])
    return (np.degrees(angle_rad) + 360) % 360

def process_video(video_path, merged_csv, output_dir, gpu_id):
    """Phase 2 processing: YOLO tracking, audio noise, stoplight detection."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    output_dir = Path(output_dir)
    # Load GPS-merged CSV
    gps_df = pd.read_csv(merged_csv).set_index("second")
    # Initialize YOLO tracker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11x.pt").to(device)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_centroids = defaultdict(lambda: None)
    track_rows = []
    # Loop through frames
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        second = int(frame_idx // fps)
        if second not in gps_df.index:
            continue
        lat = gps_df.at[second, "lat"]
        lon = gps_df.at[second, "long"]
        ref_speed = gps_df.at[second, "speed_mph"]
        timestamp = gps_df.at[second, "gpx_time"]
        # Run YOLO tracker on the frame
        results = model.track(source=frame, persist=True, verbose=False, conf=0.2)[0]
        if results is None or results.boxes is None or len(results.boxes) == 0:
            continue
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None
        classes = (results.boxes.cls.cpu().numpy().astype(int) 
                   if results.boxes.cls is not None else None)
        if ids is None or classes is None:
            continue
        # Process each detection
        for i, bbox in enumerate(boxes):
            obj_id = int(ids[i])
            cls = int(classes[i])
            obj_type = model.names[cls] if cls < len(model.names) else "unknown"
            centroid = compute_centroid(bbox)
            height = bbox[3] - bbox[1]
            prev = prev_centroids[obj_id]
            prev_centroids[obj_id] = centroid
            pixel_dist = np.linalg.norm(centroid - prev) if prev is not None else 0
            # Estimate speed (scale by ref_speed)
            obj_speed = (pixel_dist / 100) * ref_speed  # assume 100 pixels = ref_speed
            bearing = compute_bearing(prev, centroid) if prev is not None else None
            dist_m = (abs(height) / frame.shape[0]) * 50  # crude distance estimation
            if bearing is not None:
                obj_lat, obj_lon = offset_latlon(lat, lon, bearing, dist_m * 1.0)
            else:
                obj_lat, obj_lon = None, None
            track_rows.append({
                "Frame": frame_idx,
                "Second": second,
                "ObjectID": obj_id,
                "Type": obj_type,
                "Speed_mph": round(obj_speed, 2),
                "Bearing": round(bearing, 2) if bearing is not None else None,
                "Distance_m": round(dist_m, 2),
                "ObjLat": obj_lat,
                "ObjLon": obj_lon,
                "CamLat": lat,
                "CamLon": lon,
                "Time": timestamp
            })
            # Optional: stoplight detection (simple check for traffic light color)
            if obj_type.lower() == "traffic light":
                # Crop ROI and check for red vs green (placeholder logic)
                x1, y1, x2, y2 = [int(v) for v in bbox]
                roi = frame[y1:y2, x1:x2]
                if roi.size != 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    # define red color range and count pixels
                    mask_red = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    mask_green = cv2.inRange(hsv, (50, 50, 50), (70, 255, 255))
                    red_count = cv2.countNonZero(mask_red)
                    green_count = cv2.countNonZero(mask_green)
                    light_state = "red" if red_count > green_count else "green"
                    # Save or print stoplight event
                    track_rows[-1]["LightState"] = light_state
    cap.release()
    # Save tracking results
    track_df = pd.DataFrame(track_rows)
    track_df.to_csv(output_dir / f"tracking_output_{video_path.stem}.csv", index=False)
    # Audio noise analysis (simple frame-level amplitude)
    try:
        import librosa
        y, sr = librosa.load(str(video_path), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        sec_bins = int(duration)
        noise_levels = []
        for sec in range(sec_bins):
            start = sec * sr
            end = min(len(y), (sec+1) * sr)
            segment = y[start:end]
            if len(segment) == 0:
                break
            rms = np.sqrt(np.mean(segment**2))
            db = 20 * np.log10(rms + 1e-6)
            noise_levels.append({"second": sec, "noise_dB": round(db, 2)})
        noise_df = pd.DataFrame(noise_levels)
        noise_df.to_csv(output_dir / "audio_noise.csv", index=False)
    except ImportError:
        print("librosa not installed, skipping audio analysis.")
    return True

def main(video_dir, gpx_dir):
    video_dir = Path(video_dir)
    gpx_dir = Path(gpx_dir)
    matched_dir = video_dir / "matched"
    matched_dir.mkdir(exist_ok=True)
    log_file = matched_dir / "processed_videos.txt"
    # Load processed log
    processed = set()
    if log_file.exists():
        processed = set(line.strip() for line in open(log_file))
    # Collect video and GPX files
    video_files = sorted(video_dir.rglob("*.mp4")) + sorted(video_dir.rglob("*.MP4"))
    gpx_files = sorted(gpx_dir.rglob("*.gpx")) + sorted(gpx_dir.rglob("*.GPX"))
    tasks = []
    # Phase 1: Match videos to GPX
    for video_path in video_files:
        video_stem = video_path.stem
        if video_path.name in processed or (matched_dir / f"d{video_stem}").exists():
            print(f"Skipping already-processed video: {video_path.name}")
            continue
        print(f"Matching for video: {video_path.name}")
        # Compute video acceleration
        vid_accel = compute_motion_accel(video_path)
        if not vid_accel:
            print(f"Failed to compute motion for {video_path.name}, skipping.")
            continue
        # Find best matching GPX
        best_score = -np.inf
        best_offset = 0
        best_gpx = None
        best_gpx_df = None
        for gpx_path in gpx_files:
            try:
                gpx_df = parse_gpx(gpx_path)
            except Exception as e:
                print(f"Error parsing GPX {gpx_path.name}: {e}")
                continue
            gpx_accel = gpx_df['accel'].tolist()
            if len(gpx_accel) < 2 or len(vid_accel) < 2:
                continue
            offset = align_by_accel(vid_accel, gpx_accel)
            corr_len = min(len(vid_accel), len(gpx_accel) - offset)
            if corr_len <= 10:
                continue
            score = np.corrcoef(vid_accel[:corr_len], 
                                gpx_accel[offset:offset+corr_len])[0, 1]
            if not np.isnan(score) and corr_len > 0 and score > best_score:
                best_score = score
                best_offset = offset
                best_gpx = gpx_path
                best_gpx_df = gpx_df
        if best_gpx is None:
            print(f"No suitable GPX match for {video_path.name}, skipping.")
            continue
        # Prepare output directory
        new_dir = matched_dir / f"d{video_stem}"
        new_dir.mkdir(exist_ok=True)
        # Copy video and GPX into the matched folder
        video_copy = new_dir / video_path.name
        gpx_copy = new_dir / best_gpx.name
        if not video_copy.exists():
            os.system(f"cp \"{video_path}\" \"{video_copy}\"")
        if not gpx_copy.exists():
            os.system(f"cp \"{best_gpx}\" \"{gpx_copy}\"")
        # Create merged_output.csv
        merged_rows = []
        offset = best_offset
        for sec, a in enumerate(vid_accel):
            idx = sec + offset
            if 0 <= idx < len(best_gpx_df):
                row = best_gpx_df.iloc[idx]
                merged_rows.append({
                    "second": sec,
                    "lat": row["lat"],
                    "long": row["lon"],
                    "speed_mph": row["speed"],
                    "gpx_time": row["gpx_time"].strftime("%Y-%m-%dT%H:%M:%S")
                })
        merged_df = pd.DataFrame(merged_rows)
        merged_df.to_csv(new_dir / "merged_output.csv", index=False)
        print(f"Matched {video_path.name} -> {best_gpx.name} (score={best_score:.3f})")
        tasks.append((video_copy, new_dir / "merged_output.csv", new_dir))
    # Phase 2: Process matched videos in parallel (2 GPUs)
    processes = []
    gpu_ids = [0, 1]
    next_task = 0
    lock = Lock()
    # Helper to start a process with GPU assignment
    def start_task(task, gpu):
        p = Process(target=process_video, args=(task[0], task[1], task[2], gpu))
        p.start()
        return p
    while next_task < len(tasks) or processes:
        # Launch up to 2 tasks
        while len(processes) < 2 and next_task < len(tasks):
            gpu = gpu_ids[next_task % 2]
            task = tasks[next_task]
            p = start_task(task, gpu)
            processes.append((p, gpu, task))
            next_task += 1
        # Check for finished processes
        for p, gpu, task in processes:
            if not p.is_alive():
                p.join()
                print(f"Completed processing for {task[0].name} on GPU {gpu}")
                with lock:
                    with open(log_file, "a") as f:
                        f.write(f"{task[0].name}\n")
        # Remove joined processes
        processes = [(p,g,t) for (p,g,t) in processes if p.is_alive()]
    print("All processing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Video-GPX processing with matching and YOLO tracking")
    parser.add_argument("-v", "--video_dir", required=True,
                        help="Directory of input video files")
    parser.add_argument("-g", "--gpx_dir", required=True,
                        help="Directory of input GPX files")
    args = parser.parse_args()
    main(args.video_dir, args.gpx_dir)