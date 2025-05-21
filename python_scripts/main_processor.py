#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess
import logging
import time
from pathlib import Path
from multiprocessing import Process
import numpy as np
import pandas as pd
import cv2
import gpxpy

# Directory of this script (assuming other scripts are here)
script_dir = Path(__file__).parent.resolve()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_gpx_file(gpx_path):
    # Parse GPX file: output DataFrame with lat, lon, seconds, speed, accel
    points = []
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
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

def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in miles
    R = 3958.8
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def compute_motion_accel(video_path):
    # Compute video motion acceleration per second (optical flow)
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    logging.info(f"compute_motion_accel: Using CUDA: {use_cuda}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps) if fps > 0 else 0
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
                flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(
                    5, 0.5, False, 15, 3, 5, 1.2, 0)
                flow = flow_gpu.calc(gpu_prev, gpu_curr, None).download()
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.linalg.norm(flow, axis=2)
            motion_energy.append(np.mean(magnitude))
        prev_gray = gray
    cap.release()
    accel_series = [0.0] + list(np.diff(motion_energy))
    return accel_series

def align_by_accel(video_accel, gpx_accel):
    # Cross-correlate accel to find best offset
    n = min(len(video_accel), len(gpx_accel))
    if n <= 1:
        return 0
    v = np.array(video_accel[:n])
    g = np.array(gpx_accel[:n])
    corr = np.correlate(v - v.mean(), g - g.mean(), mode='full')
    offset = np.argmax(corr) - (n - 1)
    return offset

def export_aligned_csv(video_accel, gpx_df, shift, output_path):
    # Export aligned CSV mapping video second to lat, lon, speed
    rows = []
    for i, a in enumerate(video_accel):
        gpx_i = i + shift
        if 0 <= gpx_i < len(gpx_df):
            g = gpx_df.iloc[gpx_i]
            rows.append({
                "second": i,
                "lat": g["lat"],
                "lon": g["lon"],
                "speed_mph": g["speed"]
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info(f"Aligned CSV saved to {output_path}")

def merge_csv_with_gpx(aligned_csv, gpx_path, output_csv):
    # Merge aligned CSV with GPX timestamps
    csv_df = pd.read_csv(aligned_csv)
    gpx_records = []
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    gpx_records.append({
                        "gpx_lat": pt.latitude,
                        "gpx_lon": pt.longitude,
                        "gpx_time": pt.time
                    })
    gpx_df = pd.DataFrame(gpx_records)
    merged = []
    for _, row in csv_df.iterrows():
        lat = row['lat']; lon = row['lon']
        match = gpx_df[(abs(gpx_df.gpx_lat - lat) < 1e-4) & (abs(gpx_df.gpx_lon - lon) < 1e-4)]
        gpx_time = match.iloc[0]['gpx_time'] if not match.empty else None
        merged.append({
            "second": row["second"],
            "lat": lat,
            "long": lon,
            "speed_mph": row["speed_mph"],
            "gpx_time": gpx_time
        })
    pd.DataFrame(merged).to_csv(output_csv, index=False)
    logging.info(f"Merged output CSV saved to {output_csv}")

def find_best_gpx_match(video_path, gpx_paths):
    # Find the best matching GPX file for the video by accel correlation
    video_accel = compute_motion_accel(video_path)
    if not video_accel:
        return None, None, None, None
    best_score = -np.inf
    best_overlap = 0
    best = (None, 0, None, None)
    for gpx_path in gpx_paths:
        try:
            gpx_df = parse_gpx_file(gpx_path)
            gpx_accel = gpx_df['accel'].tolist()
            if len(gpx_accel) < 2 or len(video_accel) < 2:
                continue
            offset = align_by_accel(video_accel, gpx_accel)
            # Determine overlap length
            if offset >= 0:
                corr_len = min(len(video_accel), len(gpx_accel) - offset)
            else:
                corr_len = min(len(video_accel) + offset, len(gpx_accel))
            if corr_len <= 10:
                continue
            if offset >= 0:
                v_slice = video_accel[:corr_len]
                g_slice = gpx_accel[offset:offset+corr_len]
            else:
                v_slice = video_accel[-offset:-offset+corr_len]
                g_slice = gpx_accel[:corr_len]
            if len(v_slice) > 1:
                score = np.corrcoef(v_slice, g_slice)[0,1]
            else:
                score = 0
            if not np.isnan(score) and corr_len > best_overlap:
                best_score = score
                best_overlap = corr_len
                best = (gpx_path, offset, video_accel, gpx_df)
        except Exception as e:
            logging.warning(f"Error matching {gpx_path.name}: {e}")
            continue
    best_gpx, best_offset, vid_accel, gpx_df = best
    if best_gpx:
        logging.info(f"Matched {video_path.name} to {best_gpx.name} (overlap={best_overlap}s, score={best_score:.3f})")
        return best_gpx, best_offset, vid_accel, gpx_df
    else:
        logging.error(f"No suitable GPX match found for {video_path.name}")
        return None, None, None, None

def process_video_pipeline(video_path, gpx_paths, input_dir, noise_profile=None):
    # Process an individual video: match GPX, align, merge, and run postprocessing
    try:
        video_path = Path(video_path)
        video_name = video_path.stem
        out_dir = Path(input_dir) / "matched" / f"d{video_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Processing video {video_path.name} in {out_dir}")

        # Match GPX file
        best_gpx, offset, video_accel, gpx_df = find_best_gpx_match(video_path, gpx_paths)
        if not best_gpx:
            return
        # Copy video and GPX to matched directory
        matched_video = out_dir / video_path.name
        matched_gpx = out_dir / best_gpx.name
        shutil.copy(video_path, matched_video)
        shutil.copy(best_gpx, matched_gpx)

        # Create aligned_output.csv
        aligned_csv = out_dir / "aligned_output.csv"
        export_aligned_csv(video_accel, gpx_df, offset, aligned_csv)

        # Create merged_output.csv
        merged_csv = out_dir / "merged_output.csv"
        merge_csv_with_gpx(aligned_csv, matched_gpx, merged_csv)

        # Run postprocessing scripts (logs errors if fail, continue pipeline)
        # counter.py
        logging.info(f"Running counter.py on {out_dir}")
        subprocess.run([sys.executable, str(script_dir/"counter.py"), str(out_dir)], check=False)

        # process_groups_yolo_enhanced.py
        logging.info(f"Running process_groups_yolo_enhanced.py on {out_dir}")
        subprocess.run([sys.executable, str(script_dir/"process_groups_yolo_enhanced.py"), 
                        "-d", str(out_dir), "-m", "yolo11x.pt"], check=False)

        # process_noise.py
        logging.info(f"Running process_noise.py on {out_dir}")
        noise_args = [sys.executable, str(script_dir/"process_noise.py"), "-d", str(out_dir)]
        if noise_profile:
            noise_args += ["-n", str(noise_profile)]
        subprocess.run(noise_args, check=False)

        # stoplight.py
        logging.info(f"Running stoplight.py on {out_dir}")
        subprocess.run([sys.executable, str(script_dir/"stoplight.py"), str(out_dir)], check=False)

        logging.info(f"Completed processing {video_path.name}")
    except Exception as e:
        logging.error(f"Failed to process {video_path.name}: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Parallel video-GPX processing pipeline.")
    parser.add_argument("input_dir", help="Directory with unsorted .mp4 and .gpx files")
    parser.add_argument("-n", "--noise-profile", help="Optional noise profile WAV file", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Find video and GPX files
    video_paths = sorted(input_dir.rglob("*.mp4")) + sorted(input_dir.rglob("*.MP4"))
    gpx_paths = sorted(input_dir.rglob("*.gpx")) + sorted(input_dir.rglob("*.GPX"))
    if not video_paths:
        logging.error("No .mp4 files found.")
        sys.exit(1)
    if not gpx_paths:
        logging.error("No .gpx files found.")
        sys.exit(1)
    logging.info(f"Found {len(video_paths)} videos and {len(gpx_paths)} GPX files.")

    # Prepare matched output directory
    matched_dir = input_dir / "matched"
    matched_dir.mkdir(exist_ok=True)

    # GPU parallel processing: one video per GPU
    num_gpus = 2
    processes = {}  # gpu_id -> process
    video_iter = iter(video_paths)

    while True:
        # Launch processes if GPUs free and videos remain
        for gpu_id in range(num_gpus):
            if gpu_id not in processes:
                try:
                    video_path = next(video_iter)
                except StopIteration:
                    break
                logging.info(f"Starting processing {video_path.name} on GPU {gpu_id}")
                # set environment for GPU usage
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                p = Process(target=process_video_pipeline, args=(video_path, gpx_paths, input_dir, args.noise_profile))
                p.start()
                processes[gpu_id] = p

        # Remove finished processes
        for gpu_id, proc in list(processes.items()):
            if not proc.is_alive():
                proc.join()
                logging.info(f"GPU {gpu_id} finished its task")
                processes.pop(gpu_id, None)

        # Break if all videos assigned and all processes done
        if not processes:
            try:
                # If there's another video, continue loop
                _ = next(video_iter)
                # If no exception, loop continues to assign remaining videos
                video_iter = iter([_])  # push back one and continue
                continue
            except StopIteration:
                break

        # Sleep briefly to avoid busy wait
        time.sleep(1)

    logging.info("All processing complete.")

if __name__ == "__main__":
    main()
