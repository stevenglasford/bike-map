#!/usr/bin/env python3

import argparse
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_merged_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index("second")
    return df

def get_gps_data(df, sec):
    try:
        row = df.loc[sec]
        return row['gpx_lat'], row['gpx_lon'], row['gpx_speed'], row['gpx_time']
    except KeyError:
        return None, None, None, None

def compute_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def pixel_to_mph(pixel_dist, ref_speed, ref_pixels=100):
    if ref_pixels == 0:
        return 0
    return (pixel_dist / ref_pixels) * ref_speed

def process_group(directory, model_path):
    dir_path = Path(directory)
    video_file = next((f for f in dir_path.iterdir() if f.suffix.lower() == ".mp4"), None)
    merged_csv = dir_path / "merged_output.csv"

    if not video_file or not merged_csv.exists():
        print(f"Missing .mp4 or merged_output.csv in {dir_path}")
        return

    gps_df = load_merged_csv(merged_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    deepsort = DeepSort(max_age=30, n_init=2)  # confirm tracks quickly

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
        if lat is None:
            print(f"No GPS data for second {second}")
            continue

        results = model(frame, verbose=False, conf=0.1)[0]
        detections = results.boxes
        print(f"Frame {frame_idx}: {len(detections) if detections else 0} detections")

        if detections is None or detections.xyxy is None or len(detections) == 0:
            continue

        try:
            # Attempt to extract bboxes and confs
            bboxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()

            if bboxes.ndim == 1:
                bboxes = np.expand_dims(bboxes, axis=0)
            if confs.ndim == 0:
                confs = np.expand_dims(confs, axis=0)

            if bboxes.shape[1] != 4 or bboxes.shape[0] != confs.shape[0]:
                print(f"Skipping malformed detections at frame {frame_idx}")
                continue

            print(f"Raw bboxes: {bboxes}")
            print(f"Type of bbox at index {i}: {type(bbox)} - value: {bbox}")
            print(f"Raw confs: {confs}")
            print(f"Frame {frame_idx}: {len(bboxes)} bboxes, {len(confs)} confidences")

        
            detections_list = []

            try:
                bboxes = detections.xyxy.cpu().numpy()
                confs = detections.conf.cpu().numpy()
            
                if bboxes.ndim == 1:
                    bboxes = np.expand_dims(bboxes, axis=0)
                if confs.ndim == 0:
                    confs = np.expand_dims(confs, axis=0)
            
                if bboxes.shape[1] != 4 or bboxes.shape[0] != confs.shape[0]:
                    print(f"Skipping malformed detections at frame {frame_idx}")
                    continue
            
                print(f"Raw bboxes: {bboxes}")
                print(f"Raw confs: {confs}")
                print(f"Frame {frame_idx}: {len(bboxes)} bboxes, {len(confs)} confidences")
            
                for i in range(len(bboxes)):
                    try:
                        bbox = bboxes[i]
                        conf = confs[i]
            
                        if isinstance(bbox, (float, int)):
                            print(f"Skipped non-iterable bbox at frame {frame_idx}, index {i}: {bbox}")
                            continue
            
                        if isinstance(bbox, np.ndarray) and bbox.shape == (4,):
                            x1, y1, x2, y2 = map(float, bbox)
                        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x1, y1, x2, y2 = map(float, bbox)
                        else:
                            print(f"Invalid bbox structure at frame {frame_idx}, index {i}: {bbox}")
                            continue
            
                        detections_list.append([x1, y1, x2, y2, float(conf)])
            
                    except Exception as e:
                        print(f"Error parsing individual bbox at frame {frame_idx}, index {i}: {e}")
                        continue
            
            except Exception as e:
                print(f"Detection formatting error at frame {frame_idx}: {e}")
                continue
        
            if not detections_list:
                print(f"No valid detections after parsing at frame {frame_idx}")
                continue

            tracks = deepsort.update_tracks(detections_list, frame=frame)
            confirmed = [t for t in tracks if t.is_confirmed()]
            print(f"Frame {frame_idx}: {len(confirmed)} confirmed tracks")

        except Exception as e:
            print(f"Detection formatting error at frame {frame_idx}: {e}")
            continue

        for track in tracks:
            if not track.is_confirmed():
                continue
            obj_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            centroid = compute_centroid((x1, y1, x2, y2))

            prev = previous_centroids[obj_id]
            pixel_dist = np.linalg.norm(centroid - prev) if prev is not None else 0
            previous_centroids[obj_id] = centroid

            speed_mph = pixel_to_mph(pixel_dist, ref_speed)

            output_rows.append({
                "Frame": frame_idx,
                "Seconds": second,
                "Object ID": obj_id,
                "Object Speed (mph)": round(speed_mph, 2),
                "Lat": lat,
                "Long": lon,
                "Time": timestamp
            })

        print(f"Processed frame {frame_idx + 1}/{frame_count}", end='\r')

    cap.release()
    df_out = pd.DataFrame(output_rows)
    output_file = dir_path / f"speed_table_{video_file.stem}.csv"
    df_out.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, help="Directory with .mp4 and merged_output.csv")
    parser.add_argument("-m", "--model", required=False, default="yolo11x.pt", help="Path to YOLOv11x model")
    args = parser.parse_args()

    process_group(args.directory, args.model)
