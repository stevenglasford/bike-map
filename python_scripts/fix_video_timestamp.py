import argparse
import os
import shutil
import subprocess
import pandas as pd
import gpxpy
from pathlib import Path
import json
from datetime import datetime, timedelta

def find_files(directory):
    mp4_file = gpx_file = csv_file = None
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        lower = file.lower()
        if lower.endswith(".mp4") and not mp4_file:
            mp4_file = path
        elif lower.endswith(".gpx") and not gpx_file:
            gpx_file = path
        elif lower.endswith(".csv") and not csv_file:
            csv_file = path
    return mp4_file, gpx_file, csv_file


def get_first_gpx_time(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.time:
                        return point.time
    raise ValueError("No valid timestamp found in GPX file")

def get_video_codec(input_mp4):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "json", input_mp4
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    codec_info = json.loads(result.stdout)
    return codec_info["streams"][0]["codec_name"]

def correct_timestamp(input_mp4, output_mp4, start_time, codec):
    timestamp_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")

    if codec == "h265":
        encoder = "hevc_nvenc"
        pix_fmt = "p010le"
    else:
        encoder = "h264_nvenc"
        pix_fmt = "yuv420p"

    cmd = [
        "ffmpeg", "-hwaccel", "cuda",
        "-i", input_mp4,
        "-metadata", f"creation_time={timestamp_str}",
        "-c:v", encoder,
        "-pix_fmt", pix_fmt,
        "-c:a", "copy",
        "-y", str(output_mp4)
    ]

    print("Executing:", ' '.join(cmd))
    subprocess.run(cmd, check=True)


def process_directory(input_dir, output_dir):
    for subdir in Path(input_dir).iterdir():
        if not subdir.is_dir():
            continue

        mp4_file, gpx_file, csv_file = find_files(subdir)
        if not (mp4_file and gpx_file and csv_file):
            print(f"Skipping {subdir} (missing .mp4/.csv/.gpx)")
            continue

        try:
            gpx_start_time = get_first_gpx_time(gpx_file)
        except Exception as e:
            print(f"Skipping {subdir} (missing files: mp4={bool(mp4_file)}, gpx={bool(gpx_file)}, csv={bool(csv_file)})")
            continue

        try:
            codec = get_video_codec(mp4_file)
        except Exception as e:
            print(f"Failed to determine video codec of {mp4_file}: {e}")
            continue

        video_name = Path(mp4_file).stem
        output_subdir = Path(output_dir) / f"d1{video_name}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_mp4_path = output_subdir / f"{video_name}.mp4"

        try:
            correct_timestamp(mp4_file, output_mp4_path, gpx_start_time, codec)
            shutil.copy2(gpx_file, output_subdir)
            os.remove(mp4_file)
            print(f"Processed: {video_name}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed for {video_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", required=True, help="Directory of video subdirectories")
    parser.add_argument("-o", "--output_dir", required=True, help="Where corrected folders go")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
