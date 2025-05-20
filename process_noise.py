#!/usr/bin/env python3

import os
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
import librosa
import soundfile as sf
import cv2
from tqdm import tqdm

try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False

def extract_audio_with_ffmpeg(video_path, audio_path):
    command = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "1",
        str(audio_path)
    ]
    print(f"[DEBUG] Running ffmpeg command: {' '.join(command)}")
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(f"[DEBUG] Audio extracted to {audio_path}")

def reduce_vehicle_noise(audio_path, noise_profile_path):
    if not NOISE_REDUCTION_AVAILABLE:
        raise ImportError("noisereduce not installed: pip install noisereduce")
    y, sr = librosa.load(audio_path, sr=None)
    y_noise, _ = librosa.load(noise_profile_path, sr=sr)
    reduced = nr.reduce_noise(y=y, y_noise=y_noise, sr=sr)
    denoised_path = audio_path.with_name(audio_path.stem + "_denoised.wav")
    sf.write(denoised_path, reduced, sr)
    return denoised_path

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"[DEBUG] Frame rate: {fps} fps")
    return fps

def analyze_audio_by_frame(audio_path, fps):
    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)
    frame_duration = 1000 / fps
    total_frames = int(duration_ms / frame_duration)

    print(f"[DEBUG] Audio length: {duration_ms}ms, Estimated frames: {total_frames}")
    dBFS_by_frame = []

    for frame_idx in tqdm(range(total_frames), desc="Analyzing audio per frame"):
        start_ms = int(frame_idx * frame_duration)
        end_ms = int((frame_idx + 1) * frame_duration)
        segment = audio[start_ms:end_ms]
        dB = segment.dBFS if segment.dBFS != float("-inf") else -100.0
        dBFS_by_frame.append((frame_idx, round(dB, 2)))

    return dBFS_by_frame

def interpolate_gps_data(merged_df, total_frames, fps):
    merged_df = merged_df.copy()
    merged_df = merged_df.set_index("second")
    merged_df = merged_df.reindex(range(int(total_frames / fps) + 1))
    merged_df.interpolate(method="linear", inplace=True)
    merged_df = merged_df.fillna(method="bfill").fillna(method="ffill")
    
    interpolated = []
    for f in range(total_frames):
        second = f / fps
        second_floor = int(second)
        row = merged_df.loc[second_floor]
        interpolated.append({
            "frame": f,
            "gpx_time": row["gpx_time"],
            "lat": row["lat"],
            "long": row["long"]
        })
    return interpolated

def process_video(video_path, merged_df, noise_profile, output_csv_path):
    print(f"[INFO] Processing video: {video_path.name}")
    audio_temp = video_path.with_suffix(".temp.wav")
    extract_audio_with_ffmpeg(video_path, audio_temp)

    if noise_profile:
        try:
            audio_path = reduce_vehicle_noise(audio_temp, noise_profile)
            os.remove(audio_temp)
        except Exception as e:
            print(f"[WARNING] Noise reduction failed, using raw audio. Reason: {e}")
            audio_path = audio_temp
    else:
        audio_path = audio_temp

    fps = get_frame_rate(video_path)
    sound_data = analyze_audio_by_frame(audio_path, fps)
    os.remove(audio_path)

    total_frames = len(sound_data)
    gps_data = interpolate_gps_data(merged_df, total_frames, fps)

    rows = []
    for (frame_idx, dB), gps in zip(sound_data, gps_data):
        rows.append({
            "frame": frame_idx,
            "gpx_time": gps["gpx_time"],
            "lat": gps["lat"],
            "long": gps["long"],
            "noise": dB
        })

    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved output to: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract per-frame noise levels from video.")
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-o", "--output", help="Optional output file name")
    parser.add_argument("-n", "--noise-profile", help="Optional WAV file of your bike noise profile")
    args = parser.parse_args()

    directory = Path(args.directory)
    merged_csv_path = directory / "merged_output.csv"
    if not merged_csv_path.exists():
        raise FileNotFoundError(f"[ERROR] Missing merged_output.csv in {directory}")

    noise_profile = Path(args.noise_profile) if args.noise_profile else None
    if noise_profile and not noise_profile.exists():
        raise FileNotFoundError(f"[ERROR] Noise profile not found: {noise_profile}")

    print(f"[DEBUG] Reading merged_output.csv from: {merged_csv_path}")
    merged_df = pd.read_csv(merged_csv_path)
    print(f"[DEBUG] Loaded {len(merged_df)} rows")
    print(f"[DEBUG] Columns: {merged_df.columns.tolist()}")

    video_files = [f for f in directory.iterdir() if f.suffix.lower() == ".mp4"]
    print(f"[DEBUG] Found {len(video_files)} video file(s)")

    for video_file in video_files:
        output_csv = Path(args.output) if args.output else directory / f"frame_noise_{video_file.stem}.csv"
        process_video(video_file, merged_df, noise_profile, output_csv)

if __name__ == "__main__":
    main()

