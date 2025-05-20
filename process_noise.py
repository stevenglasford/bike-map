#!/usr/bin/env python3

import os
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
import librosa
import soundfile as sf
import re

try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False

def extract_audio_with_ffmpeg(video_path, audio_path):
    command = [
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",  # Remove or adjust if your ffmpeg build doesn't support CUDA
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(audio_path)
    ]
    print(f"[DEBUG] Running ffmpeg command: {' '.join(command)}")
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[DEBUG] Audio extracted to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed: {e.stderr.decode()}")
        raise

def reduce_vehicle_noise(audio_path, noise_profile_path):
    if not NOISE_REDUCTION_AVAILABLE:
        raise ImportError("noisereduce module is not installed. Install with: pip install noisereduce")
    
    print(f"[DEBUG] Loading full audio from: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    print(f"[DEBUG] Audio sample rate: {sr}, samples: {len(y)}")

    print(f"[DEBUG] Loading noise profile from: {noise_profile_path}")
    y_noise, _ = librosa.load(noise_profile_path, sr=sr)
    print(f"[DEBUG] Noise profile loaded, samples: {len(y_noise)}")

    print("[DEBUG] Running noise reduction...")
    reduced = nr.reduce_noise(y=y, y_noise=y_noise, sr=sr)
    denoised_path = audio_path.with_name(audio_path.stem + "_denoised.wav")
    sf.write(denoised_path, reduced, sr)
    print(f"[DEBUG] Denoised audio saved to {denoised_path}")
    return denoised_path

def analyze_audio(audio_path):
    print(f"[DEBUG] Analyzing audio from {audio_path}")
    audio = AudioSegment.from_wav(audio_path)
    duration = int(audio.duration_seconds)
    print(f"[DEBUG] Audio duration: {duration}s, frame rate: {audio.frame_rate}, channels: {audio.channels}")

    dBFS_by_second = []
    for sec in range(duration):
        segment = audio[sec * 1000:(sec + 1) * 1000]
        dB = segment.dBFS if segment.dBFS != float("-inf") else -100.0
        dBFS_by_second.append((sec, round(dB, 2)))
    print(f"[DEBUG] First 5 dBFS values: {dBFS_by_second[:5]}")
    return dBFS_by_second

def process_video(video_path, merged_df, noise_profile, output_csv_path):
    print(f"[INFO] Processing video: {video_path.name}")
    audio_temp = video_path.with_suffix(".temp.wav")
    extract_audio_with_ffmpeg(video_path, audio_temp)

    if noise_profile:
        try:
            denoised_path = reduce_vehicle_noise(audio_temp, noise_profile)
            os.remove(audio_temp)
            audio_path = denoised_path
        except Exception as e:
            print(f"[WARNING] Noise reduction failed, using raw audio. Reason: {e}")
            audio_path = audio_temp
    else:
        audio_path = audio_temp

    sound_data = analyze_audio(audio_path)
    os.remove(audio_path)

    rows = []
    missing_seconds = 0
    for sec, dB in sound_data:
        if sec in merged_df.index:
            row = merged_df.loc[sec]
            rows.append({
                "second": sec,
                "gpx_time": row["gpx_time"],
                "lat": row["lat"],
                "long": row["long"],
                "noise": dB
            })
        else:
            missing_seconds += 1

    if missing_seconds > 0:
        print(f"[DEBUG] {missing_seconds} seconds were skipped (not in merged_output.csv)")

    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved output to: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract noise levels from videos, optionally with bike noise reduction.")
    parser.add_argument("-d", "--directory", required=True, help="Directory with .mp4 and merged_output.csv")
    parser.add_argument("-o", "--output", help="Optional output CSV file path")
    parser.add_argument("-n", "--noise-profile", help="Optional WAV file of your vehicle's noise profile")
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
    print(f"[DEBUG] Loaded {len(merged_df)} rows from merged_output.csv")
    print(f"[DEBUG] Columns: {merged_df.columns.tolist()}")
    merged_df = merged_df.set_index("second")



    video_files = [f for f in directory.iterdir() if f.is_file() and re.search(r'\.mp4$', f.name, re.IGNORECASE)]
    print(f"[DEBUG] Found {len(video_files)} video file(s) with .mp4 or .MP4 extension")

    for video_file in video_files:
        output_csv = (
            Path(args.output)
            if args.output
            else directory / f"sound_output_{video_file.stem}.csv"
        )
        process_video(video_file, merged_df, noise_profile, output_csv)

if __name__ == "__main__":
    main()

