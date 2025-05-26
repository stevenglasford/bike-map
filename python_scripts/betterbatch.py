import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
import shutil
import argparse
from video_gpx_stitcher import parse_gpx, export_csv, dtw_align
from scipy.signal import correlate

console = Console()

def compute_video_accel(video_path, stride=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    motion = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # GPU-accelerated Farneback if OpenCV-CUDA is enabled
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion.append(np.mean(magnitude))
        prev_gray = gray
        frame_idx += 1

    cap.release()

    # Convert to acceleration
    motion = np.array(motion)
    accel = np.diff(motion, n=1, prepend=motion[0])
    return accel.tolist()

def align_signals(video_accel, gpx_accel, method="correlation"):
    if method == "correlation":
        min_len = min(len(video_accel), len(gpx_accel))
        video_accel = np.array(video_accel[:min_len])
        gpx_accel = np.array(gpx_accel[:min_len])
        corr = correlate(video_accel - np.mean(video_accel), gpx_accel - np.mean(gpx_accel), mode='full')
        offset = np.argmax(corr) - len(video_accel) + 1
        score = np.max(corr) / (np.std(video_accel) * np.std(gpx_accel) * len(video_accel))
        return offset, score
    elif method == "dtw":
        return dtw_align(video_accel, gpx_accel)

def match_gpx_to_video(video_path, gpx_files, stride=1, method="correlation"):
    video_accel = compute_video_accel(video_path, stride)
    console.print(f"[cyan]Computed video accel for {video_path.name} ({len(video_accel)} frames)[/cyan]")

    best_score = -np.inf
    best_offset = 0
    best_gpx = None
    best_gpx_df = None
    best_overlap = 0

    for gpx_path in gpx_files:
        try:
            gpx_df = parse_gpx(str(gpx_path))
            gpx_accel = gpx_df['accel'].tolist()
            if len(video_accel) < 20 or len(gpx_accel) < 20:
                continue

            offset, score = align_signals(video_accel, gpx_accel, method=method)
            overlap = min(len(video_accel), len(gpx_accel) - abs(offset))
            if not np.isnan(score) and score > best_score and overlap > 30:
                best_score = score
                best_offset = offset
                best_gpx = gpx_path
                best_gpx_df = gpx_df
                best_overlap = overlap
        except Exception as e:
            console.print(f"[red]Error with {gpx_path.name}:[/red] {e}")

    return best_gpx, best_offset, video_accel, best_gpx_df, best_overlap, best_score

def main(video_dir, gpx_dir, output_dir, stride=1, method="correlation"):
    video_dir, gpx_dir, output_dir = map(Path, [video_dir, gpx_dir, output_dir])
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.rglob("*.mp4"))
    gpx_files = sorted(gpx_dir.rglob("*.gpx"))
    console.print(f"[bold cyan]Found {len(video_files)} videos and {len(gpx_files)} GPX files[/bold cyan]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn(), console=console
    ) as progress:
        task = progress.add_task("[green]Processing...", total=len(video_files))

        for video_path in video_files:
            console.print(f"\n[yellow]Matching:[/yellow] {video_path.name}")
            match = match_gpx_to_video(video_path, gpx_files, stride=stride, method=method)
            best_gpx, offset, video_accel, gpx_df, overlap, score = match

            if best_gpx is None or score < 0.5:
                console.print(f"[red]❌ No reliable match for {video_path.name}[/red]")
                progress.advance(task)
                continue

            out_dir = output_dir / f'd{video_path.stem}'
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(video_path, out_dir / video_path.name)
            shutil.copy(best_gpx, out_dir / best_gpx.name)
            export_csv(video_accel, gpx_df, offset, out_dir / "aligned_output.csv")

            console.print(
                f"[green]✔ Matched:[/green] {best_gpx.name} [blue](score={score:.4f}, overlap={overlap}s, offset={offset})[/blue]"
            )
            progress.advance(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_dir", required=True)
    parser.add_argument("-g", "--gpx_dir", required=True)
    parser.add_argument("-o", "--output_dir", default="./matched_output")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for motion analysis")
    parser.add_argument("--method", choices=["correlation", "dtw"], default="correlation", help="Alignment method")
    args = parser.parse_args()
    main(args.video_dir, args.gpx_dir, args.output_dir, stride=args.stride, method=args.method)