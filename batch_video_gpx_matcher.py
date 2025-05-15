import os
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from video_gpx_aligner import parse_gpx, compute_motion_accel, align_by_accel, export_csv
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

console = Console()

def match_gpx_to_video(video_path, gpx_files):
    video_accel = compute_motion_accel(str(video_path))
    console.print(f"[cyan]Computed motion accel for {video_path.name} ({len(video_accel)} frames)[/cyan]")

    best_score = -np.inf
    best_offset = 0
    best_gpx = None
    best_gpx_df = None

    for gpx_path in gpx_files:
        try:
            gpx_df = parse_gpx(str(gpx_path))
            gpx_accel = list(gpx_df['accel'])

            console.print(f"[magenta]Checking:[/magenta] {gpx_path.name} with {len(gpx_df)} trackpoints")

            if len(gpx_accel) < 2 or len(video_accel) < 2:
                continue

            offset = align_by_accel(video_accel, gpx_accel)
            corr_len = min(len(video_accel), len(gpx_accel) - offset)
            if corr_len <= 0:
                continue

            score = np.corrcoef(video_accel[:corr_len], gpx_accel[offset:offset + corr_len])[0, 1]
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_offset = offset
                best_gpx = gpx_path
                best_gpx_df = gpx_df
        except Exception as e:
            console.print(f"[red]Error parsing {gpx_path.name}:[/red] {e}")
            continue

    return best_gpx, best_offset, video_accel, best_gpx_df, best_score

def main(video_dir, gpx_dir, output_dir):
    video_dir = Path(video_dir)
    gpx_dir = Path(gpx_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.MP4")))
    gpx_files = sorted(list(gpx_dir.rglob("*.gpx")) + list(gpx_dir.rglob("*.GPX")))

    console.print(f"[bold cyan]Found {len(video_files)} video(s) and {len(gpx_files)} GPX file(s).[/bold cyan]")

    for vf in video_files:
        console.print(f"[blue]Video:[/blue] {vf}")
    for gf in gpx_files:
        console.print(f"[magenta]GPX:[/magenta] {gf}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Processing videos...", total=len(video_files))

        for video_path in video_files:
            console.print(f"\n[yellow]Analyzing:[/yellow] {video_path.name}")
            best_gpx, offset, video_accel, gpx_df, score = match_gpx_to_video(video_path, gpx_files)

            if best_gpx is None:
                console.print(f"[red]No match found for:[/red] {video_path.name}")
                console.print("[dim]Consider checking if the GPX files have enough data, or if the video is too short or static.[/dim]")
                progress.advance(task)
                continue

            new_dir = output_dir / f'd{video_path.stem}'
            new_dir.mkdir(parents=True, exist_ok=True)

            new_video = new_dir / video_path.name
            new_gpx = new_dir / best_gpx.name
            new_csv = new_dir / "aligned_output.csv"

            shutil.copy(video_path, new_video)
            shutil.copy(best_gpx, new_gpx)
            export_csv(video_accel, gpx_df, offset, new_csv)

            console.print(f"[green]✔ Matched with:[/green] {best_gpx.name} [blue](corr={score:.4f})[/blue]")
            progress.advance(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_dir", required=True, help="Directory of MP4 videos")
    parser.add_argument("-g", "--gpx_dir", required=True, help="Directory of GPX files")
    parser.add_argument("-o", "--output_dir", default="./matched_output", help="Where to place output directories")
    args = parser.parse_args()

    main(args.video_dir, args.gpx_dir, args.output_dir)