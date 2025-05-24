#!/usr/bin/env python3

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import re

def get_prefix(filename):
    parts = filename.split('_')
    for i in range(len(parts)-1, 0, -1):
        prefix = '_'.join(parts[:i])
        if all(f.startswith(prefix) for f in similar_files[filename]):
            return prefix
    return filename

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\.]', '_', name)

def monitor_directory(watch_dir: Path):
    seen = set(f.name for f in watch_dir.iterdir() if f.is_file())

    while True:
        current = set(f.name for f in watch_dir.iterdir() if f.is_file())
        added = current - seen

        for fname in added:
            fpath = watch_dir / fname
            if not fpath.suffix.lower() in ['.mp4', '.gpx']:
                print(f"Deleting unsupported file: {fname}")
                try:
                    fpath.unlink()
                except Exception as e:
                    print(f"Error deleting {fname}: {e}")
                continue

            with open('new_files.txt', 'a') as f:
                f.write(f"{fname}\n")
                print(f"Logged new file: {fname}")

        seen = current
        process_csv_files(watch_dir)
        time.sleep(2)

def process_csv_files(root_dir: Path):
    serve_dir = root_dir / "serve"
    serve_dir.mkdir(exist_ok=True)

    all_csvs = list(root_dir.rglob("*.csv"))
    target_map = defaultdict(list)

    for csv in all_csvs:
        if csv.name == "new_files.txt":
            continue

        parent = csv.parent.name
        base = csv.name
        target_name = base

        if (serve_dir / base).exists():
            stem, suffix = os.path.splitext(base)
            target_name = f"{stem}_{sanitize_filename(parent)}{suffix}"

        target_map[target_name].append(csv)

    # Clear serve_dir (optional: comment this if you want to preserve old files)
    for f in serve_dir.iterdir():
        if f.is_file():
            f.unlink()

    grouped = defaultdict(list)

    for new_name, sources in target_map.items():
        for src in sources:
            dest = serve_dir / new_name
            shutil.copy2(src, dest)
            grouped[new_name].append(new_name)

    # Group files with common prefixes
    prefix_groups = defaultdict(list)
    for f in serve_dir.iterdir():
        prefix = '_'.join(f.stem.split('_')[:-1])  # ignore last part
        prefix_groups[prefix].append(f)

    for prefix, files in prefix_groups.items():
        if len(files) < 2:
            continue
        group_dir = serve_dir / prefix
        group_dir.mkdir(exist_ok=True)
        for f in files:
            shutil.move(str(f), group_dir / f.name)

def main():
    parser = argparse.ArgumentParser(description="Watch directory and process files.")
    parser.add_argument('-d', '--dir', required=True, help="Directory to watch")
    args = parser.parse_args()

    watch_dir = Path(args.dir).resolve()
    if not watch_dir.exists() or not watch_dir.is_dir():
        print(f"Directory does not exist: {watch_dir}")
        sys.exit(1)

    print(f"Watching directory: {watch_dir}")
    monitor_directory(watch_dir)

if __name__ == "__main__":
    main()