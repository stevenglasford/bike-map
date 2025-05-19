import pandas as pd
import gpxpy
import argparse
from datetime import datetime

def parse_gpx(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                # Convert timezone-aware time to naive UTC time
                points.append({
                    'gpx_time': point.time.replace(tzinfo=None),
                    'gpx_lat': point.latitude,
                    'gpx_lon': point.longitude,
                    'gpx_speed': point.speed if point.speed is not None else 0.0
                })
    return pd.DataFrame(points)

def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['second'], unit='s', origin='unix')
    return df

def merge_gpx_csv(csv_path, gpx_path, output_path):
    csv_df = parse_csv(csv_path)
    gpx_df = parse_gpx(gpx_path)

    if gpx_df.empty:
        print("No points found in GPX file.")
        return

    csv_df.sort_values('time', inplace=True)
    gpx_df.sort_values('gpx_time', inplace=True)

    # Merge on naive datetime
    merged = pd.merge_asof(
        csv_df,
        gpx_df,
        left_on='time',
        right_on='gpx_time',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=1)
    )

    merged.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', required=True, help='Path to CSV file (with "second" column)')
    parser.add_argument('-g', '--gpx', required=True, help='Path to GPX file')
    parser.add_argument('-o', '--output', default='merged_output.csv', help='Path to output CSV')
    args = parser.parse_args()

    merge_gpx_csv(args.csv, args.gpx, args.output)