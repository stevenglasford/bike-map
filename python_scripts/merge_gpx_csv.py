#!/usr/bin/env python3

import pandas as pd
import gpxpy
from pathlib import Path
from datetime import datetime
from math import isclose

LAT_LON_TOLERANCE = 0.0001  # acceptable difference for lat/lon match

def parse_gpx(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    gpx_records = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                gpx_records.append({
                    'gpx_lat': point.latitude,
                    'gpx_lon': point.longitude,
                    'gpx_time': point.time
                })
    return pd.DataFrame(gpx_records)

def match_gpx_to_csv(csv_path, gpx_path, output_path):
    csv_df = pd.read_csv(csv_path)
    gpx_df = parse_gpx(gpx_path)

    print(f"Loaded {len(csv_df)} CSV rows and {len(gpx_df)} GPX points")

    matched_rows = []

    for i, row in csv_df.iterrows():
        match = gpx_df[
            gpx_df.apply(
                lambda g: isclose(g.gpx_lat, row.lat, abs_tol=LAT_LON_TOLERANCE) and
                          isclose(g.gpx_lon, row.lon, abs_tol=LAT_LON_TOLERANCE), axis=1
            )
        ]
        if not match.empty:
            gpx_time = match.iloc[0]['gpx_time']
        else:
            gpx_time = None  # unmatched

        matched_rows.append({
            'second': row['second'],
            'lat': row['lat'],
            'long': row['lon'],
            'speed_mph': row['speed_mph'],
            'gpx_time': gpx_time
        })

    output_df = pd.DataFrame(matched_rows)
    output_df.to_csv(output_path, index=False)
    print(f"Merged file written to {output_path} with {len(output_df)} rows")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", required=True, help="Path to input CSV")
    parser.add_argument("-g", "--gpx", required=True, help="Path to input GPX")
    parser.add_argument("-o", "--output", required=True, help="Output path for merged CSV")
    args = parser.parse_args()

    match_gpx_to_csv(args.csv, args.gpx, args.output)