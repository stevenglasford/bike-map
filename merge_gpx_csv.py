import pandas as pd
import gpxpy
import gpxpy.gpx
from pathlib import Path
from geopy.distance import geodesic

def parse_gpx(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    records = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time and point.latitude and point.longitude:
                    records.append({
                        'gpx_time': pd.to_datetime(point.time),
                        'gpx_lat': point.latitude,
                        'gpx_lon': point.longitude,
                        'gpx_speed': point.speed if point.speed is not None else 0
                    })
    return pd.DataFrame(records)

def merge_on_location(csv_df, gpx_df, tolerance_meters=5):
    merged_rows = []

    for idx, row in csv_df.iterrows():
        lat, lon = row['lat'], row['lon']
        closest = None
        min_dist = float('inf')

        for _, gpx_row in gpx_df.iterrows():
            dist = geodesic((lat, lon), (gpx_row['gpx_lat'], gpx_row['gpx_lon'])).meters
            if dist < min_dist and dist <= tolerance_meters:
                closest = gpx_row
                min_dist = dist

        if closest is not None:
            merged_rows.append({
                **row,
                'gpx_time': closest['gpx_time'],
                'gpx_lat': closest['gpx_lat'],
                'gpx_lon': closest['gpx_lon'],
                'gpx_speed': closest['gpx_speed']
            })
        else:
            # fallback: keep the row without GPX data
            merged_rows.append({**row, 'gpx_time': None, 'gpx_lat': None, 'gpx_lon': None, 'gpx_speed': None})

    return pd.DataFrame(merged_rows)

def merge_gpx_csv(csv_path, gpx_path, output_path):
    csv_df = pd.read_csv(csv_path)
    gpx_df = parse_gpx(gpx_path)

    if not {'lat', 'lon', 'second'}.issubset(csv_df.columns):
        raise ValueError("CSV must contain 'second', 'lat', and 'lon' columns.")

    merged_df = merge_on_location(csv_df, gpx_df)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', required=True, help='Path to the input CSV file')
    parser.add_argument('-g', '--gpx', required=True, help='Path to the GPX file')
    parser.add_argument('-o', '--output', required=False, default='merged_output.csv', help='Output CSV path')
    args = parser.parse_args()

    merge_gpx_csv(args.csv, args.gpx, args.output)