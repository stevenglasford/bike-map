#!/usr/bin/env python3
"""
Extract the highest scoring GPX matches for each video from the JSON results file.
"""

import json
import csv
import sys
import os
from pathlib import Path

def extract_best_matches(json_file_path, output_csv_path=None):
    """
    Extract the best GPX match for each video based on combined_score.
    
    Args:
        json_file_path: Path to the input JSON file
        output_csv_path: Path to output CSV file (optional)
    """
    
    # Load JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Extract results
    results = data.get('results', {})
    if not results:
        print("No results found in JSON file")
        return
    
    # Process each video
    best_matches = []
    
    for video_path, video_data in results.items():
        matches = video_data.get('matches', [])
        if not matches:
            continue
        
        # Find the match with highest combined_score
        best_match = None
        best_score = -1
        
        for match in matches:
            score = match.get('combined_score')
            if score is not None and score > best_score:
                best_score = score
                best_match = match
        
        # If we found a match with combined_score
        if best_match:
            best_matches.append({
                'video_path': video_path,
                'video_filename': os.path.basename(video_path),
                'best_gpx_path': best_match.get('path', ''),
                'best_gpx_filename': os.path.basename(best_match.get('path', '')),
                'combined_score': best_score,
                'offset_confidence': best_match.get('offset_confidence', ''),
                'temporal_offset_seconds': best_match.get('temporal_offset_seconds', ''),
                'sync_quality': best_match.get('sync_quality', ''),
                'video_duration': best_match.get('video_duration', ''),
                'gps_duration': best_match.get('duration', '')
            })
        else:
            # Include videos with no good matches
            best_matches.append({
                'video_path': video_path,
                'video_filename': os.path.basename(video_path),
                'best_gpx_path': 'NO_GOOD_MATCH',
                'best_gpx_filename': 'NO_GOOD_MATCH',
                'combined_score': 'N/A',
                'offset_confidence': 'N/A',
                'temporal_offset_seconds': 'N/A',
                'sync_quality': 'N/A',
                'video_duration': video_data.get('matches', [{}])[0].get('video_duration', '') if matches else '',
                'gps_duration': 'N/A'
            })
    
    # Sort by combined_score (highest first)
    best_matches.sort(key=lambda x: float(x['combined_score']) if x['combined_score'] != 'N/A' else -1, reverse=True)
    
    # Determine output file path
    if output_csv_path is None:
        input_path = Path(json_file_path)
        output_csv_path = input_path.parent / f"{input_path.stem}_best_matches.csv"
    
    # Write to CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'video_filename',
                'best_gpx_filename', 
                'combined_score',
                'offset_confidence',
                'temporal_offset_seconds',
                'sync_quality',
                'video_duration',
                'gps_duration',
                'video_path',
                'best_gpx_path'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_matches)
        
        print(f"✅ Successfully extracted {len(best_matches)} video matches")
        print(f"📄 Results saved to: {output_csv_path}")
        
        # Print summary statistics
        good_matches = [m for m in best_matches if m['combined_score'] != 'N/A']
        if good_matches:
            scores = [float(m['combined_score']) for m in good_matches]
            print(f"\n📊 Summary Statistics:")
            print(f"   Videos with good matches: {len(good_matches)}/{len(best_matches)}")
            print(f"   Average combined score: {sum(scores)/len(scores):.3f}")
            print(f"   Highest combined score: {max(scores):.3f}")
            print(f"   Lowest combined score: {min(scores):.3f}")
            
            # Show top 5 matches
            print(f"\n🏆 Top 5 Highest Scoring Matches:")
            for i, match in enumerate(best_matches[:5]):
                if match['combined_score'] != 'N/A':
                    print(f"   {i+1}. {match['video_filename']} -> {match['best_gpx_filename']} "
                          f"(score: {match['combined_score']})")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_best_matches.py <input_json_file> [output_csv_file]")
        print("\nExample:")
        print("  python extract_best_matches.py complete_turbo_360_report_ramcache.json")
        print("  python extract_best_matches.py results.json best_matches.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    extract_best_matches(input_file, output_file)

if __name__ == "__main__":
    main()

