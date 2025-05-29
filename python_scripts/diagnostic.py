import cv2
import numpy as np
import gpxpy
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
import argparse
import os
import glob
from pathlib import Path
import json
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diagnostic.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleDiagnosticAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def analyze_video(self, video_path):
        """Extract basic features for diagnostic purposes"""
        logger.info(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video stats: FPS={fps}, Frames={frame_count}, Duration={duration:.2f}s")
        
        # Extract simple motion features
        motion_magnitudes = []
        prev_gray = None
        
        # Sample every 30 frames (roughly 1 per second)
        sample_interval = max(1, int(fps))
        sampled_frames = 0
        
        for frame_idx in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  # Smaller for faster processing
            
            if prev_gray is not None:
                # Simple frame difference
                diff = cv2.absdiff(prev_gray, gray)
                motion_mag = np.mean(diff)
                motion_magnitudes.append(motion_mag)
            
            prev_gray = gray
            sampled_frames += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(motion_magnitudes)} motion samples from {sampled_frames} frames")
        
        if len(motion_magnitudes) < 5:
            logger.warning(f"Too few motion samples: {len(motion_magnitudes)}")
            return None
        
        # Calculate basic statistics
        motion_array = np.array(motion_magnitudes)
        stats = {
            'mean': np.mean(motion_array),
            'std': np.std(motion_array),
            'max': np.max(motion_array),
            'min': np.min(motion_array)
        }
        
        logger.info(f"Motion stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        return {
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'motion_signal': motion_array,
            'motion_stats': stats,
            'num_samples': len(motion_magnitudes)
        }
    
    def analyze_gpx(self, gpx_path):
        """Extract basic features from GPX"""
        logger.debug(f"Analyzing GPX: {gpx_path}")
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for pt in segment.points:
                        if pt.time:
                            points.append({
                                'time': pt.time.replace(tzinfo=None),
                                'lat': pt.latitude,
                                'lon': pt.longitude
                            })
            
            if len(points) < 5:
                return None
            
            # Sort by time
            points = sorted(points, key=lambda x: x['time'])
            
            # Calculate duration
            duration = (points[-1]['time'] - points[0]['time']).total_seconds()
            
            # Calculate speeds
            speeds = []
            for i in range(1, len(points)):
                dt = (points[i]['time'] - points[i-1]['time']).total_seconds()
                if dt > 0:
                    # Simple distance calculation
                    dlat = points[i]['lat'] - points[i-1]['lat']
                    dlon = points[i]['lon'] - points[i-1]['lon']
                    dist = np.sqrt(dlat**2 + dlon**2) * 69  # Rough miles conversion
                    speed = (dist / dt) * 3600  # mph
                    speeds.append(speed)
            
            if not speeds:
                return None
            
            speed_array = np.array(speeds)
            
            # Remove outliers (speeds > 100 mph)
            speed_array = speed_array[speed_array < 100]
            
            if len(speed_array) < 5:
                return None
            
            stats = {
                'mean': np.mean(speed_array),
                'std': np.std(speed_array),
                'max': np.max(speed_array),
                'min': np.min(speed_array)
            }
            
            return {
                'duration': duration,
                'num_points': len(points),
                'speed_signal': speed_array,
                'speed_stats': stats,
                'num_samples': len(speed_array)
            }
        
        except Exception as e:
            logger.error(f"Error parsing GPX {gpx_path}: {e}")
            return None
    
    def simple_correlation(self, video_features, gpx_features):
        """Calculate simple correlation score"""
        # Duration similarity (within 20%)
        duration_ratio = min(video_features['duration'], gpx_features['duration']) / \
                        max(video_features['duration'], gpx_features['duration'])
        
        if duration_ratio < 0.8:
            return 0.0
        
        # Normalize signals to same length
        video_signal = video_features['motion_signal']
        gpx_signal = gpx_features['speed_signal']
        
        # Resample to same length
        target_len = min(len(video_signal), len(gpx_signal), 100)
        
        if target_len < 10:
            return 0.0
        
        video_resampled = np.interp(
            np.linspace(0, len(video_signal)-1, target_len),
            np.arange(len(video_signal)),
            video_signal
        )
        
        gpx_resampled = np.interp(
            np.linspace(0, len(gpx_signal)-1, target_len),
            np.arange(len(gpx_signal)),
            gpx_signal
        )
        
        # Normalize
        if np.std(video_resampled) > 0:
            video_norm = (video_resampled - np.mean(video_resampled)) / np.std(video_resampled)
        else:
            video_norm = video_resampled
            
        if np.std(gpx_resampled) > 0:
            gpx_norm = (gpx_resampled - np.mean(gpx_resampled)) / np.std(gpx_resampled)
        else:
            gpx_norm = gpx_resampled
        
        # Simple correlation
        if len(video_norm) > 0 and len(gpx_norm) > 0:
            correlation = np.corrcoef(video_norm, gpx_norm)[0, 1]
            return abs(correlation) * duration_ratio
        
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Video-GPX Matcher")
    parser.add_argument("-d", "--directory", required=True, help="Directory with videos and GPX files")
    parser.add_argument("-o", "--output", default="diagnostic_results.json", help="Output file")
    parser.add_argument("--max_videos", type=int, default=5, help="Max videos to test")
    parser.add_argument("--max_gpx", type=int, default=100, help="Max GPX files per video")
    
    args = parser.parse_args()
    
    # Find files
    video_files = []
    for ext in ['mp4', 'MP4', 'avi', 'AVI', 'mov', 'MOV']:
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
    
    gpx_files = []
    for ext in ['gpx', 'GPX']:
        gpx_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
    
    # Remove duplicates
    video_files = list(set(video_files))[:args.max_videos]
    gpx_files = list(set(gpx_files))[:args.max_gpx]
    
    logger.info(f"Testing with {len(video_files)} videos and {len(gpx_files)} GPX files")
    
    analyzer = SimpleDiagnosticAnalyzer()
    results = {
        'videos_analyzed': 0,
        'videos_failed': 0,
        'gpx_analyzed': 0,
        'gpx_failed': 0,
        'correlations_attempted': 0,
        'correlations_successful': 0,
        'details': []
    }
    
    # Analyze videos
    video_features_dict = {}
    for video_path in tqdm(video_files, desc="Analyzing videos"):
        features = analyzer.analyze_video(video_path)
        if features:
            video_features_dict[video_path] = features
            results['videos_analyzed'] += 1
        else:
            results['videos_failed'] += 1
            logger.warning(f"Failed to analyze video: {video_path}")
    
    # Analyze GPX files
    gpx_features_dict = {}
    for gpx_path in tqdm(gpx_files, desc="Analyzing GPX files"):
        features = analyzer.analyze_gpx(gpx_path)
        if features:
            gpx_features_dict[gpx_path] = features
            results['gpx_analyzed'] += 1
        else:
            results['gpx_failed'] += 1
    
    logger.info(f"Successfully analyzed {len(video_features_dict)} videos and {len(gpx_features_dict)} GPX files")
    
    # Test correlations
    for video_path, video_features in video_features_dict.items():
        video_name = Path(video_path).name
        logger.info(f"\nTesting correlations for {video_name}")
        
        correlations = []
        for gpx_path, gpx_features in gpx_features_dict.items():
            try:
                score = analyzer.simple_correlation(video_features, gpx_features)
                results['correlations_attempted'] += 1
                
                if score > 0:
                    results['correlations_successful'] += 1
                    correlations.append({
                        'gpx': Path(gpx_path).name,
                        'score': float(score),
                        'duration_match': float(gpx_features['duration'] / video_features['duration'])
                    })
                    
            except Exception as e:
                logger.error(f"Correlation error: {e}")
        
        # Sort by score
        correlations.sort(key=lambda x: x['score'], reverse=True)
        
        # Log top matches
        if correlations:
            logger.info(f"Top 3 matches for {video_name}:")
            for i, match in enumerate(correlations[:3]):
                logger.info(f"  {i+1}. {match['gpx']} - Score: {match['score']:.3f}")
        else:
            logger.warning(f"No correlations found for {video_name}")
        
        results['details'].append({
            'video': video_name,
            'duration': video_features['duration'],
            'samples': video_features['num_samples'],
            'motion_mean': float(video_features['motion_stats']['mean']),
            'motion_std': float(video_features['motion_stats']['std']),
            'top_matches': correlations[:5]
        })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nDiagnostic Summary:")
    print(f"Videos analyzed: {results['videos_analyzed']}/{len(video_files)}")
    print(f"GPX files analyzed: {results['gpx_analyzed']}/{len(gpx_files)}")
    print(f"Correlations attempted: {results['correlations_attempted']}")
    print(f"Correlations with score > 0: {results['correlations_successful']}")
    
    if results['details']:
        print("\nSample results:")
        for video_result in results['details'][:3]:
            print(f"\n{video_result['video']}:")
            print(f"  Duration: {video_result['duration']:.1f}s")
            print(f"  Motion samples: {video_result['samples']}")
            if video_result['top_matches']:
                print(f"  Best match: {video_result['top_matches'][0]['gpx']} (score: {video_result['top_matches'][0]['score']:.3f})")

if __name__ == "__main__":
    main()