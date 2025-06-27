#!/usr/bin/env python3
"""
Comprehensive Video Analysis Processor
=====================================

This script processes the output from matcher50.py and performs complete AI video analysis
on both flat and 360° panoramic videos, producing efficient CSV outputs for permanent storage.

Features:
- Handles matcher50.py output format
- Supports both flat and 360° panoramic videos
- Object detection, tracking, and speed analysis
- Stoplight detection with bearing-aware positioning
- Traffic counting and density analysis
- Audio noise level analysis
- Street scene complexity assessment
- Optimized for large-scale processing (15TB+ datasets)
- Produces space-efficient CSV outputs

Usage:
    python video_analysis_processor.py -i /path/to/matcher50_results -o /path/to/output
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import librosa
import gpxpy
from geopy.distance import distance as geopy_distance
from geopy import Point
from scipy.signal import correlate
from math import isclose, radians, cos, sin, atan2, degrees, sqrt

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["YOLO_VERBOSE"] = "False"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('video_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class VideoAnalysisProcessor:
    """Main processor for comprehensive video analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_models()
        self.setup_directories()
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'processing_times': []
        }
    
    def setup_models(self):
        """Initialize AI models"""
        logger.info("Loading AI models...")
        
        # YOLO for object detection
        self.yolo_model = YOLO(self.config.get('yolo_model', 'yolo11x.pt'))
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Object class mappings
        self.object_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
            'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
            'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13
        }
        
        # Traffic light color detection setup
        self.setup_color_detection()
    
    def setup_color_detection(self):
        """Setup color detection for traffic lights"""
        # HSV ranges for traffic light colors
        self.color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
            'yellow': [(15, 100, 100), (35, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)]
        }
    
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different analysis types
        (self.output_dir / 'object_tracking').mkdir(exist_ok=True)
        (self.output_dir / 'stoplight_detection').mkdir(exist_ok=True)
        (self.output_dir / 'traffic_counting').mkdir(exist_ok=True)
        (self.output_dir / 'audio_analysis').mkdir(exist_ok=True)
        (self.output_dir / 'scene_complexity').mkdir(exist_ok=True)
        (self.output_dir / 'processing_reports').mkdir(exist_ok=True)
    
    def load_matcher50_results(self, results_path: str) -> Dict[str, Any]:
        """Load and parse matcher50.py results"""
        logger.info(f"Loading matcher50 results from: {results_path}")
        
        if results_path.endswith('.json'):
            with open(results_path, 'r') as f:
                return json.load(f)
        elif results_path.endswith('.csv'):
            df = pd.read_csv(results_path)
            return self.convert_csv_to_dict(df)
        else:
            raise ValueError(f"Unsupported results format: {results_path}")
    
    def convert_csv_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert CSV results to dictionary format"""
        results = {}
        for _, row in df.iterrows():
            video_path = row['video_path']
            if video_path not in results:
                results[video_path] = {'matches': []}
            
            match_info = {
                'path': row['gps_path'],
                'quality': row.get('quality', 'unknown'),
                'combined_score': row.get('combined_score', 0.0),
                'is_360_video': row.get('is_360_video', False)
            }
            results[video_path]['matches'].append(match_info)
        
        return results
    
    def detect_video_type(self, video_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Detect if video is 360° panoramic or flat"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, {}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Detect 360° video by aspect ratio
        aspect_ratio = width / height
        is_360 = 1.8 <= aspect_ratio <= 2.2
        is_equirectangular = (width, height) == (3840, 1920) or (width, height) == (4096, 2048)
        
        video_info = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'aspect_ratio': aspect_ratio,
            'is_360': is_360,
            'is_equirectangular': is_equirectangular,
            'video_type': '360_panoramic' if is_360 else 'flat'
        }
        
        return True, video_info
    
    def load_gps_data(self, gps_path: str) -> pd.DataFrame:
        """Load GPS data from GPX file or merged CSV"""
        if gps_path.endswith('.gpx'):
            return self.parse_gpx_to_dataframe(gps_path)
        elif gps_path.endswith('.csv'):
            df = pd.read_csv(gps_path)
            # Standardize column names
            if 'long' in df.columns:
                df['lon'] = df['long']
            return df
        else:
            raise ValueError(f"Unsupported GPS file format: {gps_path}")
    
    def parse_gpx_to_dataframe(self, gpx_path: str) -> pd.DataFrame:
        """Parse GPX file to DataFrame"""
        with open(gpx_path, 'r') as f:
            gpx = gpxpy.parse(f)
        
        records = []
        for track in gpx.tracks:
            for segment in track.segments:
                for i, point in enumerate(segment.points):
                    records.append({
                        'second': i,  # Approximate second
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'gpx_time': point.time,
                        'elevation': point.elevation or 0
                    })
        
        df = pd.DataFrame(records)
        if len(df) > 1:
            df['speed_mph'] = self.calculate_speeds(df)
        else:
            df['speed_mph'] = 0
        
        return df
    
    def calculate_speeds(self, df: pd.DataFrame) -> List[float]:
        """Calculate speeds from GPS coordinates"""
        speeds = [0]  # First point has 0 speed
        
        for i in range(1, len(df)):
            prev_point = (df.iloc[i-1]['lat'], df.iloc[i-1]['lon'])
            curr_point = (df.iloc[i]['lat'], df.iloc[i]['lon'])
            
            # Calculate distance in meters
            dist_meters = geopy_distance(prev_point, curr_point).meters
            
            # Convert to mph (assuming 1 second intervals)
            speed_mph = dist_meters * 2.237  # m/s to mph conversion
            speeds.append(speed_mph)
        
        return speeds
    
    def add_bearing_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bearing/heading column to GPS data"""
        bearings = [0]  # First point has 0 bearing
        
        for i in range(1, len(df)):
            lat1, lon1 = radians(df.iloc[i-1]['lat']), radians(df.iloc[i-1]['lon'])
            lat2, lon2 = radians(df.iloc[i]['lat']), radians(df.iloc[i]['lon'])
            
            dlon = lon2 - lon1
            y = sin(dlon) * cos(lat2)
            x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            
            bearing = (degrees(atan2(y, x)) + 360) % 360
            bearings.append(bearing)
        
        df['bearing'] = bearings
        return df
    
    def process_360_video(self, video_path: str, gps_df: pd.DataFrame, video_info: Dict) -> Dict[str, Any]:
        """Process 360° panoramic video with specialized handling"""
        logger.info(f"Processing 360° video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps = video_info['fps']
        
        # 360° specific processing
        width, height = video_info['width'], video_info['height']
        
        # For 360° videos, we need to consider the spherical projection
        # Extract key regions: front, left, right, back
        regions = self.define_360_regions(width, height)
        
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int),
            'scene_complexity': []
        }
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            second = int(frame_idx / fps)
            
            # Get GPS data for this timestamp
            gps_row = gps_df[gps_df['second'] == second]
            if gps_row.empty:
                frame_idx += 1
                continue
            
            gps_data = gps_row.iloc[0]
            
            # Process each region of the 360° video
            for region_name, region_coords in regions.items():
                region_frame = self.extract_region(frame, region_coords)
                
                # Run YOLO detection on the region
                detections = self.yolo_model(region_frame)[0]
                
                # Process detections for this region
                region_results = self.process_detections(
                    detections, region_frame, region_name, second, gps_data, video_info
                )
                
                # Aggregate results
                results['object_tracking'].extend(region_results['tracking'])
                results['stoplight_detection'].extend(region_results['stoplights'])
                
                for obj_type, count in region_results['counting'].items():
                    results['traffic_counting'][obj_type] += count
            
            # Calculate scene complexity for the full frame
            complexity = self.calculate_scene_complexity(frame, second, gps_data)
            results['scene_complexity'].append(complexity)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames of 360° video")
        
        cap.release()
        return results
    
    def process_flat_video(self, video_path: str, gps_df: pd.DataFrame, video_info: Dict) -> Dict[str, Any]:
        """Process flat/traditional video"""
        logger.info(f"Processing flat video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps = video_info['fps']
        
        # Initialize tracking
        tracker = cv2.legacy.TrackerCSRT_create()
        tracking_objects = {}
        next_object_id = 1
        
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int),
            'scene_complexity': []
        }
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            second = int(frame_idx / fps)
            
            # Get GPS data for this timestamp
            gps_row = gps_df[gps_df['second'] == second]
            if gps_row.empty:
                frame_idx += 1
                continue
            
            gps_data = gps_row.iloc[0]
            
            # Run YOLO detection
            detections = self.yolo_model(frame)[0]
            
            # Process detections
            frame_results = self.process_detections(
                detections, frame, 'main', second, gps_data, video_info
            )
            
            # Aggregate results
            results['object_tracking'].extend(frame_results['tracking'])
            results['stoplight_detection'].extend(frame_results['stoplights'])
            
            for obj_type, count in frame_results['counting'].items():
                results['traffic_counting'][obj_type] += count
            
            # Calculate scene complexity
            complexity = self.calculate_scene_complexity(frame, second, gps_data)
            results['scene_complexity'].append(complexity)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames of flat video")
        
        cap.release()
        return results
    
    def define_360_regions(self, width: int, height: int) -> Dict[str, Tuple]:
        """Define regions for 360° video analysis"""
        # Equirectangular projection regions
        # Front: center of image
        # Left: left quarter
        # Right: right quarter  
        # Back: edges (wraparound)
        
        regions = {
            'front': (width//4, 0, 3*width//4, height),
            'left': (0, 0, width//4, height),
            'right': (3*width//4, 0, width, height),
            'back_left': (0, 0, width//8, height),
            'back_right': (7*width//8, 0, width, height)
        }
        
        return regions
    
    def extract_region(self, frame: np.ndarray, coords: Tuple) -> np.ndarray:
        """Extract a region from the frame"""
        x1, y1, x2, y2 = coords
        return frame[y1:y2, x1:x2]
    
    def process_detections(self, detections, frame: np.ndarray, region_name: str, 
                          second: int, gps_data: pd.Series, video_info: Dict) -> Dict[str, Any]:
        """Process YOLO detections for a frame/region"""
        results = {
            'tracking': [],
            'stoplights': [],
            'counting': defaultdict(int)
        }
        
        if detections.boxes is None or len(detections.boxes) == 0:
            return results
        
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy().astype(int)
        confidences = detections.boxes.conf.cpu().numpy()
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf < 0.3:  # Confidence threshold
                continue
                
            x1, y1, x2, y2 = box.astype(int)
            
            # Get object class name
            class_names = self.yolo_model.names
            obj_class = class_names.get(cls, 'unknown')
            
            # Count objects
            results['counting'][obj_class] += 1
            
            # Process tracking for mobile objects
            if obj_class in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                tracking_data = {
                    'frame_second': second,
                    'object_class': obj_class,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'region': region_name,
                    'lat': float(gps_data['lat']),
                    'lon': float(gps_data['lon']),
                    'gps_time': str(gps_data.get('gpx_time', '')),
                    'video_type': video_info['video_type']
                }
                results['tracking'].append(tracking_data)
            
            # Process traffic lights
            if obj_class == 'traffic light':
                light_data = self.analyze_traffic_light(
                    frame, box, second, gps_data, video_info, region_name
                )
                if light_data:
                    results['stoplights'].append(light_data)
        
        return results
    
    def analyze_traffic_light(self, frame: np.ndarray, box: np.ndarray, 
                             second: int, gps_data: pd.Series, video_info: Dict, 
                             region_name: str) -> Optional[Dict]:
        """Analyze traffic light color and state"""
        x1, y1, x2, y2 = box.astype(int)
        
        # Extract traffic light ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Detect color
        color = self.detect_light_color(roi)
        
        # For 360° videos, calculate bearing based on region
        if video_info['video_type'] == '360_panoramic':
            bearing = self.calculate_360_bearing(x1, x2, video_info['width'], region_name)
        else:
            bearing = gps_data.get('bearing', 0)
        
        return {
            'second': second,
            'stoplight_color': color,
            'confidence': 0.8,  # Base confidence
            'bearing': bearing,
            'region': region_name,
            'lat': float(gps_data['lat']),
            'lon': float(gps_data['lon']),
            'gps_time': str(gps_data.get('gpx_time', '')),
            'video_type': video_info['video_type']
        }
    
    def detect_light_color(self, roi: np.ndarray) -> str:
        """Detect traffic light color using HSV analysis"""
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check each color
        color_scores = {}
        
        for color, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            if color == 'red':  # Red has two ranges
                mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            
            color_scores[color] = cv2.countNonZero(mask)
        
        # Return color with highest score
        if max(color_scores.values()) > 0:
            return max(color_scores, key=color_scores.get)
        else:
            return 'unknown'
    
    def calculate_360_bearing(self, x1: int, x2: int, frame_width: int, region_name: str) -> float:
        """Calculate bearing for 360° video based on x position"""
        # Center x position of detection
        center_x = (x1 + x2) / 2
        
        # Convert x position to bearing (0-360 degrees)
        # For equirectangular projection, x position maps directly to longitude/bearing
        bearing = (center_x / frame_width) * 360
        
        # Adjust based on region
        region_offsets = {
            'front': 0,
            'left': 270,
            'right': 90,
            'back_left': 180,
            'back_right': 180
        }
        
        bearing = (bearing + region_offsets.get(region_name, 0)) % 360
        return bearing
    
    def calculate_scene_complexity(self, frame: np.ndarray, second: int, gps_data: pd.Series) -> Dict:
        """Calculate scene complexity metrics"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate various complexity metrics
        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Texture variance
        texture_var = np.var(gray)
        
        # 3. Color variance
        color_var = np.var(frame)
        
        # 4. Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        return {
            'second': second,
            'edge_density': float(edge_density),
            'texture_variance': float(texture_var),
            'color_variance': float(color_var),
            'gradient_magnitude': float(gradient_magnitude),
            'complexity_score': float((edge_density + texture_var/10000 + gradient_magnitude/100) / 3),
            'lat': float(gps_data['lat']),
            'lon': float(gps_data['lon']),
            'gps_time': str(gps_data.get('gpx_time', ''))
        }
    
    def analyze_audio(self, video_path: str, gps_df: pd.DataFrame) -> List[Dict]:
        """Analyze audio track for noise levels and characteristics"""
        logger.info(f"Analyzing audio for: {os.path.basename(video_path)}")
        
        try:
            # Load audio
            y, sr = librosa.load(video_path, sr=None)
            
            # Calculate audio features
            duration = len(y) / sr
            hop_length = sr // 2  # 0.5 second windows
            
            # RMS energy (loudness)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            
            # Zero crossing rate (roughness)
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            
            audio_results = []
            
            for i, (rms_val, centroid_val, zcr_val) in enumerate(zip(rms, spectral_centroid, zcr)):
                second = int(i * 0.5)  # 0.5 second windows
                
                # Get corresponding GPS data
                gps_row = gps_df[gps_df['second'] == second]
                if gps_row.empty:
                    continue
                
                gps_data = gps_row.iloc[0]
                
                audio_results.append({
                    'second': second,
                    'rms_energy': float(rms_val),
                    'spectral_centroid': float(centroid_val),
                    'zero_crossing_rate': float(zcr_val),
                    'noise_level': float(rms_val * 100),  # Scale to 0-100
                    'lat': float(gps_data['lat']),
                    'lon': float(gps_data['lon']),
                    'gps_time': str(gps_data.get('gpx_time', ''))
                })
            
            return audio_results
            
        except Exception as e:
            logger.warning(f"Audio analysis failed for {video_path}: {e}")
            return []
    
    def save_results_to_csv(self, results: Dict[str, Any], video_name: str):
        """Save analysis results to optimized CSV files"""
        
        # Object tracking results
        if results['object_tracking']:
            tracking_df = pd.DataFrame(results['object_tracking'])
            tracking_path = self.output_dir / 'object_tracking' / f"{video_name}_tracking.csv"
            tracking_df.to_csv(tracking_path, index=False)
        
        # Stoplight detection results
        if results['stoplight_detection']:
            stoplight_df = pd.DataFrame(results['stoplight_detection'])
            stoplight_path = self.output_dir / 'stoplight_detection' / f"{video_name}_stoplights.csv"
            stoplight_df.to_csv(stoplight_path, index=False)
        
        # Traffic counting results
        if results['traffic_counting']:
            counting_data = []
            for obj_type, count in results['traffic_counting'].items():
                counting_data.append({
                    'video_name': video_name,
                    'object_type': obj_type,
                    'total_count': count
                })
            counting_df = pd.DataFrame(counting_data)
            counting_path = self.output_dir / 'traffic_counting' / f"{video_name}_counts.csv"
            counting_df.to_csv(counting_path, index=False)
        
        # Scene complexity results
        if results['scene_complexity']:
            complexity_df = pd.DataFrame(results['scene_complexity'])
            complexity_path = self.output_dir / 'scene_complexity' / f"{video_name}_complexity.csv"
            complexity_df.to_csv(complexity_path, index=False)
        
        # Audio analysis results
        if 'audio_analysis' in results and results['audio_analysis']:
            audio_df = pd.DataFrame(results['audio_analysis'])
            audio_path = self.output_dir / 'audio_analysis' / f"{video_name}_audio.csv"
            audio_df.to_csv(audio_path, index=False)
    
    def process_single_video(self, video_path: str, matched_gps_path: str) -> Dict[str, Any]:
        """Process a single video with its matched GPS data"""
        video_name = Path(video_path).stem
        logger.info(f"Processing video: {video_name}")
        
        start_time = time.time()
        
        try:
            # Detect video type
            success, video_info = self.detect_video_type(video_path)
            if not success:
                logger.error(f"Failed to read video: {video_path}")
                return {'status': 'failed', 'error': 'Cannot read video'}
            
            # Load GPS data
            gps_df = self.load_gps_data(matched_gps_path)
            gps_df = self.add_bearing_column(gps_df)
            
            # Process based on video type
            if video_info['is_360']:
                results = self.process_360_video(video_path, gps_df, video_info)
            else:
                results = self.process_flat_video(video_path, gps_df, video_info)
            
            # Analyze audio
            if self.config.get('analyze_audio', True):
                results['audio_analysis'] = self.analyze_audio(video_path, gps_df)
            
            # Save results
            self.save_results_to_csv(results, video_name)
            
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['processed_videos'] += 1
            
            logger.info(f"Completed {video_name} in {processing_time:.2f}s")
            
            return {
                'status': 'success',
                'video_info': video_info,
                'processing_time': processing_time,
                'results_summary': {
                    'object_detections': len(results['object_tracking']),
                    'stoplight_detections': len(results['stoplight_detection']),
                    'traffic_counts': dict(results['traffic_counting']),
                    'complexity_points': len(results['scene_complexity'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            self.stats['failed_videos'] += 1
            return {'status': 'failed', 'error': str(e)}
    
    def process_matcher50_results(self, results_data: Dict[str, Any]):
        """Process all videos from matcher50 results"""
        logger.info("Starting batch processing of matched videos...")
        
        total_videos = len(results_data)
        logger.info(f"Found {total_videos} videos to process")
        
        # Process videos in parallel if requested
        if self.config.get('parallel_processing', False):
            self.process_videos_parallel(results_data)
        else:
            self.process_videos_sequential(results_data)
        
        self.generate_final_report()
    
    def process_videos_sequential(self, results_data: Dict[str, Any]):
        """Process videos sequentially"""
        for video_path, match_data in results_data.items():
            if not match_data.get('matches'):
                logger.warning(f"No matches found for {video_path}")
                continue
            
            # Use the best match (first one, assuming sorted by quality)
            best_match = match_data['matches'][0]
            gps_path = best_match['path']
            
            self.process_single_video(video_path, gps_path)
    
    def process_videos_parallel(self, results_data: Dict[str, Any]):
        """Process videos in parallel"""
        max_workers = self.config.get('max_workers', min(4, mp.cpu_count()))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for video_path, match_data in results_data.items():
                if not match_data.get('matches'):
                    continue
                
                best_match = match_data['matches'][0]
                gps_path = best_match['path']
                
                future = executor.submit(self.process_single_video, video_path, gps_path)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                except Exception as e:
                    logger.error(f"Parallel processing error: {e}")
    
    def generate_final_report(self):
        """Generate comprehensive processing report"""
        report = {
            'processing_summary': {
                'total_videos_processed': self.stats['processed_videos'],
                'total_videos_failed': self.stats['failed_videos'],
                'total_processing_time': sum(self.stats['processing_times']),
                'average_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'timestamp': datetime.now().isoformat()
            },
            'performance_metrics': {
                'videos_per_hour': (self.stats['processed_videos'] / 
                                  (sum(self.stats['processing_times']) / 3600)) if self.stats['processing_times'] else 0,
                'success_rate': (self.stats['processed_videos'] / 
                               (self.stats['processed_videos'] + self.stats['failed_videos'])) * 100 if self.stats['processed_videos'] + self.stats['failed_videos'] > 0 else 0
            }
        }
        
        report_path = self.output_dir / 'processing_reports' / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing complete! Report saved to: {report_path}")
        logger.info(f"Processed: {self.stats['processed_videos']} videos")
        logger.info(f"Failed: {self.stats['failed_videos']} videos")
        logger.info(f"Success rate: {report['performance_metrics']['success_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Video Analysis Processor for matcher50.py output"
    )
    parser.add_argument('-i', '--input', required=True,
                       help='Path to matcher50.py results file (JSON or CSV)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--yolo-model', default='yolo11x.pt',
                       help='YOLO model file (default: yolo11x.pt)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers (default: 4)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Skip audio analysis')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing (default: 10)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'output_dir': args.output,
        'yolo_model': args.yolo_model,
        'parallel_processing': args.parallel,
        'max_workers': args.max_workers,
        'analyze_audio': not args.no_audio,
        'batch_size': args.batch_size
    }
    
    # Initialize processor
    processor = VideoAnalysisProcessor(config)
    
    # Load matcher50 results
    results_data = processor.load_matcher50_results(args.input)
    
    # Process all videos
    processor.process_matcher50_results(results_data)
    
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()