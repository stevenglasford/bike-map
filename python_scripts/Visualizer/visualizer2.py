#!/usr/bin/env python3
"""
AI-Powered Video Analysis Visualizer for Matcher50.py Output
============================================================

This script takes matcher50.py JSON output and performs comprehensive AI video analysis
using YOLO object detection, tracking, and specialized processing for both flat and 360° videos.

Features:
- YOLO-based object detection and tracking
- Traffic light detection with color analysis
- Speed estimation and trajectory analysis
- Audio noise level analysis
- Scene complexity assessment
- 360° panoramic video support with region-based analysis
- GPS-synchronized analysis
- Comprehensive CSV output generation

Usage:
    python visualizer.py -i matcher50_results.json -o output_analysis/
    python visualizer.py -i results.json --filter-quality good --parallel
"""

import json
import argparse
import logging
import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# AI/ML imports
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available. Install with: pip install ultralytics torch")
    YOLO_AVAILABLE = False

# Audio analysis
try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ Audio analysis not available. Install with: pip install librosa")
    AUDIO_AVAILABLE = False

# GPS processing
try:
    import gpxpy
    from geopy.distance import distance as geopy_distance
    GPS_AVAILABLE = True
except ImportError:
    print("⚠️ GPS processing limited. Install with: pip install gpxpy geopy")
    GPS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIVideoAnalyzer:
    """AI-powered video analyzer for matcher50.py results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.setup_directories()
        self.setup_ai_models()
        
        # Statistics tracking
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_detections': 0,
            'total_frames_processed': 0,
            'processing_times': []
        }
    
    def setup_directories(self):
        """Create output directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        
        # Create analysis subdirectories
        subdirs = [
            'object_tracking', 'stoplight_detection', 'traffic_counting',
            'speed_analysis', 'scene_complexity', 'audio_analysis',
            'trajectory_analysis', 'processing_reports', 'visualizations'
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"📁 Output directory structure created: {self.output_dir}")
    
    def setup_ai_models(self):
        """Initialize AI models"""
        logger.info("🤖 Loading AI models...")
        
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO is required for video analysis. Install with: pip install ultralytics torch")
        
        # Load YOLO model
        model_path = self.config.get('yolo_model', 'yolo11x.pt')
        try:
            self.yolo_model = YOLO(model_path)
            logger.info(f"✅ YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
        logger.info(f"🎮 Using device: {self.device}")
        
        # Object class mappings (COCO classes)
        self.object_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
            'bus': 5, 'train': 6, 'truck': 7, 'traffic light': 9,
            'stop sign': 11, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16
        }
        
        # Traffic light color detection setup
        self.setup_color_detection()
        
        # Initialize tracking if available
        self.setup_tracking()
    
    def setup_color_detection(self):
        """Setup color detection for traffic lights"""
        # HSV color ranges for traffic light detection
        self.color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
            'yellow': [(15, 100, 100), (35, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)]
        }
    
    def setup_tracking(self):
        """Setup object tracking"""
        try:
            # Simple tracking using centroids
            self.active_tracks = {}
            self.next_track_id = 1
            self.max_track_distance = 100  # pixels
            logger.info("✅ Object tracking initialized")
        except Exception as e:
            logger.warning(f"⚠️ Tracking setup failed: {e}")
    
    def load_matcher50_results(self, results_path: str) -> Dict[str, Any]:
        """Load and parse matcher50.py results"""
        logger.info(f"📖 Loading matcher50 results: {results_path}")
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Handle new format
        if 'results' in data and 'processing_info' in data:
            logger.info("✅ Detected enhanced matcher50.py format")
            self.log_processing_metadata(data['processing_info'])
            return data['results']
        else:
            logger.info("⚠️ Legacy format detected")
            return data
    
    def log_processing_metadata(self, processing_info: Dict[str, Any]):
        """Log processing metadata from matcher50.py"""
        logger.info("📊 Matcher50.py Processing Metadata:")
        logger.info(f"   Version: {processing_info.get('version', 'unknown')}")
        
        if 'file_stats' in processing_info:
            stats = processing_info['file_stats']
            logger.info(f"   Videos: {stats.get('valid_videos', 0)}/{stats.get('total_videos', 0)}")
            logger.info(f"   360° videos: {stats.get('videos_360_count', 0)}")
            logger.info(f"   GPS files: {stats.get('valid_gpx', 0)}/{stats.get('total_gpx', 0)}")
        
        if 'performance_metrics' in processing_info:
            perf = processing_info['performance_metrics']
            logger.info(f"   Processing time: {perf.get('correlation_time_seconds', 0):.1f}s")
            logger.info(f"   Correlations/sec: {perf.get('correlations_per_second', 0):.1f}")
    
    def filter_matches_by_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter matches based on quality thresholds"""
        min_quality = self.config.get('min_quality', 'fair')
        min_score = self.config.get('min_score', 0.5)
        
        quality_hierarchy = {
            'excellent': 5, 'very_good': 4, 'good': 3, 
            'fair': 2, 'poor': 1, 'very_poor': 0
        }
        min_quality_level = quality_hierarchy.get(min_quality, 2)
        
        filtered_results = {}
        total_before = 0
        total_after = 0
        
        for video_path, video_data in results.items():
            if 'matches' not in video_data:
                continue
            
            matches = video_data['matches']
            total_before += len(matches)
            
            # Filter matches
            good_matches = []
            for match in matches:
                quality = match.get('quality', 'poor')
                score = match.get('combined_score', 0)
                quality_level = quality_hierarchy.get(quality, 0)
                
                if quality_level >= min_quality_level and score >= min_score:
                    good_matches.append(match)
            
            if good_matches:
                # Take only the best match for processing
                best_match = good_matches[0]
                filtered_results[video_path] = {'matches': [best_match]}
                total_after += 1
        
        reduction = ((len(results) - len(filtered_results)) / len(results) * 100) if len(results) > 0 else 0
        logger.info(f"🔍 Quality filtering: {len(results)} → {len(filtered_results)} videos ({reduction:.1f}% reduction)")
        
        return filtered_results
    
    def detect_video_type(self, video_path: str, match_info: Dict = None) -> Tuple[bool, Dict[str, Any]]:
        """Detect video type with matcher50.py intelligence"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, {}
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Use matcher50 detection if available
            is_360_from_matcher = False
            if match_info and 'matches' in match_info and match_info['matches']:
                is_360_from_matcher = match_info['matches'][0].get('is_360_video', False)
            
            # Fallback aspect ratio detection
            aspect_ratio = width / height
            is_360_manual = 1.8 <= aspect_ratio <= 2.2
            
            # Prefer matcher50 detection
            is_360 = is_360_from_matcher if match_info else is_360_manual
            
            video_info = {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'aspect_ratio': aspect_ratio,
                'is_360': is_360,
                'is_equirectangular': is_360 and aspect_ratio > 1.9,
                'detection_source': 'matcher50' if match_info else 'aspect_ratio'
            }
            
            return True, video_info
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze video {video_path}: {e}")
            return False, {}
    
    def load_gps_data(self, gps_path: str) -> pd.DataFrame:
        """Load GPS data from GPX file or CSV"""
        try:
            if gps_path.endswith('.gpx') and GPS_AVAILABLE:
                return self.parse_gpx_file(gps_path)
            elif gps_path.endswith('.csv'):
                df = pd.read_csv(gps_path)
                # Standardize column names
                if 'long' in df.columns:
                    df['lon'] = df['long']
                return df
            else:
                logger.warning(f"⚠️ Unsupported GPS format: {gps_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Failed to load GPS data from {gps_path}: {e}")
            return pd.DataFrame()
    
    def parse_gpx_file(self, gpx_path: str) -> pd.DataFrame:
        """Parse GPX file to DataFrame"""
        with open(gpx_path, 'r') as f:
            gpx = gpxpy.parse(f)
        
        records = []
        for track in gpx.tracks:
            for segment in track.segments:
                for i, point in enumerate(segment.points):
                    records.append({
                        'second': i,
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'elevation': point.elevation or 0,
                        'gpx_time': point.time
                    })
        
        df = pd.DataFrame(records)
        if len(df) > 1:
            df['speed_mph'] = self.calculate_speeds(df)
            df['bearing'] = self.calculate_bearings(df)
        else:
            df['speed_mph'] = 0
            df['bearing'] = 0
        
        return df
    
    def calculate_speeds(self, df: pd.DataFrame) -> List[float]:
        """Calculate speeds from GPS coordinates"""
        speeds = [0]  # First point has 0 speed
        
        for i in range(1, len(df)):
            prev_point = (df.iloc[i-1]['lat'], df.iloc[i-1]['lon'])
            curr_point = (df.iloc[i]['lat'], df.iloc[i]['lon'])
            
            if GPS_AVAILABLE:
                dist_meters = geopy_distance(prev_point, curr_point).meters
                speed_mph = dist_meters * 2.237  # m/s to mph (assuming 1 sec intervals)
            else:
                speed_mph = 0
            
            speeds.append(speed_mph)
        
        return speeds
    
    def calculate_bearings(self, df: pd.DataFrame) -> List[float]:
        """Calculate bearings from GPS coordinates"""
        bearings = [0]  # First point has 0 bearing
        
        for i in range(1, len(df)):
            lat1, lon1 = np.radians(df.iloc[i-1]['lat']), np.radians(df.iloc[i-1]['lon'])
            lat2, lon2 = np.radians(df.iloc[i]['lat']), np.radians(df.iloc[i]['lon'])
            
            dlon = lon2 - lon1
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            
            bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
            bearings.append(bearing)
        
        return bearings
    
    def process_video_with_ai(self, video_path: str, gps_df: pd.DataFrame, 
                             video_info: Dict, match_info: Dict) -> Dict[str, Any]:
        """Main AI video processing function"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 Processing video with AI: {video_name}")
        
        # Log match quality
        if match_info and 'matches' in match_info:
            best_match = match_info['matches'][0]
            logger.info(f"   Match quality: {best_match.get('quality', 'unknown')}")
            logger.info(f"   Match score: {best_match.get('combined_score', 0):.3f}")
        
        start_time = time.time()
        
        # Initialize results structure
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int),
            'speed_analysis': [],
            'scene_complexity': [],
            'trajectory_analysis': [],
            'match_metadata': match_info['matches'][0] if match_info and 'matches' in match_info else {}
        }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return results
        
        fps = video_info['fps']
        total_frames = video_info['frame_count']
        is_360 = video_info['is_360']
        
        # Process video based on type
        if is_360:
            logger.info("🌐 Processing as 360° panoramic video")
            results = self.process_360_video_ai(cap, gps_df, video_info, results)
        else:
            logger.info("📹 Processing as flat video")
            results = self.process_flat_video_ai(cap, gps_df, video_info, results)
        
        cap.release()
        
        # Audio analysis
        if AUDIO_AVAILABLE and self.config.get('analyze_audio', True):
            logger.info("🔊 Analyzing audio...")
            results['audio_analysis'] = self.analyze_audio(video_path, gps_df)
        
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        self.stats['processed_videos'] += 1
        
        logger.info(f"✅ Completed {video_name} in {processing_time:.2f}s")
        
        return results
    
    def process_flat_video_ai(self, cap: cv2.VideoCapture, gps_df: pd.DataFrame, 
                             video_info: Dict, results: Dict) -> Dict:
        """Process flat video with AI analysis"""
        fps = video_info['fps']
        frame_idx = 0
        
        # Tracking variables
        previous_centroids = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            second = int(frame_idx / fps)
            
            # Get GPS data for this timestamp
            gps_row = gps_df[gps_df['second'] == second] if not gps_df.empty else pd.DataFrame()
            if gps_row.empty:
                frame_idx += 1
                continue
            
            gps_data = gps_row.iloc[0]
            
            # Run YOLO detection
            detections = self.yolo_model(frame, verbose=False)[0]
            
            # Process detections
            frame_results = self.process_frame_detections(
                detections, frame, second, gps_data, video_info, 'main'
            )
            
            # Update tracking
            frame_results = self.update_tracking(frame_results, previous_centroids)
            
            # Aggregate results
            results['object_tracking'].extend(frame_results['tracking'])
            results['stoplight_detection'].extend(frame_results['stoplights'])
            results['speed_analysis'].extend(frame_results['speeds'])
            
            for obj_type, count in frame_results['counting'].items():
                results['traffic_counting'][obj_type] += count
            
            # Scene complexity
            complexity = self.calculate_scene_complexity(frame, second, gps_data)
            results['scene_complexity'].append(complexity)
            
            frame_idx += 1
            self.stats['total_frames_processed'] += 1
            
            # Progress logging
            if frame_idx % 100 == 0:
                progress = (frame_idx / video_info['frame_count']) * 100
                logger.info(f"   Progress: {progress:.1f}% ({frame_idx}/{video_info['frame_count']} frames)")
        
        return results
    
    def process_360_video_ai(self, cap: cv2.VideoCapture, gps_df: pd.DataFrame, 
                            video_info: Dict, results: Dict) -> Dict:
        """Process 360° video with specialized AI analysis"""
        fps = video_info['fps']
        width, height = video_info['width'], video_info['height']
        frame_idx = 0
        
        # Define 360° regions for analysis
        regions = self.define_360_regions(width, height)
        previous_centroids = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            second = int(frame_idx / fps)
            
            # Get GPS data
            gps_row = gps_df[gps_df['second'] == second] if not gps_df.empty else pd.DataFrame()
            if gps_row.empty:
                frame_idx += 1
                continue
            
            gps_data = gps_row.iloc[0]
            
            # Process each region of 360° video
            for region_name, region_coords in regions.items():
                region_frame = self.extract_region(frame, region_coords)
                
                # Run YOLO on region
                detections = self.yolo_model(region_frame, verbose=False)[0]
                
                # Process detections for this region
                region_results = self.process_frame_detections(
                    detections, region_frame, second, gps_data, video_info, region_name
                )
                
                # Adjust coordinates for full frame context
                region_results = self.adjust_coordinates_for_region(region_results, region_coords)
                
                # Aggregate results
                results['object_tracking'].extend(region_results['tracking'])
                results['stoplight_detection'].extend(region_results['stoplights'])
                
                for obj_type, count in region_results['counting'].items():
                    results['traffic_counting'][obj_type] += count
            
            # Scene complexity for full 360° frame
            complexity = self.calculate_scene_complexity(frame, second, gps_data)
            results['scene_complexity'].append(complexity)
            
            frame_idx += 1
            self.stats['total_frames_processed'] += 1
            
            if frame_idx % 50 == 0:  # Less frequent for 360° (more processing)
                progress = (frame_idx / video_info['frame_count']) * 100
                logger.info(f"   360° Progress: {progress:.1f}% ({frame_idx}/{video_info['frame_count']} frames)")
        
        return results
    
    def define_360_regions(self, width: int, height: int) -> Dict[str, Tuple]:
        """Define analysis regions for 360° video"""
        # Equirectangular projection regions
        return {
            'front': (width//4, 0, 3*width//4, height),           # Center view
            'left': (0, height//4, width//4, 3*height//4),        # Left view
            'right': (3*width//4, height//4, width, 3*height//4), # Right view
            'back_left': (0, height//4, width//8, 3*height//4),   # Back regions
            'back_right': (7*width//8, height//4, width, 3*height//4)
        }
    
    def extract_region(self, frame: np.ndarray, coords: Tuple) -> np.ndarray:
        """Extract region from frame"""
        x1, y1, x2, y2 = coords
        return frame[y1:y2, x1:x2]
    
    def adjust_coordinates_for_region(self, results: Dict, region_coords: Tuple) -> Dict:
        """Adjust detection coordinates back to full frame"""
        x_offset, y_offset = region_coords[0], region_coords[1]
        
        # Adjust bounding boxes in tracking results
        for track in results['tracking']:
            if 'bbox' in track:
                bbox = track['bbox']
                track['bbox'] = [
                    bbox[0] + x_offset, bbox[1] + y_offset,
                    bbox[2] + x_offset, bbox[3] + y_offset
                ]
        
        return results
    
    def process_frame_detections(self, detections, frame: np.ndarray, second: int, 
                               gps_data: pd.Series, video_info: Dict, region_name: str) -> Dict:
        """Process YOLO detections for a single frame"""
        results = {
            'tracking': [],
            'stoplights': [],
            'speeds': [],
            'counting': defaultdict(int)
        }
        
        if detections.boxes is None or len(detections.boxes) == 0:
            return results
        
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy().astype(int)
        confidences = detections.boxes.conf.cpu().numpy()
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf < self.config.get('confidence_threshold', 0.3):
                continue
            
            x1, y1, x2, y2 = box.astype(int)
            
            # Get class name
            class_names = self.yolo_model.names
            obj_class = class_names.get(cls, 'unknown')
            
            # Count detection
            results['counting'][obj_class] += 1
            self.stats['total_detections'] += 1
            
            # Create detection record
            detection_data = {
                'frame_second': second,
                'object_class': obj_class,
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'region': region_name,
                'lat': float(gps_data.get('lat', 0)),
                'lon': float(gps_data.get('lon', 0)),
                'gps_speed': float(gps_data.get('speed_mph', 0)),
                'gps_bearing': float(gps_data.get('bearing', 0)),
                'gps_time': str(gps_data.get('gpx_time', '')),
                'video_type': '360°' if video_info.get('is_360', False) else 'flat'
            }
            
            # Add to tracking
            results['tracking'].append(detection_data)
            
            # Special processing for traffic lights
            if obj_class == 'traffic light':
                stoplight_data = self.analyze_traffic_light(
                    frame, box, second, gps_data, video_info, region_name
                )
                if stoplight_data:
                    results['stoplights'].append(stoplight_data)
        
        return results
    
    def analyze_traffic_light(self, frame: np.ndarray, box: np.ndarray, 
                             second: int, gps_data: pd.Series, video_info: Dict, 
                             region_name: str) -> Optional[Dict]:
        """Analyze traffic light color and state"""
        x1, y1, x2, y2 = box.astype(int)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Detect color
        color = self.detect_light_color(roi)
        
        # Calculate bearing for 360° videos
        if video_info.get('is_360', False):
            bearing = self.calculate_360_bearing(x1, x2, video_info['width'], region_name)
        else:
            bearing = gps_data.get('bearing', 0)
        
        return {
            'second': second,
            'stoplight_color': color,
            'confidence': 0.8,
            'bearing': bearing,
            'region': region_name,
            'bbox': [x1, y1, x2, y2],
            'lat': float(gps_data.get('lat', 0)),
            'lon': float(gps_data.get('lon', 0)),
            'gps_time': str(gps_data.get('gpx_time', '')),
            'video_type': '360°' if video_info.get('is_360', False) else 'flat'
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
        """Calculate bearing for 360° video detection"""
        center_x = (x1 + x2) / 2
        bearing = (center_x / frame_width) * 360
        
        # Adjust for region
        region_offsets = {
            'front': 0, 'left': 270, 'right': 90,
            'back_left': 180, 'back_right': 180
        }
        
        bearing = (bearing + region_offsets.get(region_name, 0)) % 360
        return bearing
    
    def update_tracking(self, frame_results: Dict, previous_centroids: Dict) -> Dict:
        """Simple centroid-based tracking"""
        for detection in frame_results['tracking']:
            bbox = detection['bbox']
            centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            obj_class = detection['object_class']
            
            # Simple tracking by finding closest previous centroid
            min_distance = float('inf')
            track_id = None
            
            for tid, prev_centroid in previous_centroids.items():
                if prev_centroid['class'] == obj_class:
                    distance = np.sqrt((centroid[0] - prev_centroid['centroid'][0])**2 + 
                                     (centroid[1] - prev_centroid['centroid'][1])**2)
                    if distance < min_distance and distance < self.max_track_distance:
                        min_distance = distance
                        track_id = tid
            
            # Assign track ID
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            
            detection['track_id'] = track_id
            previous_centroids[track_id] = {
                'centroid': centroid,
                'class': obj_class,
                'last_seen': detection['frame_second']
            }
        
        return frame_results
    
    def calculate_scene_complexity(self, frame: np.ndarray, second: int, gps_data: pd.Series) -> Dict:
        """Calculate scene complexity metrics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture variance
        texture_var = np.var(gray)
        
        # Color variance
        color_var = np.var(frame)
        
        # Gradient magnitude
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
            'lat': float(gps_data.get('lat', 0)),
            'lon': float(gps_data.get('lon', 0)),
            'gps_time': str(gps_data.get('gpx_time', ''))
        }
    
    def analyze_audio(self, video_path: str, gps_df: pd.DataFrame) -> List[Dict]:
        """Analyze audio track for noise levels"""
        if not AUDIO_AVAILABLE:
            logger.warning("⚠️ Audio analysis not available")
            return []
        
        try:
            logger.info("🔊 Analyzing audio track...")
            y, sr = librosa.load(video_path, sr=None)
            
            hop_length = sr // 2  # 0.5 second windows
            
            # Audio features
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            
            audio_results = []
            
            for i, (rms_val, centroid_val, zcr_val) in enumerate(zip(rms, spectral_centroid, zcr)):
                second = int(i * 0.5)
                
                # Get GPS data for this time
                gps_row = gps_df[gps_df['second'] == second] if not gps_df.empty else pd.DataFrame()
                if gps_row.empty:
                    continue
                
                gps_data = gps_row.iloc[0]
                
                audio_results.append({
                    'second': second,
                    'rms_energy': float(rms_val),
                    'spectral_centroid': float(centroid_val),
                    'zero_crossing_rate': float(zcr_val),
                    'noise_level': float(rms_val * 100),
                    'lat': float(gps_data.get('lat', 0)),
                    'lon': float(gps_data.get('lon', 0)),
                    'gps_time': str(gps_data.get('gpx_time', ''))
                })
            
            return audio_results
            
        except Exception as e:
            logger.warning(f"⚠️ Audio analysis failed: {e}")
            return []
    
    def save_results_to_csv(self, results: Dict[str, Any], video_name: str):
        """Save comprehensive analysis results to CSV files"""
        
        # Object tracking with match metadata
        if results['object_tracking']:
            df = pd.DataFrame(results['object_tracking'])
            
            # Add match metadata
            match_meta = results.get('match_metadata', {})
            for key, value in match_meta.items():
                df[f'match_{key}'] = value
            
            output_file = self.output_dir / 'object_tracking' / f"{video_name}_tracking.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved object tracking: {output_file}")
        
        # Stoplight detection
        if results['stoplight_detection']:
            df = pd.DataFrame(results['stoplight_detection'])
            
            # Add match metadata
            match_meta = results.get('match_metadata', {})
            for key, value in match_meta.items():
                df[f'match_{key}'] = value
            
            output_file = self.output_dir / 'stoplight_detection' / f"{video_name}_stoplights.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"🚦 Saved stoplight analysis: {output_file}")
        
        # Traffic counting
        if results['traffic_counting']:
            counting_data = []
            for obj_type, count in results['traffic_counting'].items():
                row = {'video_name': video_name, 'object_type': obj_type, 'total_count': count}
                
                # Add match metadata
                match_meta = results.get('match_metadata', {})
                row.update({f'match_{k}': v for k, v in match_meta.items()})
                
                counting_data.append(row)
            
            df = pd.DataFrame(counting_data)
            output_file = self.output_dir / 'traffic_counting' / f"{video_name}_counts.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"📊 Saved traffic counting: {output_file}")
        
        # Scene complexity
        if results['scene_complexity']:
            df = pd.DataFrame(results['scene_complexity'])
            
            # Add match metadata
            match_meta = results.get('match_metadata', {})
            for key, value in match_meta.items():
                df[f'match_{key}'] = value
            
            output_file = self.output_dir / 'scene_complexity' / f"{video_name}_complexity.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"🎨 Saved scene complexity: {output_file}")
        
        # Audio analysis
        if 'audio_analysis' in results and results['audio_analysis']:
            df = pd.DataFrame(results['audio_analysis'])
            
            # Add match metadata
            match_meta = results.get('match_metadata', {})
            for key, value in match_meta.items():
                df[f'match_{key}'] = value
            
            output_file = self.output_dir / 'audio_analysis' / f"{video_name}_audio.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"🔊 Saved audio analysis: {output_file}")
    
    def process_single_video(self, video_path: str, match_info: Dict) -> Dict[str, Any]:
        """Process a single video with AI analysis"""
        video_name = Path(video_path).stem
        
        try:
            # Get best GPS match
            if not match_info or 'matches' not in match_info or not match_info['matches']:
                logger.warning(f"⚠️ No valid matches for {video_name}")
                return {'status': 'failed', 'error': 'No valid matches'}
            
            best_match = match_info['matches'][0]
            gps_path = best_match['path']
            
            # Detect video type
            success, video_info = self.detect_video_type(video_path, match_info)
            if not success:
                logger.error(f"❌ Failed to analyze video: {video_path}")
                return {'status': 'failed', 'error': 'Cannot analyze video'}
            
            # Load GPS data
            gps_df = self.load_gps_data(gps_path)
            if gps_df.empty:
                logger.warning(f"⚠️ No GPS data loaded from {gps_path}")
            
            # Process video with AI
            results = self.process_video_with_ai(video_path, gps_df, video_info, match_info)
            
            # Save results
            self.save_results_to_csv(results, video_name)
            
            return {
                'status': 'success',
                'video_info': video_info,
                'results_summary': {
                    'object_detections': len(results['object_tracking']),
                    'stoplight_detections': len(results['stoplight_detection']),
                    'traffic_counts': dict(results['traffic_counting']),
                    'complexity_points': len(results['scene_complexity'])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {video_path}: {e}")
            self.stats['failed_videos'] += 1
            return {'status': 'failed', 'error': str(e)}
    
    def run_analysis(self, results_data: Dict[str, Any]):
        """Run AI analysis on all matched videos"""
        logger.info("🚀 Starting AI video analysis pipeline...")
        
        # Filter matches if requested
        if self.config.get('filter_matches', True):
            results_data = self.filter_matches_by_quality(results_data)
        
        total_videos = len(results_data)
        logger.info(f"🎬 Processing {total_videos} videos with AI analysis")
        
        if total_videos == 0:
            logger.warning("⚠️ No videos to process after filtering")
            return
        
        # Process videos
        if self.config.get('parallel_processing', False):
            self.process_videos_parallel(results_data)
        else:
            self.process_videos_sequential(results_data)
        
        # Generate final report
        self.generate_final_report()
    
    def process_videos_sequential(self, results_data: Dict[str, Any]):
        """Process videos sequentially"""
        for i, (video_path, match_info) in enumerate(results_data.items(), 1):
            logger.info(f"📹 Processing video {i}/{len(results_data)}: {Path(video_path).name}")
            self.process_single_video(video_path, match_info)
    
    def process_videos_parallel(self, results_data: Dict[str, Any]):
        """Process videos in parallel"""
        max_workers = self.config.get('max_workers', min(4, mp.cpu_count()))
        logger.info(f"🔄 Using {max_workers} parallel workers")
        
        # Note: Parallel processing would need careful GPU memory management
        # For now, fall back to sequential for stability
        logger.info("🔄 Using sequential processing for GPU stability")
        self.process_videos_sequential(results_data)
    
    def generate_final_report(self):
        """Generate comprehensive processing report"""
        total_time = sum(self.stats['processing_times'])
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        report = {
            'processing_summary': {
                'total_videos_processed': self.stats['processed_videos'],
                'total_videos_failed': self.stats['failed_videos'],
                'total_processing_time': total_time,
                'average_processing_time': avg_time,
                'total_frames_processed': self.stats['total_frames_processed'],
                'total_detections': self.stats['total_detections'],
                'timestamp': datetime.now().isoformat()
            },
            'performance_metrics': {
                'videos_per_hour': (self.stats['processed_videos'] / (total_time / 3600)) if total_time > 0 else 0,
                'frames_per_second': (self.stats['total_frames_processed'] / total_time) if total_time > 0 else 0,
                'detections_per_video': (self.stats['total_detections'] / self.stats['processed_videos']) if self.stats['processed_videos'] > 0 else 0,
                'success_rate': (self.stats['processed_videos'] / (self.stats['processed_videos'] + self.stats['failed_videos'])) * 100 if (self.stats['processed_videos'] + self.stats['failed_videos']) > 0 else 0
            },
            'configuration': self.config
        }
        
        report_path = self.output_dir / 'processing_reports' / 'ai_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Final report saved: {report_path}")
        logger.info(f"✅ AI Analysis Complete!")
        logger.info(f"   Processed: {self.stats['processed_videos']} videos")
        logger.info(f"   Failed: {self.stats['failed_videos']} videos")
        logger.info(f"   Total detections: {self.stats['total_detections']:,}")
        logger.info(f"   Success rate: {report['performance_metrics']['success_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Video Analysis for matcher50.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic AI analysis
  python visualizer.py -i matcher50_results.json -o ai_analysis/
  
  # High-quality matches only
  python visualizer.py -i results.json -o output/ --filter-quality good --min-score 0.6
  
  # Parallel processing with custom YOLO model
  python visualizer.py -i results.json -o output/ --parallel --yolo-model yolo11n.pt
  
  # Skip audio analysis for faster processing
  python visualizer.py -i results.json -o output/ --no-audio
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path to matcher50.py results JSON file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for AI analysis results')
    parser.add_argument('--yolo-model', default='yolo11x.pt',
                       help='YOLO model to use (default: yolo11x.pt)')
    parser.add_argument('--filter-quality', choices=['poor', 'fair', 'good', 'very_good', 'excellent'],
                       default='fair', help='Minimum match quality to process')
    parser.add_argument('--min-score', type=float, default=0.5,
                       help='Minimum match score to process')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='YOLO confidence threshold')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing (experimental)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers')
    parser.add_argument('--no-audio', action='store_true',
                       help='Skip audio analysis')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU processing')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Configuration
    config = {
        'output_dir': args.output,
        'yolo_model': args.yolo_model,
        'min_quality': args.filter_quality,
        'min_score': args.min_score,
        'confidence_threshold': args.confidence_threshold,
        'parallel_processing': args.parallel,
        'max_workers': args.max_workers,
        'analyze_audio': not args.no_audio,
        'use_gpu': not args.no_gpu,
        'filter_matches': True
    }
    
    # Initialize analyzer
    try:
        analyzer = AIVideoAnalyzer(config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI analyzer: {e}")
        sys.exit(1)
    
    # Load matcher50 results
    try:
        results_data = analyzer.load_matcher50_results(args.input)
    except Exception as e:
        logger.error(f"❌ Failed to load matcher50 results: {e}")
        sys.exit(1)
    
    # Run AI analysis
    analyzer.run_analysis(results_data)
    
    logger.info("🎉 AI video analysis complete!")


if __name__ == "__main__":
    main()
