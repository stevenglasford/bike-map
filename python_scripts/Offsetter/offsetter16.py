#!/usr/bin/env python3
"""
⚡ ENHANCED DUAL GPU - 360° + FLAT VIDEO PROCESSOR ⚡
🎯 INTELLIGENT DETECTION & PROCESSING FOR PANORAMIC & FLAT VIDEOS
🔥 ADAPTIVE MOTION DETECTION, MAXIMUM ACCURACY 🔥
"""

import json
import numpy as np
import cupy as cp
import cv2
import gpxpy
import pandas as pd
from pathlib import Path
import argparse
import logging
import sys
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import traceback
import math

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enhanced_360_processor.log', mode='w')
        ]
    )
    return logging.getLogger('enhanced_360_processor')

class VideoTypeDetector:
    """Intelligent video type detection for 360° vs flat videos"""
    
    @staticmethod
    def detect_video_type(video_path: str, cap: cv2.VideoCapture) -> Dict:
        """Detect if video is 360° panoramic or flat with confidence scoring"""
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width <= 0 or height <= 0:
                return {'type': 'unknown', 'confidence': 0.0, 'width': width, 'height': height}
            
            aspect_ratio = width / height
            confidence_score = 0.0
            video_type = 'flat'
            
            # Method 1: Aspect ratio analysis
            if abs(aspect_ratio - 2.0) < 0.1:  # 2:1 ratio typical for equirectangular
                confidence_score += 0.4
                video_type = '360'
            elif 1.9 <= aspect_ratio <= 2.1:  # Close to 2:1
                confidence_score += 0.3
                video_type = '360'
            elif 1.7 <= aspect_ratio <= 1.8:  # 16:9 typical for flat
                confidence_score += 0.3
                video_type = 'flat'
            
            # Method 2: Resolution patterns
            if (width == 3840 and height == 1920) or (width == 1920 and height == 960):  # Common 360 resolutions
                confidence_score += 0.3
                video_type = '360'
            elif (width == 1920 and height == 1080) or (width == 1280 and height == 720):  # Common flat resolutions
                confidence_score += 0.2
                video_type = 'flat'
            
            # Method 3: Filename analysis
            filename = Path(video_path).name.lower()
            if any(keyword in filename for keyword in ['360', 'vr', 'spherical', 'equirect', 'panoramic']):
                confidence_score += 0.2
                video_type = '360'
            elif any(keyword in filename for keyword in ['flat', 'standard', 'regular']):
                confidence_score += 0.1
                video_type = 'flat'
            
            # Method 4: Content analysis (sample a few frames)
            try:
                # Check if edges show typical 360 distortion patterns
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Look for characteristic 360 video patterns
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Check edge distortion (360 videos often have heavy distortion at top/bottom)
                    top_row = gray[0, :]
                    bottom_row = gray[-1, :]
                    top_variance = np.var(top_row)
                    bottom_variance = np.var(bottom_row)
                    center_variance = np.var(gray[height//2, :])
                    
                    if center_variance > 0 and (top_variance < center_variance * 0.3 or bottom_variance < center_variance * 0.3):
                        confidence_score += 0.1
                        if video_type == 'flat':
                            video_type = '360'
                    
                    # Check horizontal wrapping (left edge similar to right edge)
                    left_col = gray[:, 0]
                    right_col = gray[:, -1]
                    edge_correlation = np.corrcoef(left_col, right_col)[0, 1]
                    if not np.isnan(edge_correlation) and edge_correlation > 0.7:
                        confidence_score += 0.1
                        if video_type == 'flat':
                            video_type = '360'
                        
            except Exception:
                pass
            
            # Ensure confidence is between 0 and 1
            confidence_score = min(1.0, max(0.0, confidence_score))
            
            # If confidence is low, use aspect ratio as primary indicator
            if confidence_score < 0.5:
                if aspect_ratio >= 1.8:
                    video_type = '360'
                    confidence_score = max(0.6, confidence_score)
                else:
                    video_type = 'flat'
                    confidence_score = max(0.6, confidence_score)
            
            return {
                'type': video_type,
                'confidence': confidence_score,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio
            }
            
        except Exception as e:
            return {'type': 'unknown', 'confidence': 0.0, 'error': str(e)}

class Enhanced360Worker:
    """Enhanced worker with 360° and flat video support"""
    
    def __init__(self, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, gpu_memory_gb: float = 15.0):
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.errors = 0
        self.logger = logging.getLogger(f'Enhanced360_GPU{gpu_id}')
        self.is_running = True
        
        # Enhanced GPU memory management
        self.gpu_frame_buffer = None
        self.gpu_temp_arrays = {}
        self.optical_flow_cache = {}
        
    def run(self):
        """Enhanced worker loop with 360° support"""
        
        self.logger.info(f"🎯 Enhanced GPU {self.gpu_id} STARTING with 360° support...")
        
        try:
            # GPU initialization
            cp.cuda.Device(self.gpu_id).use()
            
            if self.gpu_memory_gb > 0:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.gpu_memory_gb * 1024**3))
            
            # Pre-allocate enhanced GPU arrays
            self.gpu_temp_arrays = {
                'frame_buffer': cp.zeros((480, 960), dtype=cp.float32),  # Larger for 360 videos
                'motion_buffer': cp.zeros(500, dtype=cp.float32),
                'optical_flow_x': cp.zeros((240, 480), dtype=cp.float32),
                'optical_flow_y': cp.zeros((240, 480), dtype=cp.float32),
                'magnitude_buffer': cp.zeros((240, 480), dtype=cp.float32)
            }
            
            # GPU test
            test = cp.array([1, 2, 3])
            cp.sum(test)
            del test
            self.logger.info(f"🎯 Enhanced GPU {self.gpu_id} READY with 360° support!")
            
            # Main processing loop
            while self.is_running:
                try:
                    work_item = self.work_queue.get(timeout=2)
                    
                    if work_item is None:
                        break
                    
                    video_path, gpx_path, match = work_item
                    
                    # Quick validation
                    if not self.quick_validate_files(video_path, gpx_path):
                        error_result = self.create_error_result(match, "file_invalid")
                        self.result_queue.put(error_result)
                        self.work_queue.task_done()
                        self.errors += 1
                        continue
                    
                    # Enhanced processing
                    start_time = time.time()
                    result = self.enhanced_process(video_path, gpx_path, match)
                    processing_time = time.time() - start_time
                    
                    self.result_queue.put(result)
                    self.processed += 1
                    
                    if self.processed % 5 == 0:
                        self.logger.info(f"🎯 Enhanced GPU {self.gpu_id}: {self.processed} done ({processing_time:.1f}s)")
                    
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"💀 Enhanced GPU {self.gpu_id}: {e}")
                    error_result = self.create_error_result(match, f"processing_error")
                    self.result_queue.put(error_result)
                    self.work_queue.task_done()
                    self.errors += 1
        
        except Exception as e:
            self.logger.error(f"💀 Enhanced GPU {self.gpu_id} FATAL: {e}")
        
        finally:
            # Cleanup
            for key in list(self.gpu_temp_arrays.keys()):
                del self.gpu_temp_arrays[key]
            self.gpu_temp_arrays.clear()
            
            self.logger.info(f"🎯 Enhanced GPU {self.gpu_id} SHUTDOWN: {self.processed} processed, {self.errors} errors")
    
    def quick_validate_files(self, video_path: str, gpx_path: str) -> bool:
        """Quick file validation"""
        try:
            return Path(video_path).exists() and Path(gpx_path).exists()
        except:
            return False
    
    def create_error_result(self, match: Dict, error_type: str) -> Dict:
        """Create standardized error result"""
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': False,
            'enhanced_360_mode': True,
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': f'enhanced_gpu_{self.gpu_id}_{error_type}',
            'sync_quality': 'failed'
        })
        return result
    
    def enhanced_process(self, video_path: str, gpx_path: str, match: Dict) -> Dict:
        """Enhanced processing with 360° and flat video support"""
        
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': True,
            'enhanced_360_mode': True,
            'offset_method': f'enhanced_gpu_{self.gpu_id}'
        })
        
        try:
            cp.cuda.Device(self.gpu_id).use()
            
            # Enhanced video extraction with type detection
            video_motion, video_info = self.enhanced_extract_video(video_path)
            if video_motion is None:
                result['offset_method'] = f'enhanced_gpu_{self.gpu_id}_video_failed'
                return result
            
            # Add video type info to result
            result.update({
                'video_type': video_info.get('type', 'unknown'),
                'video_type_confidence': video_info.get('confidence', 0.0),
                'video_resolution': f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"
            })
            
            # Enhanced GPS extraction
            gps_speed = self.enhanced_extract_gps(gpx_path)
            if gps_speed is None:
                result['offset_method'] = f'enhanced_gpu_{self.gpu_id}_gps_failed'
                return result
            
            # Enhanced offset calculation
            offset, confidence = self.enhanced_calculate_offset(video_motion, gps_speed, video_info)
            
            if offset is not None and confidence >= 0.25:
                result.update({
                    'temporal_offset_seconds': float(offset),
                    'offset_confidence': float(confidence),
                    'offset_method': f'enhanced_gpu_{self.gpu_id}_success_{video_info.get("type", "unknown")}',
                    'sync_quality': 'excellent' if confidence >= 0.8 else 'good' if confidence >= 0.6 else 'fair'
                })
            else:
                result.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': float(confidence) if confidence else 0.0,
                    'offset_method': f'enhanced_gpu_{self.gpu_id}_low_confidence',
                    'sync_quality': 'poor'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"💀 Enhanced GPU {self.gpu_id}: Processing error: {e}")
            result.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'enhanced_gpu_{self.gpu_id}_exception',
                'gpu_processing': False
            })
            return result
    
    def enhanced_extract_video(self, video_path: str) -> Tuple[Optional[cp.ndarray], Dict]:
        """Enhanced video extraction with 360° support"""
        
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None, {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps <= 0 or frame_count <= 0:
                return None, {}
            
            # Detect video type
            video_info = VideoTypeDetector.detect_video_type(video_path, cap)
            video_type = video_info.get('type', 'unknown')
            
            self.logger.debug(f"Detected video type: {video_type} (confidence: {video_info.get('confidence', 0):.2f})")
            
            # Adaptive sampling based on video duration and type
            duration = frame_count / fps
            if video_type == '360':
                # 360 videos need more samples due to complex motion patterns
                if duration <= 60:
                    sample_interval = 0.8  # Sample every 0.8 seconds
                elif duration <= 300:
                    sample_interval = 1.5
                else:
                    sample_interval = 2.5
            else:
                # Flat videos can use less frequent sampling
                if duration <= 60:
                    sample_interval = 1.0
                elif duration <= 300:
                    sample_interval = 2.0
                else:
                    sample_interval = 3.0
            
            frame_interval = max(1, int(fps * sample_interval))
            
            # Adaptive resolution based on video type and size
            original_width = video_info.get('width', 1920)
            original_height = video_info.get('height', 1080)
            
            if video_type == '360':
                # Maintain aspect ratio for 360 videos, larger resolution for better motion detection
                if original_width > 3840:  # 8K 360
                    target_width = 480
                elif original_width > 1920:  # 4K 360
                    target_width = 320
                else:  # 2K or lower 360
                    target_width = 240
            else:
                # Standard processing for flat videos
                if original_width > 1920:
                    target_width = 240
                else:
                    target_width = 160
            
            target_height = int(target_width * original_height / original_width)
            target_height = target_height - (target_height % 2)  # Make even
            
            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Extract motion with appropriate method
            if video_type == '360':
                motion_values = self.extract_360_motion(cap, target_width, target_height, frame_interval, duration)
            else:
                motion_values = self.extract_flat_motion(cap, target_width, target_height, frame_interval, duration)
            
            if motion_values is not None and len(motion_values) >= 3:
                return cp.array(motion_values, dtype=cp.float32), video_info
            else:
                return None, video_info
                
        except Exception as e:
            self.logger.error(f"Enhanced video extraction error: {e}")
            return None, {}
        finally:
            if cap:
                cap.release()
    
    def extract_360_motion(self, cap: cv2.VideoCapture, target_width: int, target_height: int, frame_interval: int, duration: float) -> Optional[List[float]]:
        """Enhanced motion extraction for 360° videos using optical flow and regional analysis"""
        
        motion_values = []
        frame_idx = 0
        prev_gray = None
        max_frames = min(400, int(duration / 2) + 20)  # More frames for 360
        max_time = 12  # Slightly more time for 360 processing
        start_time = time.time()
        
        try:
            while len(motion_values) < max_frames:
                if time.time() - start_time > max_time:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        # Resize maintaining aspect ratio
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        
                        if prev_gray is not None:
                            # Method 1: Optical flow for 360 videos (more accurate for panoramic motion)
                            try:
                                flow = cv2.calcOpticalFlowPyrLK(
                                    prev_gray, gray, 
                                    np.float32([[x, y] for y in range(0, target_height, 8) for x in range(0, target_width, 8)]).reshape(-1, 1, 2),
                                    None,
                                    winSize=(15, 15),
                                    maxLevel=2,
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                                )[0]
                                
                                if flow is not None and len(flow) > 0:
                                    # Calculate optical flow magnitude
                                    flow_magnitude = np.sqrt(flow[:, 0, 0]**2 + flow[:, 0, 1]**2)
                                    motion_optical = np.mean(flow_magnitude[flow_magnitude > 0.5])  # Filter small movements
                                    
                                    if not np.isnan(motion_optical):
                                        motion_values.append(float(motion_optical))
                                    else:
                                        # Fallback to frame difference
                                        motion_values.append(self.calculate_360_frame_diff(prev_gray, gray))
                                else:
                                    # Fallback to frame difference
                                    motion_values.append(self.calculate_360_frame_diff(prev_gray, gray))
                                    
                            except Exception:
                                # Fallback to enhanced frame difference for 360
                                motion_values.append(self.calculate_360_frame_diff(prev_gray, gray))
                        
                        prev_gray = gray.copy()
                        
                    except Exception:
                        continue
                
                frame_idx += 1
            
            return motion_values if len(motion_values) >= 3 else None
            
        except Exception:
            return None
    
    def calculate_360_frame_diff(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Enhanced frame difference calculation for 360° videos"""
        try:
            # Convert to GPU for faster processing
            curr_gpu = cp.asarray(curr_gray, dtype=cp.float32)
            prev_gpu = cp.asarray(prev_gray, dtype=cp.float32)
            
            # Regional analysis for 360 videos
            height, width = curr_gray.shape
            
            # Analyze different regions with different weights
            # Center region (more important in 360 videos)
            center_h_start, center_h_end = height // 4, 3 * height // 4
            center_w_start, center_w_end = width // 4, 3 * width // 4
            
            center_diff = cp.mean(cp.abs(
                curr_gpu[center_h_start:center_h_end, center_w_start:center_w_end] - 
                prev_gpu[center_h_start:center_h_end, center_w_start:center_w_end]
            ))
            
            # Edge regions (less important but still relevant)
            edge_diff = cp.mean(cp.abs(curr_gpu - prev_gpu))
            
            # Weighted combination (center region weighted more heavily)
            motion = float(0.7 * center_diff + 0.3 * edge_diff)
            
            # Cleanup
            del curr_gpu, prev_gpu
            
            return motion
            
        except Exception:
            return 0.0
    
    def extract_flat_motion(self, cap: cv2.VideoCapture, target_width: int, target_height: int, frame_interval: int, duration: float) -> Optional[List[float]]:
        """Optimized motion extraction for flat videos"""
        
        motion_values = []
        frame_idx = 0
        prev_gray = None
        max_frames = min(300, int(duration / 2) + 10)
        max_time = 10
        start_time = time.time()
        
        try:
            while len(motion_values) < max_frames:
                if time.time() - start_time > max_time:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        
                        if prev_gray is not None:
                            # Standard frame difference for flat videos
                            curr_gpu = cp.asarray(gray, dtype=cp.float32)
                            prev_gpu = cp.asarray(prev_gray, dtype=cp.float32)
                            
                            diff = cp.abs(curr_gpu - prev_gpu)
                            motion = float(cp.mean(diff))
                            motion_values.append(motion)
                            
                            del curr_gpu, prev_gpu, diff
                        
                        prev_gray = gray
                        
                    except Exception:
                        continue
                
                frame_idx += 1
            
            return motion_values if len(motion_values) >= 3 else None
            
        except Exception:
            return None
    
    def enhanced_extract_gps(self, gpx_path: str) -> Optional[cp.ndarray]:
        """Enhanced GPS extraction with better preprocessing"""
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            # Collect points more efficiently
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time and point.latitude and point.longitude:
                            points.append({
                                'lat': point.latitude,
                                'lon': point.longitude,
                                'time': point.time
                            })
            
            if len(points) < 10:
                return None
            
            points.sort(key=lambda p: p['time'])
            
            # Enhanced downsampling
            total_duration = (points[-1]['time'] - points[0]['time']).total_seconds()
            if total_duration <= 0:
                return None
            
            # Adaptive target based on duration and point density
            point_density = len(points) / total_duration  # points per second
            
            if point_density > 2:  # High frequency GPS
                target_points = min(len(points), max(60, int(total_duration / 1.5)))
            else:  # Standard GPS
                target_points = min(len(points), max(50, int(total_duration / 2)))
            
            if len(points) > target_points:
                step = len(points) // target_points
                indices = list(range(0, len(points), step))
                if indices[-1] != len(points) - 1:
                    indices.append(len(points) - 1)
                points = [points[i] for i in indices]
            
            df = pd.DataFrame(points)
            
            # Enhanced GPU calculations
            lats = cp.array(df['lat'].values, dtype=cp.float64)  # Higher precision for GPS
            lons = cp.array(df['lon'].values, dtype=cp.float64)
            
            # Enhanced Haversine calculation
            lat1 = lats[:-1]
            lat2 = lats[1:]
            lon1 = lons[:-1]
            lon2 = lons[1:]
            
            lat1_rad = cp.radians(lat1)
            lat2_rad = cp.radians(lat2)
            lon1_rad = cp.radians(lon1)
            lon2_rad = cp.radians(lon2)
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
            distances = 6371000 * 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))
            
            # Enhanced time differences
            time_diffs = cp.array([
                max((df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds(), 0.1)
                for i in range(len(df)-1)
            ], dtype=cp.float32)
            
            # Speed calculation with enhanced outlier removal
            speeds = distances / time_diffs
            
            # Multi-level outlier removal
            # Level 1: Remove extreme outliers (>300 km/h)
            valid_mask1 = speeds < 83.3  # 300 km/h in m/s
            
            # Level 2: Statistical outlier removal
            if cp.sum(valid_mask1) > 5:
                valid_speeds = speeds[valid_mask1]
                median_speed = cp.median(valid_speeds)
                mad = cp.median(cp.abs(valid_speeds - median_speed))
                threshold = median_speed + 3 * mad * 1.4826  # Modified Z-score
                valid_mask2 = speeds < threshold
                final_mask = valid_mask1 & valid_mask2
            else:
                final_mask = valid_mask1
            
            # Ensure we keep at least 30% of data
            if cp.sum(final_mask) < len(speeds) * 0.3:
                final_mask = valid_mask1
            
            speeds = speeds[final_mask]
            
            if len(speeds) < 5:
                return None
            
            return speeds.astype(cp.float32)
            
        except Exception as e:
            return None
    
    def enhanced_calculate_offset(self, video_motion: cp.ndarray, gps_speed: cp.ndarray, video_info: Dict) -> Tuple[Optional[float], float]:
        """Enhanced offset calculation with video-type-specific optimizations"""
        
        try:
            if len(video_motion) == 0 or len(gps_speed) == 0:
                return None, 0.0
            
            video_type = video_info.get('type', 'unknown')
            
            # Enhanced normalization
            video_std = cp.std(video_motion)
            gps_std = cp.std(gps_speed)
            
            if video_std < 1e-6 or gps_std < 1e-6:
                return None, 0.0
            
            video_norm = (video_motion - cp.mean(video_motion)) / video_std
            gps_norm = (gps_speed - cp.mean(gps_speed)) / gps_std
            
            # Apply video-type-specific preprocessing
            if video_type == '360':
                # 360 videos may have different motion characteristics
                # Apply slight smoothing to reduce noise from complex 360 motion
                if len(video_norm) > 5:
                    kernel = cp.array([0.1, 0.2, 0.4, 0.2, 0.1])
                    video_norm = cp.convolve(video_norm, kernel, mode='same')
            
            # Adaptive signal length management
            if video_type == '360':
                max_len = min(len(video_norm), len(gps_norm), 250)  # Longer for 360
            else:
                max_len = min(len(video_norm), len(gps_norm), 200)  # Standard for flat
            
            video_short = video_norm[:max_len]
            gps_short = gps_norm[:max_len]
            
            # Enhanced correlation with multiple methods
            best_offset = None
            best_confidence = 0.0
            
            # Method 1: FFT cross-correlation (for longer signals)
            if len(video_short) >= 15:
                try:
                    n = len(video_short) + len(gps_short) - 1
                    next_pow2 = 1 << (n - 1).bit_length()
                    
                    v_padded = cp.pad(video_short, (0, next_pow2 - len(video_short)))
                    g_padded = cp.pad(gps_short, (0, next_pow2 - len(gps_short)))
                    
                    v_fft = cp.fft.fft(v_padded)
                    g_fft = cp.fft.fft(g_padded)
                    correlation = cp.fft.ifft(cp.conj(v_fft) * g_fft).real
                    
                    correlation = correlation[:len(video_short) + len(gps_short) - 1]
                    best_idx = cp.argmax(cp.abs(correlation))
                    fft_confidence = float(cp.abs(correlation[best_idx]) / len(video_short))
                    
                    if fft_confidence > best_confidence:
                        best_confidence = fft_confidence
                        offset_samples = int(best_idx) - len(video_short) + 1
                        
                        # Adaptive time conversion based on video type
                        if video_type == '360':
                            time_per_sample = 1.5  # 360 videos sampled more frequently
                        else:
                            time_per_sample = 2.0  # Standard sampling
                        
                        best_offset = float(offset_samples * time_per_sample)
                        
                except Exception:
                    pass
            
            # Method 2: Direct correlation with enhanced search
            if best_confidence < 0.4:  # Try direct method if FFT didn't work well
                if video_type == '360':
                    max_offset_samples = min(40, max_len//3)  # Larger search for 360
                else:
                    max_offset_samples = min(30, max_len//3)
                
                for offset in range(-max_offset_samples, max_offset_samples + 1):
                    try:
                        if offset < 0:
                            v_seg = video_short[-offset:]
                            g_seg = gps_short[:len(v_seg)]
                        elif offset > 0:
                            g_seg = gps_short[offset:]
                            v_seg = video_short[:len(g_seg)]
                        else:
                            min_len = min(len(video_short), len(gps_short))
                            v_seg = video_short[:min_len]
                            g_seg = gps_short[:min_len]
                        
                        if len(v_seg) >= 5:
                            # Enhanced correlation coefficient
                            mean_v = cp.mean(v_seg)
                            mean_g = cp.mean(g_seg)
                            
                            num = cp.sum((v_seg - mean_v) * (g_seg - mean_g))
                            den = cp.sqrt(cp.sum((v_seg - mean_v)**2) * cp.sum((g_seg - mean_g)**2))
                            
                            if den > 1e-6:
                                corr = float(num / den)
                                
                                if abs(corr) > best_confidence:
                                    best_confidence = abs(corr)
                                    
                                    # Adaptive time conversion
                                    if video_type == '360':
                                        time_per_sample = 1.5
                                    else:
                                        time_per_sample = 2.0
                                    
                                    best_offset = float(offset * time_per_sample)
                                    
                    except Exception:
                        continue
            
            # Video-type-specific confidence adjustment
            if video_type == '360' and best_confidence > 0:
                # 360 videos are more complex, so slightly boost confidence for good matches
                best_confidence = min(1.0, best_confidence * 1.1)
            
            return best_offset, best_confidence
            
        except Exception as e:
            return None, 0.0

def main():
    """Enhanced main with 360° and flat video support"""
    
    parser = argparse.ArgumentParser(description='🎯 Enhanced Dual GPU - 360° + Flat Video Processor')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Workers per GPU (default: 2)')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--gpu-memory', type=float, default=15.0, help='GPU memory limit per GPU in GB')
    parser.add_argument('--enhanced-mode', action='store_true', help='Enable enhanced 360° processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.enhanced_mode:
        logger.info("🎯🔥🎯 ENHANCED 360° + FLAT VIDEO MODE ACTIVATED! 🎯🔥🎯")
        logger.info("🌟 INTELLIGENT VIDEO TYPE DETECTION AND PROCESSING 🌟")
    
    # Input validation
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"💀 Input file not found")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"enhanced_360_{input_file.name}"
    
    # GPU initialization
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"🎯 Detected {gpu_count} CUDA GPUs")
        
        for gpu_id in [0, 1]:
            if gpu_id < gpu_count:
                cp.cuda.Device(gpu_id).use()
                test = cp.array([1, 2, 3])
                del test
                logger.info(f"🎯 Enhanced GPU {gpu_id} ready")
            
    except Exception as e:
        logger.error(f"💀 GPU initialization failed: {e}")
        sys.exit(1)
    
    # Data loading and processing
    logger.info(f"📁 Loading data...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"💀 Failed to load data: {e}")
        sys.exit(1)
    
    # Collect matches
    all_matches = []
    total_potential = 0
    
    for video_path, video_data in data.get('results', {}).items():
        for match in video_data.get('matches', []):
            total_potential += 1
            if match.get('combined_score', 0) >= args.min_score:
                gpx_path = match.get('path', '')
                if Path(video_path).exists() and Path(gpx_path).exists():
                    all_matches.append((video_path, gpx_path, match))
                
                if args.limit and len(all_matches) >= args.limit:
                    break
        if args.limit and len(all_matches) >= args.limit:
            break
    
    logger.info(f"📊 Found {total_potential} total matches, {len(all_matches)} valid for enhanced processing")
    
    if len(all_matches) == 0:
        logger.error("💀 No valid matches found!")
        sys.exit(1)
    
    # Setup enhanced processing
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    for match in all_matches:
        work_queue.put(match)
    
    # Create enhanced workers
    workers = []
    worker_threads = []
    total_workers = 2 * args.workers_per_gpu
    
    logger.info(f"🎯 Starting {total_workers} enhanced workers...")
    
    for gpu_id in [0, 1]:
        for worker_id in range(args.workers_per_gpu):
            worker = Enhanced360Worker(gpu_id, work_queue, result_queue, args.gpu_memory)
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            workers.append(worker)
            worker_threads.append(thread)
    
    # Monitor progress
    results = []
    start_time = time.time()
    last_progress_time = start_time
    
    logger.info(f"🎯 Starting enhanced processing of {len(all_matches)} matches...")
    
    for i in range(len(all_matches)):
        try:
            result = result_queue.get(timeout=60)  # Longer timeout for enhanced processing
            results.append(result)
            
            current_time = time.time()
            
            if (i + 1) % 5 == 0 or current_time - last_progress_time > 15:
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(all_matches) - i - 1) / rate if rate > 0 else 0
                
                gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
                gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
                success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
                video_360_count = sum(1 for r in results if r.get('video_type') == '360')
                
                logger.info(f"🎯 Progress: {i+1}/{len(all_matches)} ({rate:.2f}/s) | "
                           f"GPU0: {gpu0_count}, GPU1: {gpu1_count} | "
                           f"Success: {success_count} | 360°: {video_360_count} | ETA: {eta/60:.1f}m")
                last_progress_time = current_time
                
        except queue.Empty:
            logger.error(f"💀 TIMEOUT at match {i+1}")
            break
        except Exception as e:
            logger.error(f"💀 Collection error: {e}")
            break
    
    processing_time = time.time() - start_time
    
    # Shutdown workers
    logger.info("🛑 Signaling workers to stop...")
    for _ in range(total_workers):
        work_queue.put(None)
    
    for thread in worker_threads:
        thread.join(timeout=10)
    
    # Create enhanced output
    logger.info("📊 Creating enhanced output...")
    enhanced_data = data.copy()
    
    # Process results
    result_map = {}
    for i, (video_path, gpx_path, _) in enumerate(all_matches):
        if i < len(results):
            result_map[(video_path, gpx_path)] = results[i]
    
    # Merge results
    enhanced_results = {}
    for video_path, video_data in data.get('results', {}).items():
        enhanced_video_data = video_data.copy()
        enhanced_matches = []
        
        for match in video_data.get('matches', []):
            gpx_path = match.get('path')
            key = (video_path, gpx_path)
            
            if key in result_map:
                enhanced_matches.append(result_map[key])
            else:
                enhanced_matches.append(match)
        
        enhanced_video_data['matches'] = enhanced_matches
        enhanced_results[video_path] = enhanced_video_data
    
    enhanced_data['results'] = enhanced_results
    
    # Add enhanced processing metadata
    video_360_count = sum(1 for r in results if r.get('video_type') == '360')
    video_flat_count = sum(1 for r in results if r.get('video_type') == 'flat')
    success_360 = sum(1 for r in results if r.get('video_type') == '360' and r.get('temporal_offset_seconds') is not None)
    success_flat = sum(1 for r in results if r.get('video_type') == 'flat' and r.get('temporal_offset_seconds') is not None)
    
    enhanced_data['enhanced_360_processing_info'] = {
        'enhanced_360_mode': args.enhanced_mode,
        'gpu_memory_gb': args.gpu_memory,
        'workers_per_gpu': args.workers_per_gpu,
        'total_workers': total_workers,
        'processing_time_seconds': processing_time,
        'matches_attempted': len(all_matches),
        'matches_completed': len(results),
        'processing_rate_matches_per_second': len(results) / processing_time if processing_time > 0 else 0,
        'total_success_rate': sum(1 for r in results if r.get('temporal_offset_seconds') is not None) / len(results) if results else 0,
        'video_type_detection': {
            'videos_360_detected': video_360_count,
            'videos_flat_detected': video_flat_count,
            'success_rate_360': success_360 / video_360_count if video_360_count > 0 else 0,
            'success_rate_flat': success_flat / video_flat_count if video_flat_count > 0 else 0
        },
        'processed_at': datetime.now().isoformat()
    }
    
    # Save results
    logger.info(f"💾 Saving enhanced results...")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"💀 Failed to save: {e}")
        sys.exit(1)
    
    # Final summary
    success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    
    logger.info("🎯🔥🎯 ENHANCED 360° + FLAT VIDEO PROCESSING COMPLETE! 🎯🔥🎯")
    logger.info("="*70)
    logger.info(f"📊 Total processed: {len(results)}")
    logger.info(f"✅ Successful offsets: {success_count}")
    logger.info(f"🌐 360° videos processed: {video_360_count} (success: {success_360})")
    logger.info(f"📺 Flat videos processed: {video_flat_count} (success: {success_flat})")
    logger.info(f"🔥 GPU 0 processed: {gpu0_count}")
    logger.info(f"🔥 GPU 1 processed: {gpu1_count}")
    logger.info(f"⚡ Processing time: {processing_time:.1f}s ({processing_time/60:.1f}m)")
    logger.info(f"🚀 Processing rate: {len(results)/processing_time:.2f} matches/second")
    logger.info(f"📈 Overall success rate: {success_count/len(results)*100:.1f}%" if results else "0%")
    if video_360_count > 0:
        logger.info(f"🌐 360° success rate: {success_360/video_360_count*100:.1f}%")
    if video_flat_count > 0:
        logger.info(f"📺 Flat success rate: {success_flat/video_flat_count*100:.1f}%")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("="*70)

if __name__ == "__main__":
    main()