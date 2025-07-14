#!/usr/bin/env python3
"""
🎯 TRUE TIME-DOMAIN PROCESSOR 🎯
🔥 PURE TIME-BASED CORRELATION - NO SAMPLE QUANTIZATION 🔥
🌟 WORKS ENTIRELY IN REAL TIME COORDINATES 🌟
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
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
import traceback
from scipy import signal, interpolate
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('true_temporal.log', mode='w')
        ]
    )
    return logging.getLogger('true_temporal')

class TrueTemporalWorker:
    """True time-domain correlation - pure time coordinates, no quantization"""
    
    def __init__(self, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, gpu_memory_gb: float = 15.0):
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.errors = 0
        self.logger = logging.getLogger(f'TrueTemporal_GPU{gpu_id}')
        self.is_running = True
        
        # True temporal parameters - work in actual seconds
        self.search_step_seconds = 0.05  # 50ms search resolution
        self.refinement_step_seconds = 0.01  # 10ms refinement resolution
        self.max_search_range_seconds = 90.0  # ±90 seconds
        self.min_overlap_seconds = 8.0  # Minimum 8s overlap
        
    def run(self):
        """True temporal worker loop"""
        
        self.logger.info(f"🎯 True Temporal GPU {self.gpu_id} STARTING...")
        
        try:
            # GPU initialization
            cp.cuda.Device(self.gpu_id).use()
            
            if self.gpu_memory_gb > 0:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.gpu_memory_gb * 1024**3))
            
            self.logger.info(f"🎯 True Temporal GPU {self.gpu_id} READY!")
            
            # Main processing loop
            while self.is_running:
                try:
                    work_item = self.work_queue.get(timeout=2)
                    
                    if work_item is None:
                        break
                    
                    video_path, gpx_path, match = work_item
                    
                    if not self.validate_files(video_path, gpx_path):
                        error_result = self.create_error_result(match, "file_invalid")
                        self.result_queue.put(error_result)
                        self.work_queue.task_done()
                        self.errors += 1
                        continue
                    
                    # True temporal processing
                    start_time = time.time()
                    result = self.true_temporal_process(video_path, gpx_path, match)
                    processing_time = time.time() - start_time
                    
                    self.result_queue.put(result)
                    self.processed += 1
                    
                    if self.processed % 5 == 0:
                        self.logger.info(f"🎯 True Temporal GPU {self.gpu_id}: {self.processed} done ({processing_time:.1f}s)")
                    
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"💀 True Temporal GPU {self.gpu_id}: {e}")
                    error_result = self.create_error_result(match, "processing_error")
                    self.result_queue.put(error_result)
                    self.work_queue.task_done()
                    self.errors += 1
        
        except Exception as e:
            self.logger.error(f"💀 True Temporal GPU {self.gpu_id} FATAL: {e}")
        
        finally:
            self.logger.info(f"🎯 True Temporal GPU {self.gpu_id} SHUTDOWN: {self.processed} processed, {self.errors} errors")
    
    def validate_files(self, video_path: str, gpx_path: str) -> bool:
        """File validation"""
        try:
            video_file = Path(video_path)
            gpx_file = Path(gpx_path)
            
            return (video_file.exists() and gpx_file.exists() and 
                   video_file.stat().st_size > 1024 and gpx_file.stat().st_size > 100)
        except:
            return False
    
    def create_error_result(self, match: Dict, error_type: str) -> Dict:
        """Create error result"""
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': False,
            'true_temporal_mode': True,
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': f'true_temporal_gpu_{self.gpu_id}_{error_type}',
            'sync_quality': 'failed'
        })
        return result
    
    def true_temporal_process(self, video_path: str, gpx_path: str, match: Dict) -> Dict:
        """True temporal processing - pure time domain"""
        
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': True,
            'true_temporal_mode': True,
            'offset_method': f'true_temporal_gpu_{self.gpu_id}'
        })
        
        try:
            cp.cuda.Device(self.gpu_id).use()
            
            # Extract video motion with pure time coordinates
            video_data = self.extract_video_time_series(video_path)
            if video_data is None:
                result['offset_method'] = f'true_temporal_gpu_{self.gpu_id}_video_failed'
                return result
            
            video_times_seconds, video_motion, video_info = video_data
            
            # Extract GPS speed with pure time coordinates  
            gps_data = self.extract_gps_time_series(gpx_path)
            if gps_data is None:
                result['offset_method'] = f'true_temporal_gpu_{self.gpu_id}_gps_failed'
                return result
            
            gps_times_seconds, gps_speed, gps_info = gps_data
            
            # Add extraction info
            result.update({
                'video_info': video_info,
                'gps_info': gps_info,
                'video_motion_points': len(video_motion),
                'gps_speed_points': len(gps_speed),
                'video_time_span_seconds': video_times_seconds[-1] - video_times_seconds[0] if len(video_times_seconds) > 1 else 0,
                'gps_time_span_seconds': gps_times_seconds[-1] - gps_times_seconds[0] if len(gps_times_seconds) > 1 else 0
            })
            
            # Pure time-domain correlation
            correlation_result = self.true_time_domain_correlation(
                video_times_seconds, video_motion, 
                gps_times_seconds, gps_speed,
                video_info, gps_info
            )
            
            offset_seconds = correlation_result.get('offset_seconds')
            confidence = correlation_result.get('confidence', 0.0)
            
            # Add detailed correlation info
            result.update({
                'correlation_analysis': correlation_result,
                'search_parameters': {
                    'search_step_seconds': self.search_step_seconds,
                    'refinement_step_seconds': self.refinement_step_seconds,
                    'max_search_range_seconds': self.max_search_range_seconds
                }
            })
            
            if offset_seconds is not None and confidence >= 0.25:
                result.update({
                    'temporal_offset_seconds': float(offset_seconds),
                    'offset_confidence': float(confidence),
                    'offset_method': f'true_temporal_gpu_{self.gpu_id}_success',
                    'sync_quality': 'excellent' if confidence >= 0.8 else 'good' if confidence >= 0.6 else 'fair'
                })
            else:
                result.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': float(confidence) if confidence else 0.0,
                    'offset_method': f'true_temporal_gpu_{self.gpu_id}_low_confidence',
                    'sync_quality': 'poor'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"💀 True Temporal GPU {self.gpu_id}: Processing error: {e}")
            result.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'true_temporal_gpu_{self.gpu_id}_exception',
                'gpu_processing': False,
                'error_details': str(e)
            })
            return result
    
    def extract_video_time_series(self, video_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract video motion as a true time series with real time coordinates"""
        
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or frame_count <= 0:
                return None
            
            duration_seconds = frame_count / fps
            aspect_ratio = width / height
            
            # Detect video type
            video_type = '360' if 1.8 <= aspect_ratio <= 2.2 else 'flat'
            
            video_info = {
                'type': video_type,
                'width': width,
                'height': height,
                'fps': fps,
                'duration_seconds': duration_seconds,
                'aspect_ratio': aspect_ratio
            }
            
            # Adaptive processing
            if video_type == '360':
                target_width = min(400, width // 4)
                # Sample more frequently for 360° videos
                target_sample_rate_fps = min(fps, 8.0)
            else:
                target_width = min(300, width // 4)
                # Standard sampling for flat videos
                target_sample_rate_fps = min(fps, 5.0)
            
            target_height = int(target_width * height / width)
            
            # Calculate frame interval for target sample rate
            frame_interval = max(1, int(fps / target_sample_rate_fps))
            actual_sample_rate_fps = fps / frame_interval
            
            self.logger.debug(f"Video {video_type}: {width}x{height}, {fps:.1f}fps, sampling at {actual_sample_rate_fps:.1f}fps")
            
            # Extract motion with REAL TIME COORDINATES
            motion_values = []
            time_coordinates_seconds = []  # CRITICAL: Pure time coordinates
            frame_idx = 0
            prev_gray = None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while frame_idx < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        # CRITICAL: Calculate EXACT time coordinate in seconds
                        time_seconds = frame_idx / fps  # Pure time - no quantization!
                        
                        # Process frame
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        
                        if prev_gray is not None:
                            motion = self.calculate_motion(prev_gray, gray, video_type)
                            
                            if motion is not None and not np.isnan(motion):
                                motion_values.append(float(motion))
                                time_coordinates_seconds.append(time_seconds)  # Pure time!
                        
                        prev_gray = gray.copy()
                        
                    except Exception as e:
                        self.logger.debug(f"Frame processing error at {frame_idx}: {e}")
                        pass
                
                frame_idx += 1
                
                # Reasonable limit for processing time
                if len(motion_values) >= 1000:
                    break
            
            if len(motion_values) < 5:
                return None
            
            video_info.update({
                'motion_points_extracted': len(motion_values),
                'actual_sample_rate_fps': actual_sample_rate_fps,
                'time_span_seconds': time_coordinates_seconds[-1] - time_coordinates_seconds[0] if len(time_coordinates_seconds) > 1 else 0
            })
            
            # Convert to numpy arrays - PURE TIME COORDINATES
            time_array = np.array(time_coordinates_seconds, dtype=np.float64)
            motion_array = np.array(motion_values, dtype=np.float32)
            
            return time_array, motion_array, video_info
            
        except Exception as e:
            self.logger.error(f"Video extraction error: {e}")
            return None
        finally:
            if cap:
                cap.release()
    
    def calculate_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray, video_type: str) -> float:
        """Calculate motion appropriate for video type"""
        try:
            if video_type == '360':
                # Focus on equatorial region for 360° videos
                h, w = prev_gray.shape
                eq_start = h // 5
                eq_end = 4 * h // 5
                
                prev_eq = prev_gray[eq_start:eq_end, :].astype(np.float32)
                curr_eq = curr_gray[eq_start:eq_end, :].astype(np.float32)
                
                diff = np.abs(curr_eq - prev_eq)
                
                # Spatial weighting for 360°
                eq_h, eq_w = diff.shape
                y_weights = np.exp(-0.5 * ((np.arange(eq_h) - eq_h/2) / (eq_h/4))**2)
                x_weights = np.ones(eq_w)
                
                weight_grid = np.outer(y_weights, x_weights)
                weighted_diff = diff * weight_grid
                
                motion = np.sum(weighted_diff) / np.sum(weight_grid)
                return float(motion)
            else:
                # Standard motion for flat videos
                diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
                return float(np.mean(diff))
                
        except Exception:
            return 0.0
    
    def extract_gps_time_series(self, gpx_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract GPS speed as a true time series with real time coordinates"""
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            # Collect GPS points with REAL TIMESTAMPS
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time and point.latitude and point.longitude:
                            # CRITICAL: Use actual timestamp
                            timestamp_seconds = point.time.timestamp()
                            points.append({
                                'timestamp_seconds': timestamp_seconds,  # Pure time!
                                'lat': point.latitude,
                                'lon': point.longitude
                            })
            
            if len(points) < 10:
                return None
            
            # Sort by actual time
            points.sort(key=lambda p: p['timestamp_seconds'])
            
            # Convert to arrays for vectorized processing
            timestamps_absolute = np.array([p['timestamp_seconds'] for p in points], dtype=np.float64)
            lats = np.array([p['lat'] for p in points], dtype=np.float64)
            lons = np.array([p['lon'] for p in points], dtype=np.float64)
            
            # Calculate distances using Haversine formula
            lat1_rad = np.radians(lats[:-1])
            lat2_rad = np.radians(lats[1:])
            lon1_rad = np.radians(lons[:-1])
            lon2_rad = np.radians(lons[1:])
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            distances_meters = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            # Calculate time differences - PURE TIME DIFFERENCES
            time_diffs_seconds = np.diff(timestamps_absolute)
            time_diffs_seconds = np.maximum(time_diffs_seconds, 0.1)  # Minimum 0.1 second
            
            # Calculate speeds
            speeds_mps = distances_meters / time_diffs_seconds
            
            # CRITICAL: Speed timestamps are REAL TIME COORDINATES
            speed_timestamps_absolute = timestamps_absolute[:-1] + time_diffs_seconds / 2  # Midpoint times
            
            # Remove outliers
            valid_mask = speeds_mps < 138.9  # 500 km/h in m/s
            
            # Statistical outlier removal
            if np.sum(valid_mask) > 10:
                valid_speeds = speeds_mps[valid_mask]
                q75, q25 = np.percentile(valid_speeds, [75, 25])
                iqr = q75 - q25
                threshold = q75 + 2.0 * iqr
                
                statistical_mask = speeds_mps < threshold
                final_mask = valid_mask & statistical_mask
                
                # Keep at least 40% of data
                if np.sum(final_mask) < len(speeds_mps) * 0.4:
                    final_mask = valid_mask
            else:
                final_mask = valid_mask
            
            clean_speeds = speeds_mps[final_mask]
            clean_timestamps_absolute = speed_timestamps_absolute[final_mask]
            
            if len(clean_speeds) < 5:
                return None
            
            # Convert to RELATIVE TIME COORDINATES (seconds from start of GPS track)
            gps_start_time = clean_timestamps_absolute[0]
            relative_time_coordinates_seconds = clean_timestamps_absolute - gps_start_time
            
            gps_info = {
                'total_points': len(points),
                'processed_points': len(clean_speeds),
                'time_span_seconds': relative_time_coordinates_seconds[-1] if len(relative_time_coordinates_seconds) > 1 else 0,
                'average_sample_rate_hz': len(clean_speeds) / (relative_time_coordinates_seconds[-1] + 0.1) if len(relative_time_coordinates_seconds) > 1 else 0,
                'speed_stats': {
                    'mean_mps': float(np.mean(clean_speeds)),
                    'std_mps': float(np.std(clean_speeds)),
                    'max_mps': float(np.max(clean_speeds)),
                    'min_mps': float(np.min(clean_speeds))
                },
                'gps_start_timestamp': gps_start_time
            }
            
            # Return PURE TIME COORDINATES and speeds
            return relative_time_coordinates_seconds, clean_speeds, gps_info
            
        except Exception as e:
            self.logger.error(f"GPS extraction error: {e}")
            return None
    
    def true_time_domain_correlation(self, video_times_seconds: np.ndarray, video_motion: np.ndarray,
                                   gps_times_seconds: np.ndarray, gps_speed: np.ndarray,
                                   video_info: Dict, gps_info: Dict) -> Dict:
        """True time-domain correlation - work purely in time coordinates"""
        
        result = {
            'offset_seconds': None,
            'confidence': 0.0,
            'method': 'true_time_domain',
            'analysis': {}
        }
        
        try:
            # Normalize signals
            video_norm = (video_motion - np.mean(video_motion)) / (np.std(video_motion) + 1e-10)
            gps_norm = (gps_speed - np.mean(gps_speed)) / (np.std(gps_speed) + 1e-10)
            
            # Time span analysis
            video_start_s, video_end_s = video_times_seconds[0], video_times_seconds[-1]
            gps_start_s, gps_end_s = gps_times_seconds[0], gps_times_seconds[-1]
            
            self.logger.debug(f"Video time span: {video_start_s:.2f} to {video_end_s:.2f}s ({video_end_s - video_start_s:.1f}s)")
            self.logger.debug(f"GPS time span: {gps_start_s:.2f} to {gps_end_s:.2f}s ({gps_end_s - gps_start_s:.1f}s)")
            
            # CRITICAL: Search in PURE TIME COORDINATES (seconds)
            # No sample indices, no quantization - pure time offsets!
            
            search_start_s = max(-self.max_search_range_seconds, gps_start_s - video_end_s)
            search_end_s = min(self.max_search_range_seconds, gps_end_s - video_start_s)
            
            if search_end_s <= search_start_s:
                result['analysis']['error'] = 'no_temporal_overlap_possible'
                return result
            
            # Pure time-domain search - NO QUANTIZATION TO SAMPLES!
            search_offsets_seconds = np.arange(
                search_start_s, 
                search_end_s + self.search_step_seconds, 
                self.search_step_seconds
            )
            
            self.logger.debug(f"Searching {len(search_offsets_seconds)} offsets from {search_start_s:.2f} to {search_end_s:.2f}s")
            
            best_offset_seconds = None
            best_correlation = 0.0
            correlation_scores = []
            
            for offset_s in search_offsets_seconds:
                try:
                    # Apply offset IN PURE TIME COORDINATES
                    shifted_gps_times_s = gps_times_seconds + offset_s
                    
                    # Find temporal overlap
                    overlap_start_s = max(video_times_seconds[0], shifted_gps_times_s[0])
                    overlap_end_s = min(video_times_seconds[-1], shifted_gps_times_s[-1])
                    
                    if overlap_end_s - overlap_start_s < self.min_overlap_seconds:
                        correlation_scores.append(0.0)
                        continue
                    
                    # Create common time grid for overlap region - PURE TIME
                    common_time_step_s = 0.2  # 200ms resolution for correlation
                    common_times_s = np.arange(overlap_start_s, overlap_end_s, common_time_step_s)
                    
                    if len(common_times_s) < 20:  # Need sufficient points
                        correlation_scores.append(0.0)
                        continue
                    
                    # Interpolate both signals to common time grid - PURE TIME INTERPOLATION
                    video_interp = self.pure_time_interpolate(video_times_seconds, video_norm, common_times_s)
                    gps_interp = self.pure_time_interpolate(shifted_gps_times_s, gps_norm, common_times_s)
                    
                    if video_interp is None or gps_interp is None:
                        correlation_scores.append(0.0)
                        continue
                    
                    # Calculate correlation
                    correlation = self.calculate_correlation(video_interp, gps_interp)
                    correlation_scores.append(correlation)
                    
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_offset_seconds = float(offset_s)  # PURE TIME OFFSET!
                        
                except Exception as e:
                    self.logger.debug(f"Correlation error at offset {offset_s:.3f}s: {e}")
                    correlation_scores.append(0.0)
                    continue
            
            result['analysis'] = {
                'search_range_seconds': [float(search_start_s), float(search_end_s)],
                'search_points': len(search_offsets_seconds),
                'max_correlation': abs(best_correlation),
                'correlation_distribution': {
                    'mean': float(np.mean(correlation_scores)),
                    'std': float(np.std(correlation_scores)),
                    'max_abs': float(np.max(np.abs(correlation_scores)))
                }
            }
            
            # Refinement in pure time domain
            if best_offset_seconds is not None and abs(best_correlation) >= 0.25:
                refined_offset_s, refined_correlation = self.pure_time_refinement(
                    video_times_seconds, video_norm, 
                    gps_times_seconds, gps_norm, 
                    best_offset_seconds
                )
                
                if refined_offset_s is not None:
                    result['offset_seconds'] = refined_offset_s  # PURE TIME RESULT!
                    result['confidence'] = abs(refined_correlation)
                    result['analysis']['refined'] = True
                    result['analysis']['refinement_improvement'] = abs(refined_correlation) - abs(best_correlation)
                else:
                    result['offset_seconds'] = best_offset_seconds
                    result['confidence'] = abs(best_correlation)
                    result['analysis']['refined'] = False
            
            return result
            
        except Exception as e:
            result['analysis']['error'] = f'correlation_failed: {str(e)}'
            return result
    
    def pure_time_interpolate(self, time_coords_s: np.ndarray, values: np.ndarray, target_times_s: np.ndarray) -> Optional[np.ndarray]:
        """Pure time-domain interpolation - no sample index conversions"""
        try:
            # Remove duplicates in time coordinates
            unique_indices = np.unique(time_coords_s, return_index=True)[1]
            clean_times_s = time_coords_s[unique_indices]
            clean_values = values[unique_indices]
            
            # Only interpolate within data range
            valid_mask = (target_times_s >= clean_times_s[0]) & (target_times_s <= clean_times_s[-1])
            
            if np.sum(valid_mask) < 10:
                return None
            
            # Pure time interpolation
            interpolated = np.interp(target_times_s[valid_mask], clean_times_s, clean_values)
            
            return interpolated if len(interpolated) >= 10 else None
            
        except Exception:
            return None
    
    def calculate_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate robust correlation"""
        try:
            if len(signal1) != len(signal2) or len(signal1) < 10:
                return 0.0
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(signal1) | np.isnan(signal2))
            if np.sum(valid_mask) < 10:
                return 0.0
            
            clean_signal1 = signal1[valid_mask]
            clean_signal2 = signal2[valid_mask]
            
            # Pearson correlation
            correlation, p_value = pearsonr(clean_signal1, clean_signal2)
            
            # Penalize high p-values (not significant)
            if p_value > 0.05:
                correlation *= 0.5
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def pure_time_refinement(self, video_times_s: np.ndarray, video_norm: np.ndarray,
                           gps_times_s: np.ndarray, gps_norm: np.ndarray,
                           initial_offset_s: float) -> Tuple[Optional[float], float]:
        """Refine offset with higher temporal resolution - pure time domain"""
        
        try:
            # Fine search around initial offset - PURE TIME
            fine_range_s = self.search_step_seconds * 2  # ±2 search steps
            fine_offsets_s = np.arange(
                initial_offset_s - fine_range_s,
                initial_offset_s + fine_range_s + self.refinement_step_seconds,
                self.refinement_step_seconds
            )
            
            best_fine_offset_s = initial_offset_s
            best_fine_correlation = 0.0
            
            for offset_s in fine_offsets_s:
                try:
                    shifted_gps_times_s = gps_times_s + offset_s
                    
                    # Find overlap
                    overlap_start_s = max(video_times_s[0], shifted_gps_times_s[0])
                    overlap_end_s = min(video_times_s[-1], shifted_gps_times_s[-1])
                    
                    if overlap_end_s - overlap_start_s < self.min_overlap_seconds:
                        continue
                    
                    # Finer common time grid for refinement
                    common_times_s = np.arange(overlap_start_s, overlap_end_s, 0.1)  # 100ms resolution
                    
                    if len(common_times_s) < 30:
                        continue
                    
                    # Interpolate
                    video_interp = self.pure_time_interpolate(video_times_s, video_norm, common_times_s)
                    gps_interp = self.pure_time_interpolate(shifted_gps_times_s, gps_norm, common_times_s)
                    
                    if video_interp is None or gps_interp is None:
                        continue
                    
                    # Calculate correlation
                    correlation = self.calculate_correlation(video_interp, gps_interp)
                    
                    if abs(correlation) > abs(best_fine_correlation):
                        best_fine_correlation = correlation
                        best_fine_offset_s = float(offset_s)  # PURE TIME RESULT!
                        
                except Exception:
                    continue
            
            return best_fine_offset_s, best_fine_correlation
            
        except Exception:
            return None, 0.0

def main():
    """True time-domain alignment main function"""
    
    parser = argparse.ArgumentParser(description='🎯 True Time-Domain Processor')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Workers per GPU')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--gpu-memory', type=float, default=15.0, help='GPU memory limit per GPU in GB')
    parser.add_argument('--search-step', type=float, default=0.05, help='Search step in seconds')
    parser.add_argument('--refinement-step', type=float, default=0.01, help='Refinement step in seconds')
    parser.add_argument('--search-range', type=float, default=90.0, help='Maximum search range in seconds')
    parser.add_argument('--true-mode', action='store_true', help='Enable true time-domain processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.true_mode:
        logger.info("🎯🔥🎯 TRUE TIME-DOMAIN MODE ACTIVATED! 🎯🔥🎯")
        logger.info("🌟 PURE TIME COORDINATES - NO QUANTIZATION 🌟")
    
    # Input validation
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"💀 Input file not found")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"true_temporal_{input_file.name}"
    
    # GPU initialization
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"🎯 Detected {gpu_count} CUDA GPUs")
        
        for gpu_id in [0, 1]:
            if gpu_id < gpu_count:
                cp.cuda.Device(gpu_id).use()
                test = cp.array([1, 2, 3])
                del test
                logger.info(f"🎯 True Temporal GPU {gpu_id} ready")
            
    except Exception as e:
        logger.error(f"💀 GPU initialization failed: {e}")
        sys.exit(1)
    
    # Data loading
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
    
    logger.info(f"📊 Found {total_potential} total matches, {len(all_matches)} valid for true temporal processing")
    
    if len(all_matches) == 0:
        logger.error("💀 No valid matches found!")
        sys.exit(1)
    
    # Setup true temporal processing
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    for match in all_matches:
        work_queue.put(match)
    
    # Create workers
    workers = []
    worker_threads = []
    total_workers = 2 * args.workers_per_gpu
    
    logger.info(f"🎯 Starting {total_workers} true temporal workers...")
    
    for gpu_id in [0, 1]:
        for worker_id in range(args.workers_per_gpu):
            worker = TrueTemporalWorker(gpu_id, work_queue, result_queue, args.gpu_memory)
            worker.search_step_seconds = args.search_step
            worker.refinement_step_seconds = args.refinement_step
            worker.max_search_range_seconds = args.search_range
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            workers.append(worker)
            worker_threads.append(thread)
    
    # Monitor progress
    results = []
    start_time = time.time()
    last_progress_time = start_time
    
    logger.info(f"🎯 Starting true temporal processing of {len(all_matches)} matches...")
    
    for i in range(len(all_matches)):
        try:
            result = result_queue.get(timeout=180)  # Extended timeout
            results.append(result)
            
            current_time = time.time()
            
            if (i + 1) % 3 == 0 or current_time - last_progress_time > 30:
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(all_matches) - i - 1) / rate if rate > 0 else 0
                
                gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
                gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
                success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
                
                # Show sample of found offsets to verify non-quantization
                successful_offsets = [r.get('temporal_offset_seconds') for r in results[-10:] if r.get('temporal_offset_seconds') is not None]
                if successful_offsets:
                    sample_offsets = [f"{x:.3f}" for x in successful_offsets[:3]]
                    offset_sample = ", ".join(sample_offsets) + "..."
                else:
                    offset_sample = "none yet"
                
                logger.info(f"🎯 Progress: {i+1}/{len(all_matches)} ({rate:.2f}/s) | "
                           f"GPU0: {gpu0_count}, GPU1: {gpu1_count} | "
                           f"Success: {success_count} | Recent offsets: {offset_sample} | ETA: {eta/60:.1f}m")
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
        thread.join(timeout=15)
    
    # Create output
    logger.info("📊 Creating true temporal output...")
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
    
    # Analyze results
    successful_offsets = [r.get('temporal_offset_seconds') for r in results if r.get('temporal_offset_seconds') is not None]
    
    # Precision analysis
    quantization_analysis = {}
    if successful_offsets:
        # Check for various quantization patterns
        patterns = {
            'integer_seconds': sum(1 for x in successful_offsets if abs(x - round(x)) < 0.01),
            'half_seconds': sum(1 for x in successful_offsets if abs(x - round(x * 2) / 2) < 0.01),
            'tenth_seconds': sum(1 for x in successful_offsets if abs(x - round(x * 10) / 10) < 0.01),
            'precise_values': sum(1 for x in successful_offsets if abs(x - round(x * 100) / 100) > 0.01)
        }
        
        quantization_analysis = {
            'total_successful': len(successful_offsets),
            'patterns': patterns,
            'precision_ratio': patterns['precise_values'] / len(successful_offsets),
            'unique_offsets': len(set(successful_offsets)),
            'uniqueness_ratio': len(set(successful_offsets)) / len(successful_offsets)
        }
    
    # Add comprehensive metadata
    enhanced_data['true_temporal_processing_info'] = {
        'true_temporal_mode': args.true_mode,
        'search_step_seconds': args.search_step,
        'refinement_step_seconds': args.refinement_step,
        'max_search_range_seconds': args.search_range,
        'gpu_memory_gb': args.gpu_memory,
        'workers_per_gpu': args.workers_per_gpu,
        'total_workers': total_workers,
        'processing_time_seconds': processing_time,
        'matches_attempted': len(all_matches),
        'matches_completed': len(results),
        'processing_rate_matches_per_second': len(results) / processing_time if processing_time > 0 else 0,
        'success_rate': len(successful_offsets) / len(results) if results else 0,
        'offset_statistics': {
            'successful_offsets': len(successful_offsets),
            'offset_range_seconds': [float(min(successful_offsets)), float(max(successful_offsets))] if successful_offsets else None,
            'offset_mean_seconds': float(np.mean(successful_offsets)) if successful_offsets else None,
            'offset_std_seconds': float(np.std(successful_offsets)) if successful_offsets else None,
        },
        'quantization_analysis': quantization_analysis,
        'processed_at': datetime.now().isoformat()
    }
    
    # Save results
    logger.info(f"💾 Saving true temporal results...")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"💀 Failed to save: {e}")
        sys.exit(1)
    
    # Final summary
    success_count = len(successful_offsets)
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    
    logger.info("🎯🔥🎯 TRUE TIME-DOMAIN PROCESSING COMPLETE! 🎯🔥🎯")
    logger.info("="*70)
    logger.info(f"📊 Total processed: {len(results)}")
    logger.info(f"✅ Successful offsets: {success_count}")
    logger.info(f"🔥 GPU 0 processed: {gpu0_count}")
    logger.info(f"🔥 GPU 1 processed: {gpu1_count}")
    logger.info(f"⚡ Processing time: {processing_time:.1f}s ({processing_time/60:.1f}m)")
    logger.info(f"🚀 Processing rate: {len(results)/processing_time:.2f} matches/second")
    logger.info(f"📈 Success rate: {success_count/len(results)*100:.1f}%" if results else "0%")
    
    if successful_offsets:
        logger.info(f"🎯 Offset range: [{min(successful_offsets):.3f}, {max(successful_offsets):.3f}] seconds")
        logger.info(f"📊 Offset statistics: mean={np.mean(successful_offsets):.3f}s, std={np.std(successful_offsets):.3f}s")
        logger.info(f"🌟 Unique offsets: {len(set(successful_offsets))} out of {len(successful_offsets)} (ratio: {len(set(successful_offsets))/len(successful_offsets):.2f})")
        
        # Quantization analysis
        if quantization_analysis:
            precision_pct = quantization_analysis['precision_ratio'] * 100
            logger.info(f"🎯 PRECISION ANALYSIS: {precision_pct:.1f}% of offsets are non-quantized!")
            
            if precision_pct >= 80:
                logger.info("🌟 EXCELLENT: True time-domain correlation is working!")
            elif precision_pct >= 60:
                logger.info("👍 GOOD: Mostly non-quantized results!")
            elif precision_pct >= 40:
                logger.info("📊 MODERATE: Some quantization still present")
            else:
                logger.info("⚠️ LOW: Still seeing significant quantization")
        
        # Show sample offsets
        sample_offsets = sorted(set(successful_offsets))[:10]
        logger.info(f"📋 Sample offsets: {[f'{x:.3f}' for x in sample_offsets]}")
    
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("="*70)

if __name__ == "__main__":
    main()