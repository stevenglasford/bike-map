#!/usr/bin/env python3
"""
🎯 IMPROVED DUAL GPU PROCESSOR 🎯
🔥 RELIABLE GPU UTILIZATION ON BOTH GPUS 🔥
🌟 PROPER VIDEO TYPE DETECTION AND PROCESSING 🌟
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
import random
import os

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dual_gpu_processor.log', mode='w')
        ]
    )
    return logging.getLogger('dual_gpu_processor')

class GPUManager:
    """Simplified GPU management without overly strict enforcement"""
    
    @staticmethod
    def initialize_gpu(gpu_id: int, memory_gb: float = 0) -> bool:
        """Initialize GPU with basic verification"""
        try:
            # Set device
            cp.cuda.Device(gpu_id).use()
            
            # Set memory limit if specified
            if memory_gb > 0:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(memory_gb * 1024**3))
            
            # Basic computation test
            test_array = cp.random.rand(100, 100)
            result = cp.sum(test_array)
            cp.cuda.Device(gpu_id).synchronize()
            
            # Verify result is reasonable
            assert 0 < result < 10000
            
            # Clean up
            del test_array
            
            return True
            
        except Exception as e:
            logging.error(f"GPU {gpu_id} initialization failed: {e}")
            return False
    
    @staticmethod
    def get_gpu_memory_usage(gpu_id: int) -> float:
        """Get current GPU memory usage in MB"""
        try:
            cp.cuda.Device(gpu_id).use()
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            return used_bytes / (1024 * 1024)
        except:
            return 0.0

class VideoAnalyzer:
    """Proper video type detection and analysis"""
    
    @staticmethod
    def detect_video_type(video_path: str) -> Dict:
        """Detect video type by analyzing actual properties"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'type': 'unknown', 'confidence': 0.0}
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            cap.release()
            
            if width <= 0 or height <= 0:
                return {'type': 'unknown', 'confidence': 0.0}
            
            aspect_ratio = width / height
            filename = Path(video_path).name.lower()
            
            # Detection logic
            confidence = 0.0
            video_type = 'flat'
            
            # Check aspect ratio (primary indicator)
            if 1.8 <= aspect_ratio <= 2.2:
                video_type = '360'
                confidence = 0.8
            elif 0.45 <= aspect_ratio <= 0.55:  # Sometimes 360 videos are stored as 1:2
                video_type = '360'
                confidence = 0.7
            
            # Check filename keywords (secondary indicator)
            keywords_360 = ['360', 'vr', 'spherical', 'equirect', 'panoramic', 'insta360', 'theta', 'ricoh']
            keywords_flat = ['flat', 'standard', 'normal', 'regular']
            
            if any(kw in filename for kw in keywords_360):
                if video_type != '360':
                    video_type = '360'
                    confidence = 0.6
                else:
                    confidence = min(0.95, confidence + 0.15)
            
            if any(kw in filename for kw in keywords_flat):
                if video_type != 'flat':
                    video_type = 'flat'
                    confidence = 0.6
                else:
                    confidence = min(0.95, confidence + 0.15)
            
            # Check resolution patterns
            if width >= 3840 and height >= 1920:  # Common 360 resolutions
                if video_type != '360':
                    video_type = '360'
                    confidence = 0.5
                else:
                    confidence = min(0.95, confidence + 0.1)
            
            return {
                'type': video_type,
                'confidence': confidence,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'fps': fps,
                'duration_seconds': frame_count / fps if fps > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Video analysis failed for {video_path}: {e}")
            return {'type': 'unknown', 'confidence': 0.0}

class DualGPUWorker:
    """Improved dual GPU worker with better load balancing"""
    
    def __init__(self, worker_id: str, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, 
                 gpu_memory_gb: float = 0):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.errors = 0
        self.logger = logging.getLogger(f'Worker_{worker_id}')
        self.is_running = True
        
        # Processing parameters
        self.search_step_seconds = 0.05
        self.refinement_step_seconds = 0.01
        self.max_search_range_seconds = 90.0
        self.min_overlap_seconds = 8.0
        
    def run(self):
        """Worker main loop"""
        
        self.logger.info(f"🔥 Worker {self.worker_id} starting on GPU {self.gpu_id}")
        
        # Initialize GPU
        if not GPUManager.initialize_gpu(self.gpu_id, self.gpu_memory_gb):
            self.logger.error(f"💀 Worker {self.worker_id}: Failed to initialize GPU {self.gpu_id}")
            return
        
        self.logger.info(f"✅ Worker {self.worker_id} GPU {self.gpu_id} initialized successfully")
        
        # Main processing loop
        while self.is_running:
            try:
                # Get work with timeout
                try:
                    work_item = self.work_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                if work_item is None:  # Shutdown signal
                    break
                
                video_path, gpx_path, match = work_item
                
                # Process the match
                start_time = time.time()
                result = self.process_match(video_path, gpx_path, match)
                processing_time = time.time() - start_time
                
                self.result_queue.put(result)
                self.processed += 1
                
                # Progress logging
                if self.processed % 5 == 0:
                    gpu_mem = GPUManager.get_gpu_memory_usage(self.gpu_id)
                    self.logger.info(f"🔥 Worker {self.worker_id}: {self.processed} processed "
                                   f"({processing_time:.1f}s, {gpu_mem:.1f}MB GPU)")
                
                self.work_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"💀 Worker {self.worker_id}: Unexpected error: {e}")
                self.errors += 1
                self.work_queue.task_done()
        
        self.logger.info(f"🔥 Worker {self.worker_id} finished: {self.processed} processed, {self.errors} errors")
    
    def process_match(self, video_path: str, gpx_path: str, match: Dict) -> Dict:
        """Process a single match"""
        
        result = match.copy()
        result.update({
            'worker_id': self.worker_id,
            'gpu_id': self.gpu_id,
            'dual_gpu_processing': True,
            'offset_method': f'dual_gpu_{self.worker_id}'
        })
        
        try:
            # Set GPU context
            cp.cuda.Device(self.gpu_id).use()
            
            # Validate files
            if not self.validate_files(video_path, gpx_path):
                result['offset_method'] = f'dual_gpu_{self.worker_id}_file_invalid'
                return result
            
            # Extract video motion
            video_data = self.extract_video_motion(video_path)
            if video_data is None:
                result['offset_method'] = f'dual_gpu_{self.worker_id}_video_failed'
                return result
            
            video_times, video_motion, video_info = video_data
            
            # Extract GPS data
            gps_data = self.extract_gps_speed(gpx_path)
            if gps_data is None:
                result['offset_method'] = f'dual_gpu_{self.worker_id}_gps_failed'
                return result
            
            gps_times, gps_speed, gps_info = gps_data
            
            # Add extraction info
            result.update({
                'video_info': video_info,
                'gps_info': gps_info,
                'video_motion_points': len(video_motion),
                'gps_speed_points': len(gps_speed)
            })
            
            # Perform correlation
            correlation_result = self.correlate_signals(
                video_times, video_motion, gps_times, gps_speed, video_info, gps_info
            )
            
            result.update(correlation_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"💀 Worker {self.worker_id}: Processing error: {e}")
            result.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'dual_gpu_{self.worker_id}_error',
                'error_details': str(e)
            })
            return result
    
    def validate_files(self, video_path: str, gpx_path: str) -> bool:
        """Validate input files exist and have reasonable sizes"""
        try:
            video_file = Path(video_path)
            gpx_file = Path(gpx_path)
            return (video_file.exists() and gpx_file.exists() and 
                   video_file.stat().st_size > 1024 and gpx_file.stat().st_size > 100)
        except:
            return False
    
    def extract_video_motion(self, video_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract video motion using GPU where beneficial"""
        
        cap = None
        try:
            # Set GPU context
            cp.cuda.Device(self.gpu_id).use()
            
            # Analyze video first
            video_analysis = VideoAnalyzer.detect_video_type(video_path)
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or frame_count <= 0:
                return None
            
            video_type = video_analysis.get('type', 'flat')
            
            # Adaptive processing based on video type
            if video_type == '360':
                target_width = min(400, width // 4)
                target_sample_rate = min(fps, 8.0)
            else:
                target_width = min(300, width // 4)
                target_sample_rate = min(fps, 5.0)
            
            target_height = int(target_width * height / width)
            frame_interval = max(1, int(fps / target_sample_rate))
            
            # Extract motion
            motion_values = []
            time_values = []
            frame_idx = 0
            prev_gray_gpu = None
            
            while frame_idx < frame_count and len(motion_values) < 1000:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        time_seconds = frame_idx / fps
                        
                        # Resize and convert to grayscale
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        gray_cpu = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        
                        # Transfer to GPU for processing
                        gray_gpu = cp.asarray(gray_cpu)
                        
                        if prev_gray_gpu is not None:
                            motion = self.calculate_motion_gpu(prev_gray_gpu, gray_gpu, video_type)
                            if motion is not None and not np.isnan(motion):
                                motion_values.append(float(motion))
                                time_values.append(time_seconds)
                        
                        prev_gray_gpu = gray_gpu
                        
                    except Exception as e:
                        self.logger.debug(f"Frame processing error: {e}")
                
                frame_idx += 1
            
            if len(motion_values) < 5:
                return None
            
            video_info = {
                'type': video_type,
                'width': width,
                'height': height,
                'fps': fps,
                'duration_seconds': frame_count / fps,
                'motion_points': len(motion_values),
                'confidence': video_analysis.get('confidence', 0.0)
            }
            
            return np.array(time_values), np.array(motion_values), video_info
            
        except Exception as e:
            self.logger.error(f"Video motion extraction error: {e}")
            return None
        finally:
            if cap:
                cap.release()
    
    def calculate_motion_gpu(self, prev_gray: cp.ndarray, curr_gray: cp.ndarray, video_type: str) -> float:
        """Calculate motion using GPU"""
        try:
            if video_type == '360':
                # For 360 videos, focus on equatorial region
                h, w = prev_gray.shape
                eq_start = h // 5
                eq_end = 4 * h // 5
                
                prev_eq = prev_gray[eq_start:eq_end, :].astype(cp.float32)
                curr_eq = curr_gray[eq_start:eq_end, :].astype(cp.float32)
                
                # Calculate weighted difference
                diff = cp.abs(curr_eq - prev_eq)
                
                # Apply spatial weighting
                eq_h, eq_w = diff.shape
                y_weights = cp.exp(-0.5 * ((cp.arange(eq_h) - eq_h/2) / (eq_h/4))**2)
                weight_grid = cp.outer(y_weights, cp.ones(eq_w))
                
                weighted_diff = diff * weight_grid
                motion = cp.sum(weighted_diff) / cp.sum(weight_grid)
                
                return float(cp.asnumpy(motion))
            else:
                # For flat videos, use global motion
                prev_f32 = prev_gray.astype(cp.float32)
                curr_f32 = curr_gray.astype(cp.float32)
                
                diff = cp.abs(curr_f32 - prev_f32)
                motion = cp.mean(diff)
                
                return float(cp.asnumpy(motion))
                
        except Exception as e:
            self.logger.debug(f"GPU motion calculation error: {e}")
            return 0.0
    
    def extract_gps_speed(self, gpx_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract GPS speed data"""
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time and point.latitude and point.longitude:
                            points.append({
                                'time': point.time.timestamp(),
                                'lat': point.latitude,
                                'lon': point.longitude
                            })
            
            if len(points) < 10:
                return None
            
            points.sort(key=lambda p: p['time'])
            
            # Calculate speeds
            times = np.array([p['time'] for p in points])
            lats = np.array([p['lat'] for p in points])
            lons = np.array([p['lon'] for p in points])
            
            # Haversine distance calculation
            lat1, lat2 = np.radians(lats[:-1]), np.radians(lats[1:])
            lon1, lon2 = np.radians(lons[:-1]), np.radians(lons[1:])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            distances = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            time_diffs = np.diff(times)
            time_diffs = np.maximum(time_diffs, 0.1)  # Minimum 0.1s
            
            speeds = distances / time_diffs
            speed_times = times[:-1] + time_diffs / 2
            
            # Filter outliers
            valid_mask = speeds < 140  # 140 m/s = ~500 km/h
            speeds = speeds[valid_mask]
            speed_times = speed_times[valid_mask]
            
            if len(speeds) < 5:
                return None
            
            # Convert to relative time
            start_time = speed_times[0]
            relative_times = speed_times - start_time
            
            gps_info = {
                'total_points': len(points),
                'speed_points': len(speeds),
                'time_span': relative_times[-1] if len(relative_times) > 1 else 0,
                'start_timestamp': start_time
            }
            
            return relative_times, speeds, gps_info
            
        except Exception as e:
            self.logger.error(f"GPS extraction error: {e}")
            return None
    
    def correlate_signals(self, video_times: np.ndarray, video_motion: np.ndarray,
                         gps_times: np.ndarray, gps_speeds: np.ndarray,
                         video_info: Dict, gps_info: Dict) -> Dict:
        """Correlate video motion with GPS speed"""
        
        result = {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': f'dual_gpu_{self.worker_id}_correlation',
            'sync_quality': 'poor'
        }
        
        try:
            # Normalize signals
            video_norm = (video_motion - np.mean(video_motion)) / (np.std(video_motion) + 1e-10)
            gps_norm = (gps_speeds - np.mean(gps_speeds)) / (np.std(gps_speeds) + 1e-10)
            
            # Determine search range
            video_span = video_times[-1] - video_times[0]
            gps_span = gps_times[-1] - gps_times[0]
            
            max_offset = min(self.max_search_range_seconds, max(video_span, gps_span))
            search_offsets = np.arange(-max_offset, max_offset + self.search_step_seconds, 
                                     self.search_step_seconds)
            
            best_offset = None
            best_correlation = 0.0
            
            # Search for best correlation across all offsets
            for offset in search_offsets:
                try:
                    shifted_gps_times = gps_times + offset
                    
                    # Find overlap
                    overlap_start = max(video_times[0], shifted_gps_times[0])
                    overlap_end = min(video_times[-1], shifted_gps_times[-1])
                    
                    if overlap_end - overlap_start < self.min_overlap_seconds:
                        continue
                    
                    # Create common time grid
                    common_times = np.arange(overlap_start, overlap_end, 0.2)
                    
                    if len(common_times) < 20:
                        continue
                    
                    # Interpolate both signals
                    video_interp = np.interp(common_times, video_times, video_norm)
                    gps_interp = np.interp(common_times, shifted_gps_times, gps_norm)
                    
                    # Calculate correlation
                    correlation, p_value = pearsonr(video_interp, gps_interp)
                    
                    if p_value > 0.05:  # Not significant
                        correlation *= 0.5
                    
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_offset = float(offset)
                
                except Exception:
                    continue
            
            # Refine if we found something decent
            if best_offset is not None and abs(best_correlation) >= 0.3:
                refined_offset, refined_correlation = self.refine_offset(
                    video_times, video_norm, gps_times, gps_norm, best_offset
                )
                
                if refined_offset is not None:
                    result['temporal_offset_seconds'] = refined_offset
                    result['offset_confidence'] = abs(refined_correlation)
                    result['offset_method'] = f'dual_gpu_{self.worker_id}_refined'
                    
                    # Determine quality
                    if abs(refined_correlation) >= 0.7:
                        result['sync_quality'] = 'excellent'
                    elif abs(refined_correlation) >= 0.5:
                        result['sync_quality'] = 'good'
                    else:
                        result['sync_quality'] = 'fair'
                else:
                    result['temporal_offset_seconds'] = best_offset
                    result['offset_confidence'] = abs(best_correlation)
                    result['offset_method'] = f'dual_gpu_{self.worker_id}_coarse'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Correlation error: {e}")
            result['error_details'] = str(e)
            return result
    
    def refine_offset(self, video_times: np.ndarray, video_norm: np.ndarray,
                     gps_times: np.ndarray, gps_norm: np.ndarray, 
                     initial_offset: float) -> Tuple[Optional[float], float]:
        """Refine offset with higher precision"""
        
        try:
            # Fine search around initial offset
            fine_range = self.search_step_seconds * 2
            fine_offsets = np.arange(
                initial_offset - fine_range,
                initial_offset + fine_range + self.refinement_step_seconds,
                self.refinement_step_seconds
            )
            
            best_offset = initial_offset
            best_correlation = 0.0
            
            for offset in fine_offsets:
                try:
                    shifted_gps_times = gps_times + offset
                    
                    overlap_start = max(video_times[0], shifted_gps_times[0])
                    overlap_end = min(video_times[-1], shifted_gps_times[-1])
                    
                    if overlap_end - overlap_start < self.min_overlap_seconds:
                        continue
                    
                    common_times = np.arange(overlap_start, overlap_end, 0.1)
                    
                    if len(common_times) < 30:
                        continue
                    
                    video_interp = np.interp(common_times, video_times, video_norm)
                    gps_interp = np.interp(common_times, shifted_gps_times, gps_norm)
                    
                    correlation, p_value = pearsonr(video_interp, gps_interp)
                    
                    if p_value > 0.05:
                        correlation *= 0.7
                    
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_offset = float(offset)
                
                except Exception:
                    continue
            
            return best_offset, best_correlation
            
        except Exception:
            return None, 0.0

def main():
    """Main function for dual GPU processing"""
    
    parser = argparse.ArgumentParser(description='🔥 Dual GPU Video Synchronization Processor')
    parser.add_argument('input_file', help='Input JSON file with matches')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Number of workers per GPU')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score threshold')
    parser.add_argument('--top-matches', type=int, default=10, help='Maximum matches to process per video')
    parser.add_argument('--limit', type=int, help='Limit number of matches for testing')
    parser.add_argument('--gpu-memory', type=float, default=0, help='GPU memory limit per GPU in GB (0 = no limit)')
    parser.add_argument('--search-step', type=float, default=0.05, help='Search step size in seconds')
    parser.add_argument('--refinement-step', type=float, default=0.01, help='Refinement step size in seconds')
    parser.add_argument('--search-range', type=float, default=90.0, help='Maximum search range in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🔥 DUAL GPU VIDEO SYNCHRONIZATION PROCESSOR 🔥")
    logger.info("=" * 60)
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"💀 Input file not found: {input_file}")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"dual_gpu_{input_file.name}"
    
    # Initialize GPUs
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"🔥 Detected {gpu_count} CUDA GPUs")
        
        if gpu_count < 2:
            logger.error(f"💀 Need at least 2 GPUs for dual GPU processing, found {gpu_count}")
            sys.exit(1)
        
        # Test both GPUs
        for gpu_id in [0, 1]:
            if GPUManager.initialize_gpu(gpu_id):
                logger.info(f"✅ GPU {gpu_id} initialized successfully")
            else:
                logger.error(f"💀 GPU {gpu_id} initialization failed")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"💀 GPU initialization failed: {e}")
        sys.exit(1)
    
    # Load input data
    logger.info(f"📁 Loading input data from {input_file}")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"💀 Failed to load input data: {e}")
        sys.exit(1)
    
    # Collect and analyze matches - ONLY TOP MATCHES PER VIDEO
    logger.info("📊 Analyzing matches...")
    all_matches = []
    video_type_stats = {'360': 0, 'flat': 0, 'unknown': 0}
    
    logger.info(f"🎯 Processing top {args.top_matches} matches per video")
    
    for video_path, video_data in data.get('results', {}).items():
        # Quick video type detection for stats
        if Path(video_path).exists():
            video_analysis = VideoAnalyzer.detect_video_type(video_path)
            video_type_stats[video_analysis.get('type', 'unknown')] += 1
        
        # Get ALL matches for this video and sort by score
        matches = video_data.get('matches', [])
        if not matches:
            continue
            
        # Sort matches by combined_score (highest first)
        sorted_matches = sorted(matches, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Take only the top N matches for this video
        top_matches = sorted_matches[:args.top_matches]
        
        # Apply score threshold to top matches
        for match in top_matches:
            if match.get('combined_score', 0) >= args.min_score:
                gpx_path = match.get('path', '')
                if Path(video_path).exists() and Path(gpx_path).exists():
                    all_matches.append((video_path, gpx_path, match))
                    
                    if args.limit and len(all_matches) >= args.limit:
                        break
            else:
                # If even the top matches don't meet threshold, skip this video
                break
        
        if args.limit and len(all_matches) >= args.limit:
            break
    
    logger.info(f"📊 Video type analysis:")
    logger.info(f"  🌐 360° videos: {video_type_stats['360']}")
    logger.info(f"  📺 Flat videos: {video_type_stats['flat']}")
    logger.info(f"  ❓ Unknown: {video_type_stats['unknown']}")
    logger.info(f"📊 Found {len(all_matches)} valid matches for processing")
    
    if len(all_matches) == 0:
        logger.error("💀 No valid matches found!")
        sys.exit(1)
    
    # Setup dual GPU processing
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Distribute work evenly
    random.shuffle(all_matches)
    for match in all_matches:
        work_queue.put(match)
    
    # Create workers
    workers = []
    worker_threads = []
    total_workers = 2 * args.workers_per_gpu
    
    logger.info(f"🔥 Starting {total_workers} dual GPU workers...")
    
    for gpu_id in [0, 1]:
        for worker_idx in range(args.workers_per_gpu):
            worker_id = f"GPU{gpu_id}_W{worker_idx}"
            worker = DualGPUWorker(worker_id, gpu_id, work_queue, result_queue, args.gpu_memory)
            worker.search_step_seconds = args.search_step
            worker.refinement_step_seconds = args.refinement_step
            worker.max_search_range_seconds = args.search_range
            
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            workers.append(worker)
            worker_threads.append(thread)
    
    # Wait for worker initialization
    logger.info("⏳ Waiting for worker initialization...")
    time.sleep(3)
    
    # Monitor progress
    results = []
    start_time = time.time()
    last_update = start_time
    
    logger.info(f"🚀 Processing {len(all_matches)} matches with dual GPU setup...")
    
    for i in range(len(all_matches)):
        try:
            result = result_queue.get(timeout=180)  # 3 minute timeout per match
            results.append(result)
            
            current_time = time.time()
            
            # Progress updates
            if (i + 1) % 5 == 0 or current_time - last_update > 30:
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                
                gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
                gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
                success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
                
                logger.info(f"📊 Progress: {i+1}/{len(all_matches)} ({rate:.2f}/s) | "
                           f"GPU0: {gpu0_count}, GPU1: {gpu1_count} | "
                           f"Success: {success_count}")
                last_update = current_time
                
        except queue.Empty:
            logger.error(f"💀 Timeout waiting for result {i+1}")
            break
        except Exception as e:
            logger.error(f"💀 Error collecting result {i+1}: {e}")
            break
    
    processing_time = time.time() - start_time
    
    # Shutdown workers
    logger.info("🛑 Shutting down workers...")
    for _ in range(total_workers):
        work_queue.put(None)
    
    for thread in worker_threads:
        thread.join(timeout=10)
    
    # Create output
    logger.info("📊 Creating output...")
    enhanced_data = data.copy()
    
    # Map results back to original structure
    result_map = {}
    for i, (video_path, gpx_path, _) in enumerate(all_matches):
        if i < len(results):
            result_map[(video_path, gpx_path)] = results[i]
    
    # Merge results into original data structure
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
    
    # Calculate statistics
    successful_results = [r for r in results if r.get('temporal_offset_seconds') is not None]
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    
    # Add processing metadata
    enhanced_data['dual_gpu_processing_info'] = {
        'dual_gpu_mode': True,
        'processing_time_seconds': processing_time,
        'matches_processed': len(results),
        'matches_successful': len(successful_results),
        'success_rate': len(successful_results) / len(results) if results else 0,
        'processing_rate': len(results) / processing_time if processing_time > 0 else 0,
        'gpu_distribution': {
            'gpu0_matches': gpu0_count,
            'gpu1_matches': gpu1_count,
            'load_balance': min(gpu0_count, gpu1_count) / max(gpu0_count, gpu1_count, 1)
        },
        'video_type_distribution': video_type_stats,
        'parameters': {
            'search_step_seconds': args.search_step,
            'refinement_step_seconds': args.refinement_step,
            'max_search_range_seconds': args.search_range,
            'min_score_threshold': args.min_score,
            'workers_per_gpu': args.workers_per_gpu
        },
        'processed_at': datetime.now().isoformat()
    }
    
    # Save results
    logger.info(f"💾 Saving results to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"💀 Failed to save results: {e}")
        sys.exit(1)
    
    # Final statistics
    load_balance = min(gpu0_count, gpu1_count) / max(gpu0_count, gpu1_count, 1) * 100 if results else 0
    
    logger.info("🎉 DUAL GPU PROCESSING COMPLETE! 🎉")
    logger.info("=" * 60)
    logger.info(f"📊 Total processed: {len(results)}")
    logger.info(f"✅ Successful synchronizations: {len(successful_results)}")
    logger.info(f"📈 Success rate: {len(successful_results)/len(results)*100:.1f}%" if results else "0%")
    logger.info(f"🔥 GPU 0 processed: {gpu0_count}")
    logger.info(f"🔥 GPU 1 processed: {gpu1_count}")
    logger.info(f"⚖️ Load balance: {load_balance:.1f}%")
    logger.info(f"⚡ Processing time: {processing_time:.1f}s ({processing_time/60:.1f}m)")
    logger.info(f"🚀 Processing rate: {len(results)/processing_time:.2f} matches/second")
    
    if successful_results:
        offsets = [r['temporal_offset_seconds'] for r in successful_results]
        logger.info(f"🎯 Offset range: [{min(offsets):.3f}, {max(offsets):.3f}] seconds")
        logger.info(f"📊 Unique offsets: {len(set(offsets))}")
    
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()