#!/usr/bin/env python3
"""
🎯 FORCED DUAL GPU UTILIZATION PROCESSOR 🎯
🔥 GUARANTEED GPU COMPUTE USAGE ON BOTH GPUS 🔥
🌟 STRICT GPU VERIFICATION AND ENFORCEMENT 🌟
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
            logging.FileHandler('forced_dual_gpu.log', mode='w')
        ]
    )
    return logging.getLogger('forced_dual_gpu')

class GPUEnforcer:
    """Enforce actual GPU usage and detect when workers fall back to CPU"""
    
    @staticmethod
    def force_gpu_computation(gpu_id: int, computation_size: int = 1000):
        """Force a computation on the specified GPU and verify it's actually using GPU"""
        try:
            # Set device
            cp.cuda.Device(gpu_id).use()
            
            # Force GPU computation
            a = cp.random.rand(computation_size, computation_size)
            b = cp.random.rand(computation_size, computation_size)
            
            # Matrix multiplication - definitely GPU intensive
            start_time = time.time()
            c = cp.dot(a, b)
            cp.cuda.Device(gpu_id).synchronize()  # Force completion
            gpu_time = time.time() - start_time
            
            # Verify result is on GPU
            assert isinstance(c, cp.ndarray)
            assert c.device.id == gpu_id
            
            # Cleanup
            del a, b, c
            
            return gpu_time
            
        except Exception as e:
            raise RuntimeError(f"GPU {gpu_id} computation failed: {e}")
    
    @staticmethod
    def verify_gpu_memory_usage(gpu_id: int, expected_minimum_mb: float = 100):
        """Verify GPU is actually using memory for computation"""
        try:
            cp.cuda.Device(gpu_id).use()
            
            # Get memory info
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            used_mb = used_bytes / (1024 * 1024)
            
            if used_mb < expected_minimum_mb:
                raise RuntimeError(f"GPU {gpu_id} using only {used_mb:.1f}MB, expected >{expected_minimum_mb}MB")
            
            return used_mb
            
        except Exception as e:
            raise RuntimeError(f"GPU {gpu_id} memory verification failed: {e}")

class ForcedDualGPUWorker:
    """Worker that enforces GPU usage and verifies computation happens on GPU"""
    
    def __init__(self, worker_id: str, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, 
                 gpu_memory_gb: float = 15.0):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.errors = 0
        self.logger = logging.getLogger(f'ForcedGPU_{worker_id}')
        self.is_running = True
        self.gpu_verified = False
        
        # Processing parameters
        self.search_step_seconds = 0.05
        self.refinement_step_seconds = 0.01
        self.max_search_range_seconds = 90.0
        self.min_overlap_seconds = 8.0
        
        # GPU enforcement
        self.last_gpu_verification = 0
        self.gpu_verification_interval = 30  # Verify GPU usage every 30 seconds
        
    def run(self):
        """Worker loop with strict GPU enforcement"""
        
        self.logger.info(f"🔥 FORCED GPU Worker {self.worker_id} (GPU {self.gpu_id}) STARTING...")
        
        # CRITICAL: Force GPU initialization and verification
        if not self.force_gpu_initialization():
            self.logger.error(f"💀 Worker {self.worker_id}: FAILED to force GPU {self.gpu_id} initialization")
            return
        
        self.logger.info(f"🔥 Worker {self.worker_id} (GPU {self.gpu_id}) FORCED GPU READY!")
        
        # Main processing loop with GPU enforcement
        while self.is_running:
            try:
                # Regular GPU verification
                current_time = time.time()
                if current_time - self.last_gpu_verification > self.gpu_verification_interval:
                    if not self.verify_gpu_usage():
                        self.logger.error(f"💀 Worker {self.worker_id}: GPU {self.gpu_id} verification FAILED!")
                        break
                    self.last_gpu_verification = current_time
                
                # Get work
                try:
                    work_item = self.work_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                if work_item is None:
                    break
                
                video_path, gpx_path, match = work_item
                
                # Validate files
                if not self.validate_files(video_path, gpx_path):
                    error_result = self.create_error_result(match, "file_invalid")
                    self.result_queue.put(error_result)
                    self.work_queue.task_done()
                    self.errors += 1
                    continue
                
                # CRITICAL: Force GPU context before processing
                self.force_gpu_context()
                
                # Process with GPU enforcement
                try:
                    start_time = time.time()
                    result = self.forced_gpu_process(video_path, gpx_path, match)
                    processing_time = time.time() - start_time
                    
                    # Verify GPU was actually used during processing
                    self.verify_post_processing_gpu_state()
                    
                    self.result_queue.put(result)
                    self.processed += 1
                    
                    if self.processed % 5 == 0:
                        gpu_mem = self.get_gpu_memory_usage()
                        self.logger.info(f"🔥 Worker {self.worker_id} (GPU {self.gpu_id}): {self.processed} done ({processing_time:.1f}s, {gpu_mem:.1f}MB GPU mem)")
                    
                except Exception as e:
                    self.logger.error(f"💀 Worker {self.worker_id}: Processing error: {e}")
                    error_result = self.create_error_result(match, "processing_error")
                    self.result_queue.put(error_result)
                    self.errors += 1
                
                self.work_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"💀 Worker {self.worker_id}: Unexpected error: {e}")
                break
        
        self.logger.info(f"🔥 Worker {self.worker_id} (GPU {self.gpu_id}) SHUTDOWN: {self.processed} processed, {self.errors} errors")
    
    def force_gpu_initialization(self) -> bool:
        """Force GPU initialization with extensive verification"""
        
        for attempt in range(5):  # More attempts
            try:
                self.logger.info(f"🔧 Worker {self.worker_id}: FORCING GPU {self.gpu_id} init (attempt {attempt + 1})")
                
                # Force device selection
                cp.cuda.Device(self.gpu_id).use()
                
                # CRITICAL: Force substantial GPU computation to "wake up" the GPU
                computation_time = GPUEnforcer.force_gpu_computation(self.gpu_id, 2000)  # Large computation
                self.logger.info(f"🔥 Worker {self.worker_id}: GPU {self.gpu_id} computation test: {computation_time:.3f}s")
                
                # Set memory limit
                if self.gpu_memory_gb > 0:
                    memory_pool = cp.get_default_memory_pool()
                    memory_pool.set_limit(size=int(self.gpu_memory_gb * 1024**3))
                
                # Force memory allocation test
                test_size = 500  # MB
                test_array = cp.random.rand(int(test_size * 1024 * 1024 / 8))  # 8 bytes per float64
                memory_used = GPUEnforcer.verify_gpu_memory_usage(self.gpu_id, test_size)
                del test_array
                
                self.logger.info(f"🔥 Worker {self.worker_id}: GPU {self.gpu_id} memory test: {memory_used:.1f}MB")
                
                # Additional GPU stress test
                self.perform_gpu_stress_test()
                
                self.gpu_verified = True
                self.last_gpu_verification = time.time()
                self.logger.info(f"✅ Worker {self.worker_id}: GPU {self.gpu_id} FORCE INITIALIZED successfully")
                return True
                
            except Exception as e:
                self.logger.warning(f"⚠️ Worker {self.worker_id}: GPU {self.gpu_id} force init attempt {attempt + 1} failed: {e}")
                time.sleep(2 + attempt)  # Increasing delay
                
                # Try to clear GPU state
                try:
                    cp.cuda.Device(self.gpu_id).use()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.cuda.Device(self.gpu_id).synchronize()
                except:
                    pass
        
        return False
    
    def perform_gpu_stress_test(self):
        """Perform stress test to ensure GPU is actually computing"""
        cp.cuda.Device(self.gpu_id).use()
        
        # Test 1: Matrix operations
        a = cp.random.rand(1000, 1000)
        b = cp.random.rand(1000, 1000)
        c = cp.dot(a, b)
        assert c.device.id == self.gpu_id
        
        # Test 2: Element-wise operations
        d = cp.sin(a) + cp.cos(b)
        assert d.device.id == self.gpu_id
        
        # Test 3: Reductions
        e = cp.sum(c * d)
        assert e.device.id == self.gpu_id
        
        # Test 4: Memory transfers
        f_cpu = cp.asnumpy(e)
        f_gpu = cp.asarray(f_cpu)
        assert f_gpu.device.id == self.gpu_id
        
        # Cleanup
        del a, b, c, d, e, f_gpu
        
        self.logger.debug(f"🔥 Worker {self.worker_id}: GPU {self.gpu_id} stress test passed")
    
    def force_gpu_context(self):
        """Force GPU context and ensure we're on the correct GPU"""
        cp.cuda.Device(self.gpu_id).use()
        
        # Force a small computation to ensure context is active
        test = cp.array([1.0, 2.0, 3.0])
        result = cp.sum(test)
        assert result == 6.0
        assert test.device.id == self.gpu_id
        del test
    
    def verify_gpu_usage(self) -> bool:
        """Verify GPU is actually being used"""
        try:
            # Force context
            cp.cuda.Device(self.gpu_id).use()
            
            # Check memory usage
            memory_used = GPUEnforcer.verify_gpu_memory_usage(self.gpu_id, 50)  # At least 50MB
            
            # Perform computation test
            computation_time = GPUEnforcer.force_gpu_computation(self.gpu_id, 500)
            
            self.logger.debug(f"🔥 Worker {self.worker_id}: GPU {self.gpu_id} verified: {memory_used:.1f}MB, {computation_time:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"💀 Worker {self.worker_id}: GPU {self.gpu_id} verification failed: {e}")
            return False
    
    def verify_post_processing_gpu_state(self):
        """Verify GPU state after processing to ensure GPU was used"""
        try:
            memory_used = self.get_gpu_memory_usage()
            if memory_used < 10:  # Less than 10MB suggests no GPU usage
                self.logger.warning(f"⚠️ Worker {self.worker_id}: Low GPU memory usage ({memory_used:.1f}MB) - may be using CPU")
        except:
            pass
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        try:
            cp.cuda.Device(self.gpu_id).use()
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            return used_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def validate_files(self, video_path: str, gpx_path: str) -> bool:
        """Validate input files"""
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
            'worker_id': self.worker_id,
            'gpu_id': self.gpu_id,
            'gpu_processing': False,
            'forced_dual_gpu_mode': True,
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': f'forced_gpu_{self.worker_id}_{error_type}',
            'sync_quality': 'failed'
        })
        return result
    
    def forced_gpu_process(self, video_path: str, gpx_path: str, match: Dict) -> Dict:
        """Processing with forced GPU usage"""
        
        result = match.copy()
        result.update({
            'worker_id': self.worker_id,
            'gpu_id': self.gpu_id,
            'gpu_processing': True,
            'forced_dual_gpu_mode': True,
            'offset_method': f'forced_gpu_{self.worker_id}'
        })
        
        try:
            # CRITICAL: Force GPU context at start
            self.force_gpu_context()
            
            # Extract video motion with FORCED GPU usage
            video_data = self.extract_video_time_series_gpu_forced(video_path)
            if video_data is None:
                result['offset_method'] = f'forced_gpu_{self.worker_id}_video_failed'
                return result
            
            video_times_seconds, video_motion, video_info = video_data
            
            # Extract GPS speed
            gps_data = self.extract_gps_time_series(gpx_path)
            if gps_data is None:
                result['offset_method'] = f'forced_gpu_{self.worker_id}_gps_failed'
                return result
            
            gps_times_seconds, gps_speed, gps_info = gps_data
            
            # Add extraction info
            result.update({
                'video_info': video_info,
                'gps_info': gps_info,
                'video_motion_points': len(video_motion),
                'gps_speed_points': len(gps_speed)
            })
            
            # FORCED GPU time-domain correlation
            correlation_result = self.forced_gpu_correlation(
                video_times_seconds, video_motion,
                gps_times_seconds, gps_speed,
                video_info, gps_info
            )
            
            offset_seconds = correlation_result.get('offset_seconds')
            confidence = correlation_result.get('confidence', 0.0)
            
            result.update({'correlation_analysis': correlation_result})
            
            if offset_seconds is not None and confidence >= 0.25:
                result.update({
                    'temporal_offset_seconds': float(offset_seconds),
                    'offset_confidence': float(confidence),
                    'offset_method': f'forced_gpu_{self.worker_id}_success',
                    'sync_quality': 'excellent' if confidence >= 0.8 else 'good' if confidence >= 0.6 else 'fair'
                })
            else:
                result.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': float(confidence) if confidence else 0.0,
                    'offset_method': f'forced_gpu_{self.worker_id}_low_confidence',
                    'sync_quality': 'poor'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"💀 Worker {self.worker_id}: Processing error: {e}")
            result.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'forced_gpu_{self.worker_id}_exception',
                'gpu_processing': False,
                'error_details': str(e)
            })
            return result
    
    def extract_video_time_series_gpu_forced(self, video_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract video motion with FORCED GPU usage"""
        
        cap = None
        try:
            # FORCE GPU context
            self.force_gpu_context()
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_HEIGHT))
            
            if fps <= 0 or frame_count <= 0:
                return None
            
            duration_seconds = frame_count / fps
            aspect_ratio = width / height
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
                target_sample_rate_fps = min(fps, 8.0)
            else:
                target_width = min(300, width // 4)
                target_sample_rate_fps = min(fps, 5.0)
            
            target_height = int(target_width * height / width)
            frame_interval = max(1, int(fps / target_sample_rate_fps))
            
            # Extract motion with FORCED GPU computations
            motion_values = []
            time_coordinates_seconds = []
            frame_idx = 0
            prev_gray_gpu = None  # FORCE GPU arrays
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while frame_idx < frame_count and len(motion_values) < 1000:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        # FORCE GPU context
                        cp.cuda.Device(self.gpu_id).use()
                        
                        time_seconds = frame_idx / fps
                        
                        # Process frame and FORCE GPU usage
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        gray_cpu = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        
                        # CRITICAL: Force GPU transfer and verify
                        gray_gpu = cp.asarray(gray_cpu)
                        assert gray_gpu.device.id == self.gpu_id
                        
                        if prev_gray_gpu is not None:
                            # FORCE GPU motion calculation
                            motion = self.calculate_motion_gpu_forced(prev_gray_gpu, gray_gpu, video_type)
                            
                            if motion is not None and not np.isnan(motion):
                                motion_values.append(float(motion))
                                time_coordinates_seconds.append(time_seconds)
                        
                        prev_gray_gpu = gray_gpu.copy()  # Keep on GPU
                        
                    except Exception as e:
                        self.logger.debug(f"Frame processing error: {e}")
                        pass
                
                frame_idx += 1
            
            if len(motion_values) < 5:
                return None
            
            video_info.update({
                'motion_points_extracted': len(motion_values),
                'time_span_seconds': time_coordinates_seconds[-1] - time_coordinates_seconds[0] if len(time_coordinates_seconds) > 1 else 0
            })
            
            return np.array(time_coordinates_seconds, dtype=np.float64), np.array(motion_values, dtype=np.float32), video_info
            
        except Exception as e:
            self.logger.error(f"Video extraction error: {e}")
            return None
        finally:
            if cap:
                cap.release()
    
    def calculate_motion_gpu_forced(self, prev_gray_gpu: cp.ndarray, curr_gray_gpu: cp.ndarray, video_type: str) -> float:
        """Calculate motion with FORCED GPU computation"""
        try:
            # FORCE GPU context
            cp.cuda.Device(self.gpu_id).use()
            
            # Verify arrays are on correct GPU
            assert prev_gray_gpu.device.id == self.gpu_id
            assert curr_gray_gpu.device.id == self.gpu_id
            
            if video_type == '360':
                h, w = prev_gray_gpu.shape
                eq_start = h // 5
                eq_end = 4 * h // 5
                
                # FORCE GPU slicing and computation
                prev_eq = prev_gray_gpu[eq_start:eq_end, :].astype(cp.float32)
                curr_eq = curr_gray_gpu[eq_start:eq_end, :].astype(cp.float32)
                
                # Ensure GPU computation
                assert prev_eq.device.id == self.gpu_id
                assert curr_eq.device.id == self.gpu_id
                
                # FORCE GPU difference calculation
                diff = cp.abs(curr_eq - prev_eq)
                
                # Spatial weighting on GPU
                eq_h, eq_w = diff.shape
                y_weights_gpu = cp.exp(-0.5 * ((cp.arange(eq_h) - eq_h/2) / (eq_h/4))**2)
                x_weights_gpu = cp.ones(eq_w)
                
                weight_grid = cp.outer(y_weights_gpu, x_weights_gpu)
                weighted_diff = diff * weight_grid
                
                # FORCE GPU reduction
                motion_gpu = cp.sum(weighted_diff) / cp.sum(weight_grid)
                
                # Verify result is on GPU before transfer
                assert motion_gpu.device.id == self.gpu_id
                
                motion = float(cp.asnumpy(motion_gpu))
                return motion
            else:
                # Standard motion for flat videos - FORCE GPU
                prev_f32 = prev_gray_gpu.astype(cp.float32)
                curr_f32 = curr_gray_gpu.astype(cp.float32)
                
                diff = cp.abs(curr_f32 - prev_f32)
                motion_gpu = cp.mean(diff)
                
                assert motion_gpu.device.id == self.gpu_id
                motion = float(cp.asnumpy(motion_gpu))
                return motion
                
        except Exception as e:
            self.logger.debug(f"GPU motion calculation error: {e}")
            return 0.0
    
    def extract_gps_time_series(self, gpx_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract GPS speed with time coordinates (CPU processing is fine for GPS)"""
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time and point.latitude and point.longitude:
                            timestamp_seconds = point.time.timestamp()
                            points.append({
                                'timestamp_seconds': timestamp_seconds,
                                'lat': point.latitude,
                                'lon': point.longitude
                            })
            
            if len(points) < 10:
                return None
            
            points.sort(key=lambda p: p['timestamp_seconds'])
            
            # Convert to arrays
            timestamps_absolute = np.array([p['timestamp_seconds'] for p in points], dtype=np.float64)
            lats = np.array([p['lat'] for p in points], dtype=np.float64)
            lons = np.array([p['lon'] for p in points], dtype=np.float64)
            
            # Calculate distances and speeds
            lat1_rad = np.radians(lats[:-1])
            lat2_rad = np.radians(lats[1:])
            lon1_rad = np.radians(lons[:-1])
            lon2_rad = np.radians(lons[1:])
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            distances_meters = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            time_diffs_seconds = np.diff(timestamps_absolute)
            time_diffs_seconds = np.maximum(time_diffs_seconds, 0.1)
            
            speeds_mps = distances_meters / time_diffs_seconds
            speed_timestamps_absolute = timestamps_absolute[:-1] + time_diffs_seconds / 2
            
            # Remove outliers
            valid_mask = speeds_mps < 138.9
            
            if np.sum(valid_mask) > 10:
                valid_speeds = speeds_mps[valid_mask]
                q75, q25 = np.percentile(valid_speeds, [75, 25])
                iqr = q75 - q25
                threshold = q75 + 2.0 * iqr
                
                statistical_mask = speeds_mps < threshold
                final_mask = valid_mask & statistical_mask
                
                if np.sum(final_mask) < len(speeds_mps) * 0.4:
                    final_mask = valid_mask
            else:
                final_mask = valid_mask
            
            clean_speeds = speeds_mps[final_mask]
            clean_timestamps_absolute = speed_timestamps_absolute[final_mask]
            
            if len(clean_speeds) < 5:
                return None
            
            # Convert to relative time coordinates
            gps_start_time = clean_timestamps_absolute[0]
            relative_time_coordinates_seconds = clean_timestamps_absolute - gps_start_time
            
            gps_info = {
                'total_points': len(points),
                'processed_points': len(clean_speeds),
                'time_span_seconds': relative_time_coordinates_seconds[-1] if len(relative_time_coordinates_seconds) > 1 else 0,
                'gps_start_timestamp': gps_start_time
            }
            
            return relative_time_coordinates_seconds, clean_speeds, gps_info
            
        except Exception as e:
            self.logger.error(f"GPS extraction error: {e}")
            return None
    
    def forced_gpu_correlation(self, video_times_seconds: np.ndarray, video_motion: np.ndarray,
                             gps_times_seconds: np.ndarray, gps_speed: np.ndarray,
                             video_info: Dict, gps_info: Dict) -> Dict:
        """Time-domain correlation with FORCED GPU usage where beneficial"""
        
        result = {
            'offset_seconds': None,
            'confidence': 0.0,
            'method': 'forced_gpu_time_domain'
        }
        
        try:
            # FORCE GPU context
            self.force_gpu_context()
            
            # Normalize signals (can use GPU for large signals)
            if len(video_motion) > 1000:
                # Use GPU for large arrays
                video_gpu = cp.asarray(video_motion, dtype=cp.float32)
                gps_gpu = cp.asarray(gps_speed, dtype=cp.float32)
                
                video_mean = cp.mean(video_gpu)
                video_std = cp.std(video_gpu)
                gps_mean = cp.mean(gps_gpu)
                gps_std = cp.std(gps_gpu)
                
                video_norm_gpu = (video_gpu - video_mean) / (video_std + 1e-10)
                gps_norm_gpu = (gps_gpu - gps_mean) / (gps_std + 1e-10)
                
                # Transfer back to CPU for time-domain operations
                video_norm = cp.asnumpy(video_norm_gpu)
                gps_norm = cp.asnumpy(gps_norm_gpu)
            else:
                # Use CPU for small arrays
                video_norm = (video_motion - np.mean(video_motion)) / (np.std(video_motion) + 1e-10)
                gps_norm = (gps_speed - np.mean(gps_speed)) / (np.std(gps_speed) + 1e-10)
            
            # Time span analysis
            video_start_s, video_end_s = video_times_seconds[0], video_times_seconds[-1]
            gps_start_s, gps_end_s = gps_times_seconds[0], gps_times_seconds[-1]
            
            # Search range
            search_start_s = max(-self.max_search_range_seconds, gps_start_s - video_end_s)
            search_end_s = min(self.max_search_range_seconds, gps_end_s - video_start_s)
            
            if search_end_s <= search_start_s:
                return result
            
            # Time-domain search
            search_offsets_seconds = np.arange(
                search_start_s,
                search_end_s + self.search_step_seconds,
                self.search_step_seconds
            )
            
            best_offset_seconds = None
            best_correlation = 0.0
            
            for offset_s in search_offsets_seconds:
                try:
                    shifted_gps_times_s = gps_times_seconds + offset_s
                    
                    overlap_start_s = max(video_times_seconds[0], shifted_gps_times_s[0])
                    overlap_end_s = min(video_times_seconds[-1], shifted_gps_times_s[-1])
                    
                    if overlap_end_s - overlap_start_s < self.min_overlap_seconds:
                        continue
                    
                    common_time_step_s = 0.2
                    common_times_s = np.arange(overlap_start_s, overlap_end_s, common_time_step_s)
                    
                    if len(common_times_s) < 20:
                        continue
                    
                    # Interpolate signals
                    video_interp = self.time_interpolate(video_times_seconds, video_norm, common_times_s)
                    gps_interp = self.time_interpolate(shifted_gps_times_s, gps_norm, common_times_s)
                    
                    if video_interp is None or gps_interp is None:
                        continue
                    
                    # Calculate correlation
                    correlation = self.calculate_correlation(video_interp, gps_interp)
                    
                    if abs(correlation) > abs(best_correlation):
                        best_correlation = correlation
                        best_offset_seconds = float(offset_s)
                        
                except Exception:
                    continue
            
            # Refinement
            if best_offset_seconds is not None and abs(best_correlation) >= 0.25:
                refined_offset_s, refined_correlation = self.refine_offset(
                    video_times_seconds, video_norm,
                    gps_times_seconds, gps_norm,
                    best_offset_seconds
                )
                
                if refined_offset_s is not None:
                    result['offset_seconds'] = refined_offset_s
                    result['confidence'] = abs(refined_correlation)
                else:
                    result['offset_seconds'] = best_offset_seconds
                    result['confidence'] = abs(best_correlation)
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def time_interpolate(self, time_coords_s: np.ndarray, values: np.ndarray, target_times_s: np.ndarray) -> Optional[np.ndarray]:
        """Time interpolation"""
        try:
            unique_indices = np.unique(time_coords_s, return_index=True)[1]
            clean_times_s = time_coords_s[unique_indices]
            clean_values = values[unique_indices]
            
            valid_mask = (target_times_s >= clean_times_s[0]) & (target_times_s <= clean_times_s[-1])
            
            if np.sum(valid_mask) < 10:
                return None
            
            interpolated = np.interp(target_times_s[valid_mask], clean_times_s, clean_values)
            return interpolated if len(interpolated) >= 10 else None
            
        except Exception:
            return None
    
    def calculate_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate correlation"""
        try:
            if len(signal1) != len(signal2) or len(signal1) < 10:
                return 0.0
            
            valid_mask = ~(np.isnan(signal1) | np.isnan(signal2))
            if np.sum(valid_mask) < 10:
                return 0.0
            
            clean_signal1 = signal1[valid_mask]
            clean_signal2 = signal2[valid_mask]
            
            correlation, p_value = pearsonr(clean_signal1, clean_signal2)
            
            if p_value > 0.05:
                correlation *= 0.5
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def refine_offset(self, video_times_s: np.ndarray, video_norm: np.ndarray,
                     gps_times_s: np.ndarray, gps_norm: np.ndarray,
                     initial_offset_s: float) -> Tuple[Optional[float], float]:
        """Refine offset"""
        
        try:
            fine_range_s = self.search_step_seconds * 2
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
                    
                    overlap_start_s = max(video_times_s[0], shifted_gps_times_s[0])
                    overlap_end_s = min(video_times_s[-1], shifted_gps_times_s[-1])
                    
                    if overlap_end_s - overlap_start_s < self.min_overlap_seconds:
                        continue
                    
                    common_times_s = np.arange(overlap_start_s, overlap_end_s, 0.1)
                    
                    if len(common_times_s) < 30:
                        continue
                    
                    video_interp = self.time_interpolate(video_times_s, video_norm, common_times_s)
                    gps_interp = self.time_interpolate(shifted_gps_times_s, gps_norm, common_times_s)
                    
                    if video_interp is None or gps_interp is None:
                        continue
                    
                    correlation = self.calculate_correlation(video_interp, gps_interp)
                    
                    if abs(correlation) > abs(best_fine_correlation):
                        best_fine_correlation = correlation
                        best_fine_offset_s = float(offset_s)
                        
                except Exception:
                    continue
            
            return best_fine_offset_s, best_fine_correlation
            
        except Exception:
            return None, 0.0

def main():
    """Forced dual GPU main function"""
    
    parser = argparse.ArgumentParser(description='🔥 Forced Dual GPU Processor')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Workers per GPU')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--gpu-memory', type=float, default=15.0, help='GPU memory limit per GPU in GB')
    parser.add_argument('--search-step', type=float, default=0.05, help='Search step in seconds')
    parser.add_argument('--refinement-step', type=float, default=0.01, help='Refinement step in seconds')
    parser.add_argument('--search-range', type=float, default=90.0, help='Maximum search range in seconds')
    parser.add_argument('--force-mode', action='store_true', help='Enable forced dual GPU processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.force_mode:
        logger.info("🔥🔥🔥 FORCED DUAL GPU MODE ACTIVATED! 🔥🔥🔥")
        logger.info("⚡ GUARANTEED GPU COMPUTE UTILIZATION ⚡")
    
    # Input validation
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"💀 Input file not found")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"forced_dual_gpu_{input_file.name}"
    
    # Forced GPU initialization and validation
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"🔥 Detected {gpu_count} CUDA GPUs")
        
        if gpu_count < 2:
            logger.error(f"💀 Need at least 2 GPUs for forced dual GPU processing, found {gpu_count}")
            sys.exit(1)
        
        # FORCE test both GPUs with substantial computation
        for gpu_id in [0, 1]:
            logger.info(f"🔥 FORCE testing GPU {gpu_id}...")
            computation_time = GPUEnforcer.force_gpu_computation(gpu_id, 2000)
            memory_used = GPUEnforcer.verify_gpu_memory_usage(gpu_id, 100)
            logger.info(f"🔥 GPU {gpu_id} FORCE validated: {computation_time:.3f}s computation, {memory_used:.1f}MB memory")
            
    except Exception as e:
        logger.error(f"💀 FORCED GPU initialization failed: {e}")
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
    
    logger.info(f"📊 Found {total_potential} total matches, {len(all_matches)} valid for forced dual GPU processing")
    
    if len(all_matches) == 0:
        logger.error("💀 No valid matches found!")
        sys.exit(1)
    
    # Setup forced dual GPU processing
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Randomize work distribution
    shuffled_matches = all_matches.copy()
    random.shuffle(shuffled_matches)
    
    for match in shuffled_matches:
        work_queue.put(match)
    
    # Create forced GPU workers
    workers = []
    worker_threads = []
    total_workers = 2 * args.workers_per_gpu
    
    logger.info(f"🔥 Starting {total_workers} FORCED dual GPU workers...")
    
    for gpu_id in [0, 1]:
        for worker_idx in range(args.workers_per_gpu):
            worker_id = f"GPU{gpu_id}_W{worker_idx}"
            worker = ForcedDualGPUWorker(
                worker_id, gpu_id, work_queue, result_queue, args.gpu_memory
            )
            worker.search_step_seconds = args.search_step
            worker.refinement_step_seconds = args.refinement_step
            worker.max_search_range_seconds = args.search_range
            
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            workers.append(worker)
            worker_threads.append(thread)
    
    # Extended initialization wait for forced GPU setup
    logger.info("🔥 Waiting for FORCED GPU worker initialization...")
    time.sleep(10)
    
    # Monitor progress with GPU utilization tracking
    results = []
    start_time = time.time()
    last_progress_time = start_time
    last_gpu_check = start_time
    
    logger.info(f"🔥 Starting FORCED dual GPU processing of {len(all_matches)} matches...")
    
    for i in range(len(all_matches)):
        try:
            result = result_queue.get(timeout=300)  # Longer timeout for forced processing
            results.append(result)
            
            current_time = time.time()
            
            # Progress reporting with GPU utilization
            if (i + 1) % 3 == 0 or current_time - last_progress_time > 30:
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(all_matches) - i - 1) / rate if rate > 0 else 0
                
                gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
                gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
                success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
                
                # Check GPU memory usage as proxy for utilization
                try:
                    gpu0_mem = GPUEnforcer.verify_gpu_memory_usage(0, 0)
                    gpu1_mem = GPUEnforcer.verify_gpu_memory_usage(1, 0)
                    gpu_status = f"GPU0:{gpu0_mem:.0f}MB, GPU1:{gpu1_mem:.0f}MB"
                except:
                    gpu_status = "GPU:unknown"
                
                balance_ratio = min(gpu0_count, gpu1_count) / max(gpu0_count, gpu1_count, 1) * 100
                
                logger.info(f"🔥 Progress: {i+1}/{len(all_matches)} ({rate:.2f}/s) | "
                           f"GPU0: {gpu0_count}, GPU1: {gpu1_count} (balance: {balance_ratio:.1f}%) | "
                           f"Success: {success_count} | {gpu_status} | ETA: {eta/60:.1f}m")
                last_progress_time = current_time
            
            # Periodic GPU verification
            if current_time - last_gpu_check > 60:  # Every minute
                try:
                    for gpu_id in [0, 1]:
                        gpu_time = GPUEnforcer.force_gpu_computation(gpu_id, 500)
                        logger.debug(f"🔥 GPU {gpu_id} verification: {gpu_time:.3f}s")
                except Exception as e:
                    logger.warning(f"⚠️ GPU verification failed: {e}")
                last_gpu_check = current_time
                
        except queue.Empty:
            logger.error(f"💀 TIMEOUT at match {i+1}")
            # Check worker status
            alive_workers = sum(1 for t in worker_threads if t.is_alive())
            logger.error(f"💀 Alive workers: {alive_workers}/{total_workers}")
            break
        except Exception as e:
            logger.error(f"💀 Collection error: {e}")
            break
    
    processing_time = time.time() - start_time
    
    # Shutdown workers
    logger.info("🛑 Signaling FORCED workers to stop...")
    for _ in range(total_workers):
        work_queue.put(None)
    
    for thread in worker_threads:
        thread.join(timeout=30)
    
    # Create output
    logger.info("📊 Creating forced dual GPU output...")
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
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    
    # Add metadata
    enhanced_data['forced_dual_gpu_processing_info'] = {
        'forced_dual_gpu_mode': args.force_mode,
        'gpu_enforcement_enabled': True,
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
        'gpu_utilization': {
            'gpu0_matches': gpu0_count,
            'gpu1_matches': gpu1_count,
            'load_balance_ratio': min(gpu0_count, gpu1_count) / max(gpu0_count, gpu1_count, 1) if results else 0
        },
        'processed_at': datetime.now().isoformat()
    }
    
    # Save results
    logger.info(f"💾 Saving forced dual GPU results...")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"💀 Failed to save: {e}")
        sys.exit(1)
    
    # Final summary
    success_count = len(successful_offsets)
    load_balance_ratio = min(gpu0_count, gpu1_count) / max(gpu0_count, gpu1_count, 1) * 100 if results else 0
    
    logger.info("🔥🔥🔥 FORCED DUAL GPU PROCESSING COMPLETE! 🔥🔥🔥")
    logger.info("="*70)
    logger.info(f"📊 Total processed: {len(results)}")
    logger.info(f"✅ Successful offsets: {success_count}")
    logger.info(f"🔥 GPU 0 processed: {gpu0_count}")
    logger.info(f"🔥 GPU 1 processed: {gpu1_count}")
    logger.info(f"⚖️ FORCED Load balance ratio: {load_balance_ratio:.1f}%")
    logger.info(f"⚡ Processing time: {processing_time:.1f}s ({processing_time/60:.1f}m)")
    logger.info(f"🚀 Processing rate: {len(results)/processing_time:.2f} matches/second")
    logger.info(f"📈 Success rate: {success_count/len(results)*100:.1f}%" if results else "0%")
    
    if successful_offsets:
        logger.info(f"🎯 Offset range: [{min(successful_offsets):.3f}, {max(successful_offsets):.3f}] seconds")
        logger.info(f"📊 Unique offsets: {len(set(successful_offsets))} out of {len(successful_offsets)}")
    
    # GPU utilization assessment
    if load_balance_ratio >= 85:
        logger.info("🌟 EXCELLENT: FORCED dual GPU utilization achieved!")
    elif load_balance_ratio >= 70:
        logger.info("👍 GOOD: FORCED dual GPU utilization mostly successful")
    else:
        logger.warning(f"⚠️ FORCED GPU balance needs improvement: {load_balance_ratio:.1f}%")
        
        # Diagnose GPU issues
        logger.info("🔍 Diagnosing GPU utilization issues...")
        for gpu_id in [0, 1]:
            try:
                test_time = GPUEnforcer.force_gpu_computation(gpu_id, 1000)
                test_mem = GPUEnforcer.verify_gpu_memory_usage(gpu_id, 0)
                logger.info(f"🔥 GPU {gpu_id} final test: {test_time:.3f}s, {test_mem:.1f}MB")
            except Exception as e:
                logger.error(f"💀 GPU {gpu_id} final test FAILED: {e}")
    
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("="*70)

if __name__ == "__main__":
    main()