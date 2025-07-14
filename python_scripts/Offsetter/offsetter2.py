#!/usr/bin/env python3
"""
Fixed GPU-accelerated temporal offset calculator
Fixes OpenCV CUDA API issues
"""

import json
import numpy as np
import cupy as cp
import torch
import cv2
import gpxpy
import pandas as pd
from pathlib import Path
import argparse
import logging
import sys
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
import psutil
import traceback
import gc
import os
from contextlib import contextmanager
import math

# GPU-specific imports
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fixed_offsetter.log', mode='w')
    ]
)
logger = logging.getLogger('fixed_offsetter')

@dataclass
class OffsetConfig:
    """Configuration for offset extraction"""
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    max_gpu_memory_gb: float = 10.0
    gpu_batch_size: int = 512
    cuda_streams: int = 8
    
    # Processing settings
    video_sample_rate: float = 1.0
    gps_sample_rate: float = 1.0
    min_correlation_confidence: float = 0.3
    max_offset_search_seconds: float = 600.0
    min_video_duration: float = 5.0
    min_gps_duration: float = 10.0
    
    # GPU optimization
    prefer_gpu_processing: bool = True
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    
    # Strict mode
    strict_mode: bool = False
    enable_validation: bool = True
    error_recovery: bool = True

class FixedGPUMemoryManager:
    """Fixed GPU memory management"""
    
    def __init__(self, config: OffsetConfig):
        self.config = config
        self.memory_pools = {}
        self.gpu_streams = {}
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU acceleration disabled")
        
        self._initialize_gpu_resources()
    
    def _initialize_gpu_resources(self):
        """Initialize GPU memory pools and streams"""
        logger.info(f"🎮 Initializing GPU resources for devices: {self.config.gpu_ids}")
        
        for gpu_id in self.config.gpu_ids:
            try:
                cp.cuda.Device(gpu_id).use()
                
                # Create memory pool
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.config.max_gpu_memory_gb * 1024**3))
                self.memory_pools[gpu_id] = memory_pool
                
                # Create CUDA streams
                streams = []
                for i in range(self.config.cuda_streams):
                    stream = cp.cuda.Stream(non_blocking=True)
                    streams.append(stream)
                self.gpu_streams[gpu_id] = streams
                
                device = cp.cuda.Device(gpu_id)
                total_memory = device.mem_info[1] / (1024**3)
                
                logger.info(f"   GPU {gpu_id}: {total_memory:.1f}GB total, "
                           f"{self.config.max_gpu_memory_gb:.1f}GB allocated, "
                           f"{self.config.cuda_streams} streams")
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU {gpu_id}: {e}")
                if self.config.strict_mode:
                    raise
    
    @contextmanager
    def gpu_context(self, gpu_id: int):
        """Context manager for GPU operations"""
        original_device = cp.cuda.Device()
        try:
            cp.cuda.Device(gpu_id).use()
            yield gpu_id
        finally:
            original_device.use()
    
    def get_stream(self, gpu_id: int, stream_idx: int = 0) -> cp.cuda.Stream:
        """Get CUDA stream for GPU"""
        streams = self.gpu_streams.get(gpu_id, [])
        if stream_idx < len(streams):
            return streams[stream_idx]
        return cp.cuda.Stream()
    
    def cleanup(self):
        """Cleanup GPU resources"""
        logger.info("🧹 Cleaning up GPU resources")
        for gpu_id in self.config.gpu_ids:
            try:
                with self.gpu_context(gpu_id):
                    if gpu_id in self.memory_pools:
                        self.memory_pools[gpu_id].free_all_blocks()
                    
                    for stream in self.gpu_streams.get(gpu_id, []):
                        stream.synchronize()
                    
                    cp.cuda.Device().synchronize()
            except Exception as e:
                logger.debug(f"GPU {gpu_id} cleanup error: {e}")

class FixedVideoProcessor:
    """Fixed video processor with correct OpenCV CUDA usage"""
    
    def __init__(self, config: OffsetConfig, memory_manager: FixedGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        
        # Check OpenCV CUDA support
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(config.gpu_ids[0])
                logger.info("✅ OpenCV CUDA support detected")
                self.opencv_cuda_available = True
            else:
                logger.warning("⚠️ OpenCV CUDA not available, using CPU fallback")
                self.opencv_cuda_available = False
        except Exception as e:
            logger.warning(f"OpenCV CUDA check failed: {e}")
            self.opencv_cuda_available = False
    
    def extract_motion_signature_gpu(self, video_path: str, gpu_id: int = 0) -> Optional[Dict]:
        """GPU-accelerated video motion signature extraction with fixed OpenCV usage"""
        try:
            with self.memory_manager.gpu_context(gpu_id):
                return self._process_video_on_gpu(video_path, gpu_id)
        except Exception as e:
            logger.debug(f"GPU video processing failed for {video_path}: {e}")
            if self.config.strict_mode:
                raise
            return None
    
    def _process_video_on_gpu(self, video_path: str, gpu_id: int) -> Optional[Dict]:
        """Core GPU video processing with fixed OpenCV CUDA API"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0 or frame_count <= 0:
            cap.release()
            return None
        
        duration = frame_count / fps
        if duration < self.config.min_video_duration:
            cap.release()
            return None
        
        # Detect 360° video
        is_360 = (width / height) >= 1.8
        
        # GPU processing setup
        stream = self.memory_manager.get_stream(gpu_id, 0)
        
        # Frame processing
        frame_interval = max(1, int(fps / self.config.video_sample_rate))
        motion_values = []
        motion_energy = []
        timestamps = []
        
        prev_frame_cpu = None
        frame_idx = 0
        
        try:
            # Create Gaussian filter for GPU (if available)
            if self.opencv_cuda_available:
                try:
                    # Use correct OpenCV CUDA API
                    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
                    optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
                    logger.info("✅ Using OpenCV CUDA filters")
                    use_gpu_filters = True
                except Exception as e:
                    logger.debug(f"OpenCV CUDA filter creation failed: {e}")
                    use_gpu_filters = False
            else:
                use_gpu_filters = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Resize for efficiency
                    if frame.shape[1] > 640:
                        scale = 640 / frame.shape[1]
                        new_width = 640
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Gaussian blur
                    if use_gpu_filters:
                        try:
                            # Upload to GPU and apply filter
                            gpu_frame = cv2.cuda_GpuMat()
                            gpu_frame.upload(gray)
                            
                            # Apply Gaussian filter on GPU
                            gpu_blurred = cv2.cuda_GpuMat()
                            gaussian_filter.apply(gpu_frame, gpu_blurred)
                            
                            # Download result
                            gray = gpu_blurred.download()
                        except Exception as e:
                            logger.debug(f"GPU filtering failed: {e}")
                            # Fallback to CPU
                            gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    else:
                        # CPU Gaussian blur
                        gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    if prev_frame_cpu is not None:
                        # Calculate optical flow
                        if use_gpu_filters:
                            try:
                                # GPU optical flow
                                gpu_prev = cv2.cuda_GpuMat()
                                gpu_curr = cv2.cuda_GpuMat()
                                gpu_prev.upload(prev_frame_cpu)
                                gpu_curr.upload(gray)
                                
                                gpu_flow = cv2.cuda_GpuMat()
                                optical_flow.calc(gpu_prev, gpu_curr, gpu_flow)
                                
                                flow = gpu_flow.download()
                            except Exception as e:
                                logger.debug(f"GPU optical flow failed: {e}")
                                # Fallback to CPU optical flow
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_frame_cpu, gray, None,
                                    0.5, 3, 15, 3, 5, 1.2, 0
                                )
                        else:
                            # CPU optical flow
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_frame_cpu, gray, None,
                                0.5, 3, 15, 3, 5, 1.2, 0
                            )
                        
                        # Calculate motion metrics using CuPy for GPU acceleration
                        try:
                            with cp.cuda.Stream(stream):
                                # Upload flow to GPU for calculation
                                flow_gpu = cp.asarray(flow)
                                magnitude = cp.sqrt(flow_gpu[..., 0]**2 + flow_gpu[..., 1]**2)
                                
                                motion_mag = float(cp.mean(magnitude))
                                motion_eng = float(cp.sum(magnitude ** 2))
                                
                                motion_values.append(motion_mag)
                                motion_energy.append(motion_eng)
                        except Exception as e:
                            # CPU fallback for motion calculation
                            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            motion_mag = np.mean(magnitude)
                            motion_eng = np.sum(magnitude ** 2)
                            
                            motion_values.append(motion_mag)
                            motion_energy.append(motion_eng)
                        
                        # Timestamp
                        timestamp = frame_idx / fps
                        timestamps.append(timestamp)
                    
                    prev_frame_cpu = gray
                
                frame_idx += 1
            
            cap.release()
            
            if len(motion_values) < 3:
                return None
            
            # Convert to CuPy arrays for GPU storage
            try:
                with cp.cuda.Stream(stream):
                    result = {
                        'motion_magnitude': cp.array(motion_values),
                        'motion_energy': cp.array(motion_energy),
                        'timestamps': cp.array(timestamps),
                        'duration': duration,
                        'fps': fps,
                        'is_360': is_360,
                        'frame_count': len(motion_values),
                        'gpu_id': gpu_id,
                        'processing_method': 'gpu_enhanced' if use_gpu_filters else 'cpu_fallback'
                    }
            except Exception as e:
                # CPU fallback for storage
                result = {
                    'motion_magnitude': np.array(motion_values),
                    'motion_energy': np.array(motion_energy),
                    'timestamps': np.array(timestamps),
                    'duration': duration,
                    'fps': fps,
                    'is_360': is_360,
                    'frame_count': len(motion_values),
                    'gpu_id': gpu_id,
                    'processing_method': 'cpu_fallback'
                }
            
            return result
            
        except Exception as e:
            cap.release()
            logger.debug(f"GPU video processing error: {e}")
            raise

class FixedGPXProcessor:
    """Fixed GPX processor with GPU acceleration"""
    
    def __init__(self, config: OffsetConfig, memory_manager: FixedGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
    
    def extract_motion_signature_gpu(self, gpx_path: str, gpu_id: int = 0) -> Optional[Dict]:
        """GPU-accelerated GPX motion signature extraction"""
        try:
            with self.memory_manager.gpu_context(gpu_id):
                return self._process_gpx_on_gpu(gpx_path, gpu_id)
        except Exception as e:
            logger.debug(f"GPU GPX processing failed for {gpx_path}: {e}")
            if self.config.strict_mode:
                raise
            return None
    
    def _process_gpx_on_gpu(self, gpx_path: str, gpu_id: int) -> Optional[Dict]:
        """Core GPU GPX processing using CuPy"""
        # Load GPX data (CPU)
        with open(gpx_path, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
        
        # Collect points
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.time:
                        points.append({
                            'lat': point.latitude,
                            'lon': point.longitude,
                            'ele': getattr(point, 'elevation', 0) or 0,
                            'time': point.time
                        })
        
        if len(points) < 10:
            return None
        
        # Sort by time and create DataFrame
        points.sort(key=lambda p: p['time'])
        df = pd.DataFrame(points)
        
        duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        if duration < self.config.min_gps_duration:
            return None
        
        # GPU-accelerated calculations using CuPy
        stream = self.memory_manager.get_stream(gpu_id, 1)
        
        try:
            with cp.cuda.Stream(stream):
                # Upload coordinates to GPU
                lats_gpu = cp.array(df['lat'].values, dtype=cp.float32)
                lons_gpu = cp.array(df['lon'].values, dtype=cp.float32)
                
                # Vectorized distance calculation on GPU (Haversine)
                lat1 = lats_gpu[:-1]
                lat2 = lats_gpu[1:]
                lon1 = lons_gpu[:-1] 
                lon2 = lons_gpu[1:]
                
                # Convert to radians
                lat1_rad = cp.radians(lat1)
                lat2_rad = cp.radians(lat2)
                lon1_rad = cp.radians(lon1)
                lon2_rad = cp.radians(lon2)
                
                # Haversine formula on GPU
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
                distances = 6371000 * 2 * cp.arcsin(cp.sqrt(a))  # meters
                
                # Time differences
                time_diffs = cp.array([
                    (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds() 
                    for i in range(len(df)-1)
                ], dtype=cp.float32)
                
                # Speed calculation on GPU
                speeds = cp.where(time_diffs > 0, distances / time_diffs, 0)
                
                # Acceleration calculation on GPU
                accelerations = cp.zeros_like(speeds)
                accelerations[1:] = cp.where(
                    time_diffs[1:] > 0,
                    (speeds[1:] - speeds[:-1]) / time_diffs[1:],
                    0
                )
                
                # Resample to consistent intervals
                resampled_data = self._resample_gps_data_gpu(
                    speeds, accelerations, 
                    cp.cumsum(cp.concatenate([cp.array([0]), time_diffs])), 
                    duration, stream
                )
        
        except Exception as e:
            logger.debug(f"GPU GPX processing failed, using CPU fallback: {e}")
            return self._process_gpx_cpu_fallback(df, duration)
        
        return {
            'speed': resampled_data['speed'],
            'acceleration': resampled_data['acceleration'],
            'timestamps': df['time'].tolist(),
            'time_offsets': resampled_data['time_offsets'],
            'duration': duration,
            'point_count': len(speeds),
            'start_time': df['time'].iloc[0],
            'end_time': df['time'].iloc[-1],
            'gpu_id': gpu_id,
            'processing_method': 'gpu_enhanced'
        }
    
    def _resample_gps_data_gpu(self, speeds: cp.ndarray, accelerations: cp.ndarray,
                              time_offsets: cp.ndarray, duration: float, 
                              stream: cp.cuda.Stream) -> Dict:
        """GPU-accelerated resampling using CuPy interpolation"""
        with cp.cuda.Stream(stream):
            # Create target time points
            target_times = cp.arange(0, duration, self.config.gps_sample_rate, dtype=cp.float32)
            
            # Interpolate using CuPy
            resampled_speed = cp.interp(target_times, time_offsets, speeds)
            resampled_accel = cp.interp(target_times, time_offsets, accelerations)
            
            return {
                'speed': resampled_speed,
                'acceleration': resampled_accel,
                'time_offsets': target_times
            }
    
    def _process_gpx_cpu_fallback(self, df: pd.DataFrame, duration: float) -> Dict:
        """CPU fallback for GPX processing"""
        # Basic CPU processing
        lats = df['lat'].values
        lons = df['lon'].values
        
        # Calculate distances using numpy
        lat1 = np.radians(lats[:-1])
        lat2 = np.radians(lats[1:])
        lon1 = np.radians(lons[:-1])
        lon2 = np.radians(lons[1:])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        distances = 6371000 * 2 * np.arcsin(np.sqrt(a))
        
        # Time differences
        time_diffs = np.array([
            (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds() 
            for i in range(len(df)-1)
        ])
        
        # Speeds and accelerations
        speeds = np.divide(distances, time_diffs, out=np.zeros_like(distances), where=time_diffs!=0)
        accelerations = np.zeros_like(speeds)
        accelerations[1:] = np.divide(
            np.diff(speeds), time_diffs[1:], 
            out=np.zeros_like(speeds[1:]), where=time_diffs[1:]!=0
        )
        
        # Resample
        time_offsets = np.cumsum(np.concatenate([[0], time_diffs]))
        target_times = np.arange(0, duration, self.config.gps_sample_rate)
        
        resampled_speed = np.interp(target_times, time_offsets[:-1], speeds)
        resampled_accel = np.interp(target_times, time_offsets[:-1], accelerations)
        
        return {
            'speed': resampled_speed,
            'acceleration': resampled_accel,
            'timestamps': df['time'].tolist(),
            'time_offsets': target_times,
            'duration': duration,
            'point_count': len(speeds),
            'start_time': df['time'].iloc[0],
            'end_time': df['time'].iloc[-1],
            'processing_method': 'cpu_fallback'
        }

class FixedOffsetCalculator:
    """Fixed offset calculator with robust GPU processing"""
    
    def __init__(self, config: OffsetConfig, memory_manager: FixedGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
    
    def calculate_offset_gpu(self, video_data: Dict, gps_data: Dict, gpu_id: int = 0) -> Dict:
        """GPU-accelerated offset calculation"""
        result = {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'failed',
            'video_start_gps_time': None,
            'video_end_gps_time': None,
            'sync_quality': 'poor',
            'gpu_processing': True
        }
        
        try:
            with self.memory_manager.gpu_context(gpu_id):
                # Method 1: GPU FFT Cross-correlation
                offset, confidence = self._gpu_fft_cross_correlation(video_data, gps_data, gpu_id)
                
                if offset is not None and confidence >= self.config.min_correlation_confidence:
                    result.update({
                        'temporal_offset_seconds': float(offset),
                        'offset_confidence': float(confidence),
                        'offset_method': 'gpu_fft_cross_correlation',
                        'sync_quality': self._assess_sync_quality(confidence)
                    })
                    
                    # Calculate GPS time range
                    gps_times = self._calculate_gps_time_range(video_data, gps_data, offset)
                    result.update(gps_times)
                else:
                    # Fallback to CPU method
                    offset, confidence = self._cpu_cross_correlation_fallback(video_data, gps_data)
                    if offset is not None and confidence >= self.config.min_correlation_confidence:
                        result.update({
                            'temporal_offset_seconds': float(offset),
                            'offset_confidence': float(confidence),
                            'offset_method': 'cpu_cross_correlation_fallback',
                            'sync_quality': self._assess_sync_quality(confidence)
                        })
                        
                        gps_times = self._calculate_gps_time_range(video_data, gps_data, offset)
                        result.update(gps_times)
        
        except Exception as e:
            logger.debug(f"GPU offset calculation failed: {e}")
            result['offset_method'] = f'gpu_error: {str(e)[:100]}'
            if self.config.strict_mode:
                raise
        
        return result
    
    def _gpu_fft_cross_correlation(self, video_data: Dict, gps_data: Dict, gpu_id: int) -> Tuple[Optional[float], float]:
        """GPU-accelerated FFT cross-correlation using CuPy"""
        try:
            stream = self.memory_manager.get_stream(gpu_id, 2)
            
            with cp.cuda.Stream(stream):
                # Get motion signals on GPU
                video_signal = self._get_gpu_signal(video_data, 'video')
                gps_signal = self._get_gpu_signal(gps_data, 'gps')
                
                if video_signal is None or gps_signal is None:
                    return None, 0.0
                
                # Ensure signals are on GPU
                if isinstance(video_signal, np.ndarray):
                    video_signal = cp.array(video_signal)
                if isinstance(gps_signal, np.ndarray):
                    gps_signal = cp.array(gps_signal)
                
                # Normalize signals on GPU
                video_norm = self._normalize_signal_gpu(video_signal)
                gps_norm = self._normalize_signal_gpu(gps_signal)
                
                # Pad to power of 2 for efficient FFT
                max_len = len(video_norm) + len(gps_norm) - 1
                pad_len = 1 << (max_len - 1).bit_length()
                
                video_padded = cp.pad(video_norm, (0, pad_len - len(video_norm)))
                gps_padded = cp.pad(gps_norm, (0, pad_len - len(gps_norm)))
                
                # GPU FFT cross-correlation
                video_fft = cp.fft.fft(video_padded)
                gps_fft = cp.fft.fft(gps_padded)
                correlation = cp.fft.ifft(cp.conj(video_fft) * gps_fft).real
                
                # Find peak
                peak_idx = cp.argmax(correlation)
                confidence = float(correlation[peak_idx] / len(video_norm))
                
                # Convert to offset
                offset_samples = int(peak_idx - len(video_norm) + 1)
                offset_seconds = offset_samples * self.config.gps_sample_rate
                
                # Clamp to reasonable range
                if abs(offset_seconds) > self.config.max_offset_search_seconds:
                    return None, 0.0
                
                return float(offset_seconds), min(abs(confidence), 1.0)
                
        except Exception as e:
            logger.debug(f"GPU FFT cross-correlation failed: {e}")
            return None, 0.0
    
    def _cpu_cross_correlation_fallback(self, video_data: Dict, gps_data: Dict) -> Tuple[Optional[float], float]:
        """CPU fallback for cross-correlation"""
        try:
            from scipy.signal import correlate
            
            # Get signals as numpy arrays
            video_signal = self._get_cpu_signal(video_data, 'video')
            gps_signal = self._get_cpu_signal(gps_data, 'gps')
            
            if video_signal is None or gps_signal is None:
                return None, 0.0
            
            # Normalize signals
            video_norm = self._normalize_signal_cpu(video_signal)
            gps_norm = self._normalize_signal_cpu(gps_signal)
            
            # Cross-correlation
            correlation = correlate(gps_norm, video_norm, mode='full')
            
            # Find peak
            peak_idx = np.argmax(correlation)
            confidence = correlation[peak_idx] / len(video_norm)
            
            # Convert to offset
            offset_samples = peak_idx - len(video_norm) + 1
            offset_seconds = offset_samples * self.config.gps_sample_rate
            
            if abs(offset_seconds) > self.config.max_offset_search_seconds:
                return None, 0.0
            
            return float(offset_seconds), min(abs(confidence), 1.0)
            
        except Exception as e:
            logger.debug(f"CPU cross-correlation fallback failed: {e}")
            return None, 0.0
    
    def _get_gpu_signal(self, data: Dict, data_type: str):
        """Get motion signal for GPU processing"""
        if data_type == 'video':
            signal_keys = ['motion_magnitude', 'motion_energy']
        else:
            signal_keys = ['speed', 'acceleration']
        
        for key in signal_keys:
            if key in data:
                signal = data[key]
                if isinstance(signal, cp.ndarray):
                    return signal
                elif isinstance(signal, np.ndarray) and len(signal) > 3:
                    return cp.array(signal)
        
        return None
    
    def _get_cpu_signal(self, data: Dict, data_type: str):
        """Get motion signal for CPU processing"""
        if data_type == 'video':
            signal_keys = ['motion_magnitude', 'motion_energy']
        else:
            signal_keys = ['speed', 'acceleration']
        
        for key in signal_keys:
            if key in data:
                signal = data[key]
                if isinstance(signal, np.ndarray):
                    return signal
                elif isinstance(signal, cp.ndarray):
                    return cp.asnumpy(signal)
        
        return None
    
    def _normalize_signal_gpu(self, signal: cp.ndarray) -> cp.ndarray:
        """Normalize signal on GPU using CuPy"""
        if len(signal) == 0:
            return signal
        
        mean = cp.mean(signal)
        std = cp.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean
    
    def _normalize_signal_cpu(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal on CPU"""
        if len(signal) == 0:
            return signal
        
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean
    
    def _assess_sync_quality(self, confidence: float) -> str:
        """Assess synchronization quality"""
        if confidence >= 0.8:
            return 'excellent'
        elif confidence >= 0.6:
            return 'good'
        elif confidence >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_gps_time_range(self, video_data: Dict, gps_data: Dict, offset: float) -> Dict:
        """Calculate GPS time range for video"""
        try:
            video_duration = video_data.get('duration', 0)
            gps_start_time = gps_data.get('start_time')
            
            if gps_start_time and video_duration > 0:
                video_start_gps = gps_start_time + timedelta(seconds=offset)
                video_end_gps = video_start_gps + timedelta(seconds=video_duration)
                
                return {
                    'video_start_gps_time': video_start_gps.isoformat(),
                    'video_end_gps_time': video_end_gps.isoformat()
                }
        
        except Exception as e:
            logger.debug(f"GPS time range calculation failed: {e}")
        
        return {
            'video_start_gps_time': None,
            'video_end_gps_time': None
        }

class FixedOffsetProcessor:
    """Main fixed offset processor"""
    
    def __init__(self, config: OffsetConfig):
        self.config = config
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU acceleration requires CuPy")
        
        self.memory_manager = FixedGPUMemoryManager(config)
        self.video_processor = FixedVideoProcessor(config, self.memory_manager)
        self.gpx_processor = FixedGPXProcessor(config, self.memory_manager)
        self.offset_calculator = FixedOffsetCalculator(config, self.memory_manager)
        
        logger.info(f"🎮 Fixed GPU Offset Processor initialized with {len(config.gpu_ids)} GPUs")
    
    def process_match(self, video_path: str, gpx_path: str, original_match: Dict) -> Dict:
        """Process single match with GPU acceleration"""
        enhanced_match = original_match.copy()
        
        # Initialize offset fields
        enhanced_match.update({
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'not_processed',
            'video_start_gps_time': None,
            'video_end_gps_time': None,
            'sync_quality': 'unknown',
            'gpu_processing': False,
            'processing_timestamp': datetime.now().isoformat()
        })
        
        try:
            # Validate files
            if not Path(video_path).exists():
                enhanced_match['offset_method'] = 'video_file_not_found'
                return enhanced_match
            
            if not Path(gpx_path).exists():
                enhanced_match['offset_method'] = 'gpx_file_not_found'
                return enhanced_match
            
            # Select GPU
            gpu_id = self.config.gpu_ids[hash(video_path) % len(self.config.gpu_ids)]
            
            # Extract motion signatures
            video_data = self.video_processor.extract_motion_signature_gpu(video_path, gpu_id)
            if video_data is None:
                enhanced_match['offset_method'] = 'video_extraction_failed'
                return enhanced_match
            
            gps_data = self.gpx_processor.extract_motion_signature_gpu(gpx_path, gpu_id)
            if gps_data is None:
                enhanced_match['offset_method'] = 'gps_extraction_failed'
                return enhanced_match
            
            # Calculate offset
            offset_result = self.offset_calculator.calculate_offset_gpu(video_data, gps_data, gpu_id)
            
            # Update match
            enhanced_match.update(offset_result)
            enhanced_match['gpu_processing'] = True
            
            # Add metadata
            enhanced_match['processing_metadata'] = {
                'gpu_id_used': gpu_id,
                'video_processing_method': video_data.get('processing_method', 'unknown'),
                'gps_processing_method': gps_data.get('processing_method', 'unknown'),
                'video_duration': video_data.get('duration', 0),
                'gps_duration': gps_data.get('duration', 0),
                'is_360_video': video_data.get('is_360', False)
            }
            
        except Exception as e:
            logger.debug(f"Processing error for {Path(video_path).name}: {e}")
            enhanced_match['offset_method'] = 'processing_error'
            enhanced_match['error_details'] = str(e)[:200]
            if self.config.strict_mode:
                raise
        
        return enhanced_match
    
    def cleanup(self):
        """Cleanup GPU resources"""
        self.memory_manager.cleanup()

def main():
    """Main function with fixed GPU processing"""
    parser = argparse.ArgumentParser(description='Fixed GPU-accelerated temporal offset calculator')
    parser.add_argument('input_file', help='Input JSON file from matcher')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0, 1], help='GPU IDs to use')
    parser.add_argument('--max-gpu-memory', type=float, default=10.0, help='Max GPU memory per GPU in GB')
    parser.add_argument('--gpu-batch-size', type=int, default=256, help='GPU batch size')
    parser.add_argument('--cuda-streams', type=int, default=8, help='CUDA streams per GPU')
    parser.add_argument('--workers', type=int, default=4, help='CPU worker threads')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum match score')
    parser.add_argument('--min-confidence', type=float, default=0.3, help='Minimum offset confidence')
    parser.add_argument('--max-offset', type=float, default=600.0, help='Maximum offset search seconds')
    parser.add_argument('--strict', action='store_true', help='Enable strict mode')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"fixed_offset_{input_file.name}"
    
    # Check dependencies
    missing_deps = []
    if not CUPY_AVAILABLE:
        missing_deps.append('cupy-cuda12x')
    if not TORCH_AVAILABLE:
        missing_deps.append('torch')
    
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    # Check GPU
    if not cp.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    logger.info(f"🎮 Detected {gpu_count} CUDA GPUs")
    
    # Configure processing
    config = OffsetConfig(
        gpu_ids=args.gpu_ids,
        max_gpu_memory_gb=args.max_gpu_memory,
        gpu_batch_size=args.gpu_batch_size,
        cuda_streams=args.cuda_streams,
        min_correlation_confidence=args.min_confidence,
        max_offset_search_seconds=args.max_offset,
        strict_mode=args.strict
    )
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Count matches
    total_matches = 0
    video_results = data.get('results', {})
    
    for video_path, results in video_results.items():
        matches = results.get('matches', [])
        for match in matches:
            if match.get('combined_score', 0) >= args.min_score:
                total_matches += 1
                if args.limit and total_matches >= args.limit:
                    break
        if args.limit and total_matches >= args.limit:
            break
    
    logger.info(f"🎮 Processing {total_matches} matches")
    
    # Initialize processor
    try:
        processor = FixedOffsetProcessor(config)
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process matches
    enhanced_results = {}
    processed_count = 0
    successful_count = 0
    
    start_time = time.time()
    
    try:
        for video_path, results in video_results.items():
            if args.limit and processed_count >= args.limit:
                break
            
            enhanced_matches = []
            
            for match in results.get('matches', []):
                if match.get('combined_score', 0) < args.min_score:
                    enhanced_matches.append(match)
                    continue
                
                enhanced_match = processor.process_match(video_path, match['path'], match)
                enhanced_matches.append(enhanced_match)
                
                processed_count += 1
                if enhanced_match.get('temporal_offset_seconds') is not None:
                    successful_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Progress: {processed_count}/{total_matches}, "
                               f"Success: {successful_count} ({successful_count/processed_count*100:.1f}%)")
                
                if args.limit and processed_count >= args.limit:
                    break
            
            enhanced_video_results = results.copy()
            enhanced_video_results['matches'] = enhanced_matches
            enhanced_results[video_path] = enhanced_video_results
    
    finally:
        processor.cleanup()
    
    # Save results
    enhanced_data = data.copy()
    enhanced_data['results'] = enhanced_results
    
    processing_time = time.time() - start_time
    enhanced_data['fixed_offset_info'] = {
        'processed_at': datetime.now().isoformat(),
        'total_matches_processed': processed_count,
        'successful_offsets': successful_count,
        'success_rate': successful_count / processed_count if processed_count > 0 else 0,
        'processing_time_seconds': processing_time,
        'gpu_config': {
            'gpu_ids': config.gpu_ids,
            'max_gpu_memory_gb': config.max_gpu_memory_gb,
            'strict_mode': config.strict_mode
        }
    }
    
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    
    # Summary
    logger.info("="*50)
    logger.info("🎮 PROCESSING COMPLETE")
    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Successful offsets: {successful_count}")
    logger.info(f"Success rate: {successful_count/processed_count*100:.1f}%")
    logger.info(f"Processing time: {processing_time/60:.1f} minutes")
    logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()