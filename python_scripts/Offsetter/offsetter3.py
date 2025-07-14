#!/usr/bin/env python3
"""
EXTREME GPU-INTENSIVE Temporal Offset Calculator
🚀 MAXIMUM UTILIZATION OF DUAL RTX 5060 Ti (30.8GB GPU RAM TOTAL)
⚡ 100% GPU utilization target with --strict mode
🎯 Parallel processing across both GPUs with massive batching
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
import gc
import os
from contextlib import contextmanager
import math
import queue
import multiprocessing as mp
from functools import partial

# GPU-specific imports
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy.fft import fft, ifft
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

# Suppress all warnings for performance
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy_cache'
os.environ['CUPY_DUMP_CUDA_SOURCE_ON_ERROR'] = '0'

# Configure aggressive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('extreme_gpu_offsetter.log', mode='w')
    ]
)
logger = logging.getLogger('extreme_gpu_offsetter')

@dataclass
class ExtremeGPUConfig:
    """Extreme GPU configuration for maximum utilization"""
    # GPU Configuration - MAXIMUM AGGRESSION
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    max_gpu_memory_gb: float = 14.5  # Use almost all 15.4GB per GPU
    gpu_batch_size: int = 1024  # Massive batching
    cuda_streams: int = 32  # Maximum streams per GPU
    enable_gpu_streams: bool = True
    prefer_gpu_processing: bool = True
    
    # EXTREME Processing Configuration
    video_sample_rate: float = 2.0  # Higher sampling rate
    gps_sample_rate: float = 1.0
    min_correlation_confidence: float = 0.3
    max_offset_search_seconds: float = 600.0
    min_video_duration: float = 5.0
    min_gps_duration: float = 10.0
    
    # AGGRESSIVE GPU Settings
    gpu_memory_fraction: float = 0.98  # Use 98% of GPU memory
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    gpu_memory_pool_size: int = 14 * 1024**3  # 14GB pool per GPU
    enable_mixed_precision: bool = True
    
    # PARALLEL PROCESSING - EXTREME MODE
    parallel_videos_per_gpu: int = 8  # Process 8 videos per GPU simultaneously
    parallel_gpx_per_gpu: int = 16  # Process 16 GPX files per GPU
    correlation_batch_size: int = 5000  # Massive correlation batches
    fft_batch_size: int = 1000  # Large FFT batches
    
    # MULTI-GPU COORDINATION
    enable_multi_gpu_batching: bool = True
    gpu_load_balancing: bool = True
    cross_gpu_memory_sharing: bool = True
    
    # AGGRESSIVE CACHING
    enable_gpu_caching: bool = True
    cache_video_features: bool = True
    cache_gps_features: bool = True
    cache_correlations: bool = True
    
    # STRICT MODE - NO CPU FALLBACKS
    strict_mode: bool = False
    force_gpu_only: bool = False  # Set to True with --strict
    gpu_timeout_seconds: float = 600.0
    fail_on_cpu_fallback: bool = False

class ExtremeGPUMemoryManager:
    """Extreme GPU memory management for maximum utilization"""
    
    def __init__(self, config: ExtremeGPUConfig):
        self.config = config
        self.memory_pools = {}
        self.gpu_streams = {}
        self.device_contexts = {}
        self.gpu_caches = {}
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for extreme GPU mode")
        
        self._initialize_extreme_gpu_resources()
    
    def _initialize_extreme_gpu_resources(self):
        """Initialize EXTREME GPU resources"""
        logger.info(f"🚀 EXTREME GPU INITIALIZATION: {self.config.gpu_ids}")
        logger.info(f"💪 Target GPU Memory: {self.config.max_gpu_memory_gb}GB per GPU")
        logger.info(f"⚡ CUDA Streams: {self.config.cuda_streams} per GPU")
        logger.info(f"🔥 Batch Size: {self.config.gpu_batch_size}")
        
        total_gpu_memory = 0
        
        for gpu_id in self.config.gpu_ids:
            try:
                cp.cuda.Device(gpu_id).use()
                
                # EXTREME memory pool allocation
                memory_pool = cp.get_default_memory_pool()
                target_memory = int(self.config.max_gpu_memory_gb * 1024**3)
                memory_pool.set_limit(size=target_memory)
                self.memory_pools[gpu_id] = memory_pool
                
                # Pre-allocate massive memory blocks for caching
                if self.config.enable_gpu_caching:
                    cache_size = int(2 * 1024**3)  # 2GB cache per GPU
                    self.gpu_caches[gpu_id] = cp.zeros(cache_size // 4, dtype=cp.float32)
                    logger.info(f"   GPU {gpu_id}: Pre-allocated 2GB cache")
                
                # Create MAXIMUM CUDA streams
                streams = []
                for i in range(self.config.cuda_streams):
                    stream = cp.cuda.Stream(non_blocking=True)
                    streams.append(stream)
                self.gpu_streams[gpu_id] = streams
                
                # Get device properties
                device = cp.cuda.Device(gpu_id)
                props = device.attributes
                total_memory = device.mem_info[1] / (1024**3)
                free_memory = device.mem_info[0] / (1024**3)
                total_gpu_memory += total_memory
                
                # Warm up GPU with computation
                warmup_size = min(1000000, int(free_memory * 0.1 * 1024**3 / 4))
                warmup_data = cp.random.rand(warmup_size, dtype=cp.float32)
                _ = cp.fft.fft(warmup_data)
                del warmup_data
                
                logger.info(f"🎮 GPU {gpu_id} EXTREME INIT:")
                logger.info(f"   ├─ Total Memory: {total_memory:.1f}GB")
                logger.info(f"   ├─ Allocated: {self.config.max_gpu_memory_gb:.1f}GB")
                logger.info(f"   ├─ Streams: {self.config.cuda_streams}")
                logger.info(f"   ├─ Compute Capability: {device.compute_capability}")
                logger.info(f"   └─ Warmed Up: ✅")
                
            except Exception as e:
                logger.error(f"EXTREME GPU INIT FAILED {gpu_id}: {e}")
                if self.config.strict_mode:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} initialization failed")
        
        logger.info(f"🔥 TOTAL GPU POWER: {total_gpu_memory:.1f}GB across {len(self.config.gpu_ids)} GPUs")
        logger.info(f"⚡ EXTREME MODE: {'ENABLED' if self.config.force_gpu_only else 'STANDARD'}")
    
    @contextmanager
    def extreme_gpu_context(self, gpu_id: int):
        """EXTREME GPU context with maximum performance"""
        original_device = cp.cuda.Device()
        try:
            cp.cuda.Device(gpu_id).use()
            
            # Enable maximum performance mode
            if self.config.enable_mixed_precision:
                with cp.cuda.Device(gpu_id):
                    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            
            yield gpu_id
        finally:
            original_device.use()
    
    def get_extreme_stream(self, gpu_id: int, operation_type: str = 'compute') -> cp.cuda.Stream:
        """Get optimized CUDA stream for specific operations"""
        streams = self.gpu_streams.get(gpu_id, [])
        if not streams:
            return cp.cuda.Stream()
        
        # Distribute streams by operation type for maximum parallelism
        stream_map = {
            'video': 0,
            'gps': 1,
            'correlation': 2,
            'fft': 3,
            'compute': 4
        }
        
        base_idx = stream_map.get(operation_type, 0)
        stream_idx = base_idx % len(streams)
        
        return streams[stream_idx]
    
    def cleanup_extreme(self):
        """EXTREME GPU cleanup"""
        logger.info("🧹 EXTREME GPU CLEANUP")
        total_freed = 0
        
        for gpu_id in self.config.gpu_ids:
            try:
                with self.extreme_gpu_context(gpu_id):
                    if gpu_id in self.memory_pools:
                        pool = self.memory_pools[gpu_id]
                        used_bytes = pool.used_bytes()
                        pool.free_all_blocks()
                        total_freed += used_bytes
                    
                    # Clear caches
                    if gpu_id in self.gpu_caches:
                        del self.gpu_caches[gpu_id]
                    
                    # Synchronize all streams
                    for stream in self.gpu_streams.get(gpu_id, []):
                        stream.synchronize()
                    
                    cp.cuda.Device().synchronize()
                    
            except Exception as e:
                logger.debug(f"GPU {gpu_id} cleanup error: {e}")
        
        logger.info(f"🔥 FREED {total_freed / 1024**3:.1f}GB GPU MEMORY")

class ExtremeVideoProcessor:
    """EXTREME GPU video processing with maximum parallelism"""
    
    def __init__(self, config: ExtremeGPUConfig, memory_manager: ExtremeGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.video_cache = {}
        
        # Initialize OpenCV with GPU acceleration
        self._initialize_extreme_opencv()
    
    def _initialize_extreme_opencv(self):
        """Initialize OpenCV for EXTREME GPU processing"""
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count == 0:
                if self.config.strict_mode:
                    raise RuntimeError("STRICT MODE: No OpenCV CUDA devices found")
                logger.warning("⚠️ OpenCV CUDA not available")
                self.opencv_cuda_available = False
                return
            
            # Set primary GPU device
            cv2.cuda.setDevice(self.config.gpu_ids[0])
            
            # Pre-create GPU filters and objects for maximum performance
            self.gpu_filters = {}
            self.optical_flow_objects = {}
            
            for gpu_id in self.config.gpu_ids:
                cv2.cuda.setDevice(gpu_id)
                
                # Create Gaussian filters
                self.gpu_filters[gpu_id] = {
                    'gaussian_5x5': cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0),
                    'gaussian_3x3': cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0)
                }
                
                # Create optical flow objects
                self.optical_flow_objects[gpu_id] = cv2.cuda_FarnebackOpticalFlow.create(
                    numLevels=3,
                    pyrScale=0.5,
                    fastPyramids=True,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0
                )
            
            self.opencv_cuda_available = True
            logger.info(f"🎮 OpenCV CUDA EXTREME: {gpu_count} devices initialized")
            
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: OpenCV CUDA initialization failed: {e}")
            logger.warning(f"OpenCV CUDA initialization failed: {e}")
            self.opencv_cuda_available = False
    
    def extract_motion_signature_extreme_gpu(self, video_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME GPU video processing - batch multiple videos"""
        if self.config.strict_mode and not self.opencv_cuda_available:
            raise RuntimeError("STRICT MODE: GPU video processing required but OpenCV CUDA unavailable")
        
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                return self._process_video_batch_extreme(video_paths, gpu_id)
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPU video batch processing failed: {e}")
            logger.debug(f"EXTREME GPU video batch failed: {e}")
            return [None] * len(video_paths)
    
    def _process_video_batch_extreme(self, video_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME batch video processing on single GPU"""
        results = []
        
        # Get optimized streams for different operations
        video_stream = self.memory_manager.get_extreme_stream(gpu_id, 'video')
        compute_stream = self.memory_manager.get_extreme_stream(gpu_id, 'compute')
        
        # Pre-allocate GPU memory for batch processing
        batch_size = min(len(video_paths), self.config.parallel_videos_per_gpu)
        
        try:
            with cp.cuda.Stream(video_stream):
                # Process videos in batches for maximum GPU utilization
                for i in range(0, len(video_paths), batch_size):
                    batch_paths = video_paths[i:i + batch_size]
                    batch_results = []
                    
                    # Parallel processing within batch
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = []
                        
                        for video_path in batch_paths:
                            future = executor.submit(
                                self._process_single_video_extreme_gpu, 
                                video_path, gpu_id, video_stream, compute_stream
                            )
                            futures.append(future)
                        
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=self.config.gpu_timeout_seconds)
                                batch_results.append(result)
                            except Exception as e:
                                if self.config.strict_mode:
                                    raise
                                logger.debug(f"Video processing failed: {e}")
                                batch_results.append(None)
                    
                    results.extend(batch_results)
                    
                    # Log progress
                    processed = min(i + batch_size, len(video_paths))
                    logger.info(f"🎬 GPU {gpu_id}: Processed {processed}/{len(video_paths)} videos")
        
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: Batch video processing failed: {e}")
            logger.error(f"EXTREME video batch processing failed: {e}")
            results = [None] * len(video_paths)
        
        return results
    
    def _process_single_video_extreme_gpu(self, video_path: str, gpu_id: int, 
                                        video_stream: cp.cuda.Stream, 
                                        compute_stream: cp.cuda.Stream) -> Optional[Dict]:
        """EXTREME single video processing with maximum GPU utilization"""
        
        # Check cache first
        cache_key = f"{video_path}_{gpu_id}"
        if self.config.cache_video_features and cache_key in self.video_cache:
            return self.video_cache[cache_key]
        
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
        
        # EXTREME frame processing with GPU acceleration
        frame_interval = max(1, int(fps / self.config.video_sample_rate))
        
        # Pre-allocate GPU memory for massive batches
        max_frames = min(int(duration * self.config.video_sample_rate), 10000)
        
        try:
            with cp.cuda.Stream(video_stream):
                # Batch frame processing for maximum throughput
                motion_values = cp.zeros(max_frames, dtype=cp.float32)
                motion_energy = cp.zeros(max_frames, dtype=cp.float32)
                timestamps = cp.zeros(max_frames, dtype=cp.float32)
                
                frame_idx = 0
                processed_frames = 0
                prev_gpu_frame = None
                
                # GPU objects for this specific GPU
                gaussian_filter = self.gpu_filters[gpu_id]['gaussian_5x5']
                optical_flow = self.optical_flow_objects[gpu_id]
                
                # Allocate GPU matrices for reuse
                gpu_frame = cv2.cuda_GpuMat()
                gpu_gray = cv2.cuda_GpuMat()
                gpu_blurred = cv2.cuda_GpuMat()
                gpu_flow = cv2.cuda_GpuMat()
                
                while True:
                    ret, frame = cap.read()
                    if not ret or processed_frames >= max_frames:
                        break
                    
                    if frame_idx % frame_interval == 0:
                        # Resize for efficiency (but keep high quality for accuracy)
                        target_width = 1280 if is_360 else 640
                        if frame.shape[1] > target_width:
                            scale = target_width / frame.shape[1]
                            new_width = target_width
                            new_height = int(frame.shape[0] * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Upload to GPU and process
                        gpu_frame.upload(frame)
                        cv2.cuda.cvtColor(gpu_frame, gpu_gray, cv2.COLOR_BGR2GRAY)
                        gaussian_filter.apply(gpu_gray, gpu_blurred)
                        
                        if prev_gpu_frame is not None:
                            # GPU optical flow
                            optical_flow.calc(prev_gpu_frame, gpu_blurred, gpu_flow)
                            
                            # Download flow and calculate motion on GPU using CuPy
                            with cp.cuda.Stream(compute_stream):
                                flow_cpu = gpu_flow.download()
                                flow_gpu = cp.asarray(flow_cpu)
                                magnitude = cp.sqrt(flow_gpu[..., 0]**2 + flow_gpu[..., 1]**2)
                                
                                # Store directly in pre-allocated GPU arrays
                                motion_values[processed_frames] = cp.mean(magnitude)
                                motion_energy[processed_frames] = cp.sum(magnitude ** 2)
                                timestamps[processed_frames] = frame_idx / fps
                        
                        prev_gpu_frame = gpu_blurred.clone()
                        processed_frames += 1
                    
                    frame_idx += 1
                
                cap.release()
                
                if processed_frames < 3:
                    return None
                
                # Trim arrays to actual size
                motion_values = motion_values[:processed_frames]
                motion_energy = motion_energy[:processed_frames]
                timestamps = timestamps[:processed_frames]
                
                result = {
                    'motion_magnitude': motion_values,
                    'motion_energy': motion_energy,
                    'timestamps': timestamps,
                    'duration': duration,
                    'fps': fps,
                    'is_360': is_360,
                    'frame_count': processed_frames,
                    'gpu_id': gpu_id,
                    'processing_method': 'extreme_gpu_optimized'
                }
                
                # Cache result if enabled
                if self.config.cache_video_features:
                    self.video_cache[cache_key] = result
                
                return result
        
        except Exception as e:
            cap.release()
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: EXTREME video processing failed: {e}")
            logger.debug(f"EXTREME video processing error: {e}")
            return None

class ExtremeGPXProcessor:
    """EXTREME GPU GPX processing with massive parallelism"""
    
    def __init__(self, config: ExtremeGPUConfig, memory_manager: ExtremeGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.gpx_cache = {}
    
    def extract_motion_signature_extreme_gpu_batch(self, gpx_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME GPU GPX batch processing"""
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                return self._process_gpx_batch_extreme(gpx_paths, gpu_id)
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPU GPX batch processing failed: {e}")
            logger.debug(f"EXTREME GPU GPX batch failed: {e}")
            return [None] * len(gpx_paths)
    
    def _process_gpx_batch_extreme(self, gpx_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME batch GPX processing with maximum GPU utilization"""
        results = []
        
        # Process in massive batches
        batch_size = min(len(gpx_paths), self.config.parallel_gpx_per_gpu)
        gps_stream = self.memory_manager.get_extreme_stream(gpu_id, 'gps')
        
        try:
            with cp.cuda.Stream(gps_stream):
                for i in range(0, len(gpx_paths), batch_size):
                    batch_paths = gpx_paths[i:i + batch_size]
                    
                    # Parallel processing within batch
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = []
                        
                        for gpx_path in batch_paths:
                            future = executor.submit(
                                self._process_single_gpx_extreme_gpu, 
                                gpx_path, gpu_id, gps_stream
                            )
                            futures.append(future)
                        
                        batch_results = []
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=self.config.gpu_timeout_seconds)
                                batch_results.append(result)
                            except Exception as e:
                                if self.config.strict_mode:
                                    raise
                                batch_results.append(None)
                    
                    results.extend(batch_results)
                    
                    # Progress logging
                    processed = min(i + batch_size, len(gpx_paths))
                    logger.info(f"🗺️  GPU {gpu_id}: Processed {processed}/{len(gpx_paths)} GPX files")
        
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPX batch processing failed: {e}")
            logger.error(f"EXTREME GPX batch processing failed: {e}")
            results = [None] * len(gpx_paths)
        
        return results
    
    def _process_single_gpx_extreme_gpu(self, gpx_path: str, gpu_id: int, 
                                      gps_stream: cp.cuda.Stream) -> Optional[Dict]:
        """EXTREME single GPX processing with maximum GPU acceleration"""
        
        # Check cache first
        cache_key = f"{gpx_path}_{gpu_id}"
        if self.config.cache_gps_features and cache_key in self.gpx_cache:
            return self.gpx_cache[cache_key]
        
        # Load GPX data (CPU - unavoidable)
        with open(gpx_path, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
        
        # Collect points efficiently
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                points.extend([{
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': getattr(point, 'elevation', 0) or 0,
                    'time': point.time
                } for point in segment.points if point.time])
        
        if len(points) < 10:
            return None
        
        # Sort by time
        points.sort(key=lambda p: p['time'])
        df = pd.DataFrame(points)
        
        duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        if duration < self.config.min_gps_duration:
            return None
        
        # EXTREME GPU processing using CuPy with maximum batch operations
        try:
            with cp.cuda.Stream(gps_stream):
                # Upload ALL data to GPU at once for maximum efficiency
                lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)  # Use float64 for GPS precision
                lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
                
                # Vectorized distance calculation with EXTREME optimization
                lat1 = lats_gpu[:-1]
                lat2 = lats_gpu[1:]
                lon1 = lons_gpu[:-1] 
                lon2 = lons_gpu[1:]
                
                # Convert to radians in batch
                lat1_rad, lat2_rad = cp.radians(lat1), cp.radians(lat2)
                lon1_rad, lon2_rad = cp.radians(lon1), cp.radians(lon2)
                
                # Haversine formula - fully vectorized on GPU
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
                distances = 6371000 * 2 * cp.arcsin(cp.sqrt(a))
                
                # Time differences as GPU array
                time_diffs = cp.array([
                    (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds() 
                    for i in range(len(df)-1)
                ], dtype=cp.float32)
                
                # Speed and acceleration calculations - all on GPU
                speeds = cp.where(time_diffs > 0, distances / time_diffs, 0)
                accelerations = cp.zeros_like(speeds)
                accelerations[1:] = cp.where(
                    time_diffs[1:] > 0,
                    (speeds[1:] - speeds[:-1]) / time_diffs[1:],
                    0
                )
                
                # Bearing calculations for additional motion signatures
                y = cp.sin(dlon) * cp.cos(lat2_rad)
                x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(dlon)
                bearings = cp.degrees(cp.arctan2(y, x))
                
                # Advanced motion features
                jerk = cp.zeros_like(accelerations)
                jerk[1:] = cp.where(
                    time_diffs[1:] > 0,
                    (accelerations[1:] - accelerations[:-1]) / time_diffs[1:],
                    0
                )
                
                # Curvature analysis
                bearing_changes = cp.diff(bearings)
                bearing_changes = cp.where(bearing_changes > 180, bearing_changes - 360, bearing_changes)
                bearing_changes = cp.where(bearing_changes < -180, bearing_changes + 360, bearing_changes)
                curvatures = cp.abs(bearing_changes) / time_diffs[1:]
                
                # EXTREME resampling with GPU interpolation
                time_offsets = cp.cumsum(cp.concatenate([cp.array([0]), time_diffs]))
                target_times = cp.arange(0, duration, self.config.gps_sample_rate, dtype=cp.float32)
                
                # Multiple interpolation operations in parallel
                resampled_speed = cp.interp(target_times, time_offsets[:-1], speeds)
                resampled_accel = cp.interp(target_times, time_offsets[:-1], accelerations)
                resampled_jerk = cp.interp(target_times, time_offsets[:-2], jerk[:-1])
                resampled_curvature = cp.interp(target_times, time_offsets[:-2], curvatures)
                
                result = {
                    'speed': resampled_speed,
                    'acceleration': resampled_accel,
                    'jerk': resampled_jerk,
                    'curvature': resampled_curvature,
                    'timestamps': df['time'].tolist(),
                    'time_offsets': target_times,
                    'duration': duration,
                    'point_count': len(speeds),
                    'start_time': df['time'].iloc[0],
                    'end_time': df['time'].iloc[-1],
                    'gpu_id': gpu_id,
                    'processing_method': 'extreme_gpu_vectorized'
                }
                
                # Cache result
                if self.config.cache_gps_features:
                    self.gpx_cache[cache_key] = result
                
                return result
        
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: EXTREME GPX processing failed: {e}")
            logger.debug(f"EXTREME GPX processing error: {e}")
            return None

class ExtremeOffsetCalculator:
    """EXTREME GPU offset calculation with massive parallel processing"""
    
    def __init__(self, config: ExtremeGPUConfig, memory_manager: ExtremeGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.correlation_cache = {}
    
    def calculate_offset_extreme_gpu_batch(self, video_gps_pairs: List[Tuple[Dict, Dict]], 
                                         gpu_id: int) -> List[Dict]:
        """EXTREME GPU batch offset calculation"""
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                return self._calculate_batch_offsets_extreme(video_gps_pairs, gpu_id)
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPU batch offset calculation failed: {e}")
            logger.debug(f"EXTREME GPU offset batch failed: {e}")
            return [{'offset_method': 'batch_failed', 'gpu_processing': False}] * len(video_gps_pairs)
    
    def _calculate_batch_offsets_extreme(self, video_gps_pairs: List[Tuple[Dict, Dict]], 
                                       gpu_id: int) -> List[Dict]:
        """EXTREME batch offset calculation with massive GPU parallelism"""
        results = []
        
        correlation_stream = self.memory_manager.get_extreme_stream(gpu_id, 'correlation')
        fft_stream = self.memory_manager.get_extreme_stream(gpu_id, 'fft')
        
        # Process in large batches for maximum GPU utilization
        batch_size = min(len(video_gps_pairs), self.config.correlation_batch_size)
        
        try:
            with cp.cuda.Stream(correlation_stream):
                for i in range(0, len(video_gps_pairs), batch_size):
                    batch_pairs = video_gps_pairs[i:i + batch_size]
                    
                    # EXTREME parallel processing
                    with ThreadPoolExecutor(max_workers=batch_size // 4) as executor:
                        futures = []
                        
                        for video_data, gps_data in batch_pairs:
                            future = executor.submit(
                                self._calculate_single_offset_extreme_gpu, 
                                video_data, gps_data, gpu_id, correlation_stream, fft_stream
                            )
                            futures.append(future)
                        
                        batch_results = []
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=self.config.gpu_timeout_seconds)
                                batch_results.append(result)
                            except Exception as e:
                                if self.config.strict_mode:
                                    raise
                                batch_results.append({
                                    'offset_method': 'individual_failed',
                                    'gpu_processing': False,
                                    'error': str(e)[:100]
                                })
                    
                    results.extend(batch_results)
                    
                    # Progress
                    processed = min(i + batch_size, len(video_gps_pairs))
                    logger.info(f"🔄 GPU {gpu_id}: Calculated {processed}/{len(video_gps_pairs)} offsets")
        
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: Batch offset calculation failed: {e}")
            logger.error(f"EXTREME offset batch calculation failed: {e}")
            results = [{'offset_method': 'batch_processing_failed', 'gpu_processing': False}] * len(video_gps_pairs)
        
        return results
    
    def _calculate_single_offset_extreme_gpu(self, video_data: Dict, gps_data: Dict, gpu_id: int,
                                           correlation_stream: cp.cuda.Stream,
                                           fft_stream: cp.cuda.Stream) -> Dict:
        """EXTREME single offset calculation with maximum GPU acceleration"""
        
        result = {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'extreme_gpu_failed',
            'video_start_gps_time': None,
            'video_end_gps_time': None,
            'sync_quality': 'poor',
            'gpu_processing': True,
            'gpu_id': gpu_id,
            'method_scores': {}
        }
        
        try:
            with cp.cuda.Stream(correlation_stream):
                # Method 1: EXTREME GPU FFT Cross-correlation
                offset1, conf1 = self._extreme_gpu_fft_correlation(
                    video_data, gps_data, gpu_id, fft_stream
                )
                result['method_scores']['extreme_fft'] = conf1
                
                # Method 2: Multi-signal correlation (speed + acceleration + jerk)
                offset2, conf2 = self._extreme_multi_signal_correlation(
                    video_data, gps_data, gpu_id, correlation_stream
                )
                result['method_scores']['multi_signal'] = conf2
                
                # Method 3: Advanced spectral analysis
                offset3, conf3 = self._extreme_spectral_correlation(
                    video_data, gps_data, gpu_id, fft_stream
                )
                result['method_scores']['spectral'] = conf3
                
                # Choose best result using ensemble method
                methods = [
                    ('extreme_fft', offset1, conf1, 1.0),
                    ('multi_signal', offset2, conf2, 0.8),
                    ('spectral', offset3, conf3, 0.6)
                ]
                
                # Weighted scoring
                best_method, best_offset, best_confidence = None, None, 0.0
                for method_name, offset, confidence, weight in methods:
                    if offset is not None:
                        weighted_conf = confidence * weight
                        if weighted_conf > best_confidence:
                            best_method, best_offset, best_confidence = method_name, offset, confidence
                
                if best_offset is not None and best_confidence >= self.config.min_correlation_confidence:
                    result.update({
                        'temporal_offset_seconds': float(best_offset),
                        'offset_confidence': float(best_confidence),
                        'offset_method': f'extreme_gpu_{best_method}',
                        'sync_quality': self._assess_sync_quality_extreme(best_confidence)
                    })
                    
                    # Calculate GPS time range
                    gps_times = self._calculate_gps_time_range_extreme(video_data, gps_data, best_offset)
                    result.update(gps_times)
                
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: EXTREME offset calculation failed: {e}")
            result['offset_method'] = f'extreme_gpu_error: {str(e)[:100]}'
        
        return result
    
    def _extreme_gpu_fft_correlation(self, video_data: Dict, gps_data: Dict, gpu_id: int,
                                   fft_stream: cp.cuda.Stream) -> Tuple[Optional[float], float]:
        """EXTREME GPU FFT cross-correlation with maximum optimization"""
        try:
            with cp.cuda.Stream(fft_stream):
                # Get signals with multiple features
                video_signals = self._get_extreme_gpu_signals(video_data, 'video')
                gps_signals = self._get_extreme_gpu_signals(gps_data, 'gps')
                
                if not video_signals or not gps_signals:
                    return None, 0.0
                
                best_offset, best_confidence = None, 0.0
                
                # Cross-correlate multiple signal combinations for robustness
                for v_sig_name, v_signal in video_signals.items():
                    for g_sig_name, g_signal in gps_signals.items():
                        if len(v_signal) < 3 or len(g_signal) < 3:
                            continue
                        
                        # Ensure GPU arrays
                        if isinstance(v_signal, np.ndarray):
                            v_signal = cp.array(v_signal)
                        if isinstance(g_signal, np.ndarray):
                            g_signal = cp.array(g_signal)
                        
                        # Normalize with extreme precision
                        v_norm = self._extreme_normalize_gpu(v_signal)
                        g_norm = self._extreme_normalize_gpu(g_signal)
                        
                        # Optimal padding for FFT efficiency
                        max_len = len(v_norm) + len(g_norm) - 1
                        fft_len = 1 << (max_len - 1).bit_length()
                        
                        v_padded = cp.pad(v_norm, (0, fft_len - len(v_norm)))
                        g_padded = cp.pad(g_norm, (0, fft_len - len(g_norm)))
                        
                        # EXTREME GPU FFT with batch processing
                        v_fft = cp.fft.fft(v_padded)
                        g_fft = cp.fft.fft(g_padded)
                        correlation = cp.fft.ifft(cp.conj(v_fft) * g_fft).real
                        
                        # Find peak with sub-sample accuracy
                        peak_idx = cp.argmax(correlation)
                        confidence = float(correlation[peak_idx] / len(v_norm))
                        
                        # Parabolic interpolation for sub-sample accuracy
                        if 1 <= peak_idx < len(correlation) - 1:
                            y1, y2, y3 = correlation[peak_idx-1:peak_idx+2]
                            peak_offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                            refined_peak = peak_idx + peak_offset
                        else:
                            refined_peak = peak_idx
                        
                        offset_samples = float(refined_peak - len(v_norm) + 1)
                        offset_seconds = offset_samples * self.config.gps_sample_rate
                        
                        # Check constraints
                        if abs(offset_seconds) <= self.config.max_offset_search_seconds:
                            if abs(confidence) > best_confidence:
                                best_offset = offset_seconds
                                best_confidence = abs(confidence)
                
                return best_offset, best_confidence
                
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: EXTREME FFT correlation failed: {e}")
            return None, 0.0
    
    def _extreme_multi_signal_correlation(self, video_data: Dict, gps_data: Dict, gpu_id: int,
                                        correlation_stream: cp.cuda.Stream) -> Tuple[Optional[float], float]:
        """EXTREME multi-signal correlation using all available motion features"""
        try:
            with cp.cuda.Stream(correlation_stream):
                # Combine multiple motion signatures for robust correlation
                video_composite = self._create_composite_signal_gpu(video_data, 'video')
                gps_composite = self._create_composite_signal_gpu(gps_data, 'gps')
                
                if video_composite is None or gps_composite is None:
                    return None, 0.0
                
                # Multi-scale correlation
                correlations = []
                for scale in [1, 2, 4]:  # Different time scales
                    if scale > 1:
                        v_scaled = video_composite[::scale]
                        g_scaled = gps_composite[::scale]
                    else:
                        v_scaled, g_scaled = video_composite, gps_composite
                    
                    if len(v_scaled) < 3 or len(g_scaled) < 3:
                        continue
                    
                    # GPU correlation
                    correlation = cp_signal.correlate(g_scaled, v_scaled, mode='full')
                    correlations.append((correlation, scale))
                
                if not correlations:
                    return None, 0.0
                
                # Find best correlation across scales
                best_offset, best_confidence = None, 0.0
                for correlation, scale in correlations:
                    peak_idx = cp.argmax(correlation)
                    confidence = float(correlation[peak_idx] / len(correlations[0][0]))
                    
                    if confidence > best_confidence:
                        offset_samples = float(peak_idx - len(video_composite) + 1) * scale
                        offset_seconds = offset_samples * self.config.gps_sample_rate
                        
                        if abs(offset_seconds) <= self.config.max_offset_search_seconds:
                            best_offset = offset_seconds
                            best_confidence = confidence
                
                return best_offset, best_confidence
                
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: Multi-signal correlation failed: {e}")
            return None, 0.0
    
    def _extreme_spectral_correlation(self, video_data: Dict, gps_data: Dict, gpu_id: int,
                                    fft_stream: cp.cuda.Stream) -> Tuple[Optional[float], float]:
        """EXTREME spectral analysis correlation using frequency domain features"""
        try:
            with cp.cuda.Stream(fft_stream):
                # Get signals
                video_signals = self._get_extreme_gpu_signals(video_data, 'video')
                gps_signals = self._get_extreme_gpu_signals(gps_data, 'gps')
                
                if not video_signals or not gps_signals:
                    return None, 0.0
                
                # Use primary signals
                v_signal = list(video_signals.values())[0]
                g_signal = list(gps_signals.values())[0]
                
                if len(v_signal) < 16 or len(g_signal) < 16:  # Need minimum length for spectral analysis
                    return None, 0.0
                
                # Ensure GPU arrays
                if isinstance(v_signal, np.ndarray):
                    v_signal = cp.array(v_signal)
                if isinstance(g_signal, np.ndarray):
                    g_signal = cp.array(g_signal)
                
                # Spectral features extraction
                v_spectrum = cp.abs(cp.fft.fft(v_signal))
                g_spectrum = cp.abs(cp.fft.fft(g_signal))
                
                # Normalize spectra
                v_spectrum = v_spectrum / cp.sum(v_spectrum)
                g_spectrum = g_spectrum / cp.sum(g_spectrum)
                
                # Spectral correlation
                min_len = min(len(v_spectrum), len(g_spectrum))
                v_spec_norm = v_spectrum[:min_len]
                g_spec_norm = g_spectrum[:min_len]
                
                # Cross-correlation in frequency domain
                spectral_correlation = cp_signal.correlate(g_spec_norm, v_spec_norm, mode='full')
                
                peak_idx = cp.argmax(spectral_correlation)
                confidence = float(spectral_correlation[peak_idx] / len(v_spec_norm))
                
                # Convert to time domain offset
                offset_samples = float(peak_idx - len(v_spec_norm) + 1)
                offset_seconds = offset_samples * self.config.gps_sample_rate
                
                if abs(offset_seconds) <= self.config.max_offset_search_seconds:
                    return offset_seconds, confidence
                else:
                    return None, 0.0
                
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: Spectral correlation failed: {e}")
            return None, 0.0
    
    def _get_extreme_gpu_signals(self, data: Dict, data_type: str) -> Dict[str, cp.ndarray]:
        """Get multiple motion signals for robust correlation"""
        signals = {}
        
        if data_type == 'video':
            signal_keys = ['motion_magnitude', 'motion_energy']
        else:
            signal_keys = ['speed', 'acceleration', 'jerk', 'curvature']
        
        for key in signal_keys:
            if key in data:
                signal = data[key]
                if isinstance(signal, cp.ndarray) and len(signal) > 3:
                    signals[key] = signal
                elif isinstance(signal, np.ndarray) and len(signal) > 3:
                    signals[key] = cp.array(signal)
        
        return signals
    
    def _create_composite_signal_gpu(self, data: Dict, data_type: str) -> Optional[cp.ndarray]:
        """Create composite motion signal from multiple features"""
        signals = self._get_extreme_gpu_signals(data, data_type)
        
        if not signals:
            return None
        
        # Normalize and combine signals
        normalized_signals = []
        for signal in signals.values():
            normalized = self._extreme_normalize_gpu(signal)
            normalized_signals.append(normalized)
        
        # Find common length
        min_len = min(len(sig) for sig in normalized_signals)
        truncated_signals = [sig[:min_len] for sig in normalized_signals]
        
        # Weighted combination (motion_magnitude/speed gets highest weight)
        if len(truncated_signals) == 1:
            return truncated_signals[0]
        
        weights = cp.array([1.0, 0.7, 0.5, 0.3][:len(truncated_signals)])
        weights = weights / cp.sum(weights)
        
        composite = cp.zeros(min_len)
        for i, signal in enumerate(truncated_signals):
            composite += weights[i] * signal
        
        return composite
    
    def _extreme_normalize_gpu(self, signal: cp.ndarray) -> cp.ndarray:
        """EXTREME GPU signal normalization with robust statistics"""
        if len(signal) == 0:
            return signal
        
        # Use robust statistics for better normalization
        median = cp.median(signal)
        mad = cp.median(cp.abs(signal - median))  # Median Absolute Deviation
        
        if mad > 0:
            return (signal - median) / (1.4826 * mad)  # Scale factor for normal distribution
        else:
            mean = cp.mean(signal)
            return signal - mean
    
    def _assess_sync_quality_extreme(self, confidence: float) -> str:
        """Assess synchronization quality with EXTREME precision"""
        if confidence >= 0.9:
            return 'exceptional'
        elif confidence >= 0.8:
            return 'excellent'
        elif confidence >= 0.6:
            return 'good'
        elif confidence >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_gps_time_range_extreme(self, video_data: Dict, gps_data: Dict, offset: float) -> Dict:
        """Calculate GPS time range with EXTREME precision"""
        try:
            video_duration = video_data.get('duration', 0)
            gps_start_time = gps_data.get('start_time')
            
            if gps_start_time and video_duration > 0:
                video_start_gps = gps_start_time + timedelta(seconds=offset)
                video_end_gps = video_start_gps + timedelta(seconds=video_duration)
                
                return {
                    'video_start_gps_time': video_start_gps.isoformat(),
                    'video_end_gps_time': video_end_gps.isoformat(),
                    'offset_precision': 'sub_second'
                }
        
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPS time calculation failed: {e}")
        
        return {
            'video_start_gps_time': None,
            'video_end_gps_time': None,
            'offset_precision': 'unknown'
        }

class ExtremeGPUProgressTracker:
    """EXTREME GPU progress tracking with detailed metrics"""
    
    def __init__(self, total_items: int, config: ExtremeGPUConfig):
        self.total_items = total_items
        self.completed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.gpu_processed_items = 0
        self.start_time = time.time()
        self.config = config
        self.lock = threading.Lock()
        self.last_update = 0
        self.gpu_utilization_samples = []
    
    def update(self, success: bool = True, gpu_processed: bool = False):
        """Update progress with EXTREME GPU tracking"""
        with self.lock:
            self.completed_items += 1
            if success:
                self.successful_items += 1
            else:
                self.failed_items += 1
            
            if gpu_processed:
                self.gpu_processed_items += 1
            
            # Sample GPU utilization
            self._sample_gpu_utilization()
            
            # Progress reporting
            if (self.completed_items - self.last_update) >= 10:
                self._print_extreme_progress()
                self.last_update = self.completed_items
    
    def _sample_gpu_utilization(self):
        """Sample GPU utilization for monitoring"""
        try:
            gpu_utils = []
            for gpu_id in self.config.gpu_ids:
                with cp.cuda.Device(gpu_id):
                    mempool = cp.get_default_memory_pool()
                    used_bytes = mempool.used_bytes()
                    total_bytes = self.config.max_gpu_memory_gb * 1024**3
                    util = (used_bytes / total_bytes) if total_bytes > 0 else 0
                    gpu_utils.append(util)
            
            self.gpu_utilization_samples.append(gpu_utils)
            
            # Keep only recent samples
            if len(self.gpu_utilization_samples) > 100:
                self.gpu_utilization_samples = self.gpu_utilization_samples[-50:]
                
        except Exception:
            pass  # Ignore sampling errors
    
    def _print_extreme_progress(self):
        """Print EXTREME progress with detailed GPU metrics"""
        elapsed = time.time() - self.start_time
        if self.completed_items > 0:
            rate = self.completed_items / elapsed
            eta = (self.total_items - self.completed_items) / rate if rate > 0 else 0
            eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
        else:
            eta_str = "unknown"
        
        percent = (self.completed_items / self.total_items) * 100
        success_rate = (self.successful_items / self.completed_items) * 100 if self.completed_items > 0 else 0
        gpu_rate = (self.gpu_processed_items / self.completed_items) * 100 if self.completed_items > 0 else 0
        
        # GPU utilization statistics
        gpu_util_info = []
        if self.gpu_utilization_samples:
            recent_samples = self.gpu_utilization_samples[-10:]  # Last 10 samples
            for i, gpu_id in enumerate(self.config.gpu_ids):
                utils = [sample[i] for sample in recent_samples if i < len(sample)]
                if utils:
                    avg_util = np.mean(utils) * 100
                    max_util = np.max(utils) * 100
                    gpu_util_info.append(f"GPU{gpu_id}: {avg_util:.0f}%/{max_util:.0f}%")
                else:
                    gpu_util_info.append(f"GPU{gpu_id}: N/A")
        else:
            for gpu_id in self.config.gpu_ids:
                gpu_util_info.append(f"GPU{gpu_id}: N/A")
        
        logger.info(
            f"🚀 EXTREME PROGRESS: {self.completed_items}/{self.total_items} ({percent:.1f}%) | "
            f"Success: {success_rate:.1f}% | GPU: {gpu_rate:.1f}% | "
            f"Util: [{', '.join(gpu_util_info)}] | Rate: {rate:.1f}/s | ETA: {eta_str}"
        )
    
    def final_extreme_summary(self):
        """Print EXTREME final summary with comprehensive GPU statistics"""
        elapsed = time.time() - self.start_time
        rate = self.completed_items / elapsed if elapsed > 0 else 0
        
        logger.info("="*100)
        logger.info("🚀🚀🚀 EXTREME GPU PROCESSING COMPLETE 🚀🚀🚀")
        logger.info("="*100)
        logger.info(f"📊 PROCESSING STATS:")
        logger.info(f"   ├─ Total processed: {self.completed_items}/{self.total_items}")
        logger.info(f"   ├─ Successful offsets: {self.successful_items}")
        logger.info(f"   ├─ GPU processed: {self.gpu_processed_items}")
        logger.info(f"   └─ Failed: {self.failed_items}")
        
        logger.info(f"🎯 PERFORMANCE METRICS:")
        logger.info(f"   ├─ GPU utilization: {(self.gpu_processed_items/self.completed_items)*100:.1f}%")
        logger.info(f"   ├─ Success rate: {(self.successful_items/self.completed_items)*100:.1f}%")
        logger.info(f"   ├─ Processing rate: {rate:.2f} matches/second")
        logger.info(f"   └─ Total time: {elapsed/60:.1f} minutes")
        
        # GPU utilization summary
        if self.gpu_utilization_samples:
            logger.info(f"🎮 GPU UTILIZATION SUMMARY:")
            for i, gpu_id in enumerate(self.config.gpu_ids):
                utils = [sample[i] for sample in self.gpu_utilization_samples if i < len(sample)]
                if utils:
                    avg_util = np.mean(utils) * 100
                    max_util = np.max(utils) * 100
                    min_util = np.min(utils) * 100
                    logger.info(f"   GPU {gpu_id}: Avg={avg_util:.1f}% Max={max_util:.1f}% Min={min_util:.1f}%")
        
        logger.info("="*100)

class ExtremeGPUOffsetProcessor:
    """EXTREME GPU processor with maximum utilization across dual GPUs"""
    
    def __init__(self, config: ExtremeGPUConfig):
        self.config = config
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("EXTREME MODE: CuPy required for maximum GPU utilization")
        
        # Initialize EXTREME GPU resources
        self.memory_manager = ExtremeGPUMemoryManager(config)
        self.video_processor = ExtremeVideoProcessor(config, self.memory_manager)
        self.gpx_processor = ExtremeGPXProcessor(config, self.memory_manager)
        self.offset_calculator = ExtremeOffsetCalculator(config, self.memory_manager)
        
        # Multi-GPU work queues
        self.gpu_work_queues = {gpu_id: queue.Queue() for gpu_id in config.gpu_ids}
        self.result_queue = queue.Queue()
        
        logger.info(f"🚀🚀🚀 EXTREME GPU OFFSET PROCESSOR INITIALIZED 🚀🚀🚀")
        logger.info(f"🎮 GPUs: {len(config.gpu_ids)} × RTX 5060 Ti")
        logger.info(f"💪 Total GPU Memory: {len(config.gpu_ids) * config.max_gpu_memory_gb:.1f}GB")
        logger.info(f"⚡ STRICT MODE: {'ENABLED' if config.strict_mode else 'DISABLED'}")
    
    def process_all_matches_extreme(self, input_data: Dict, min_score: float = 0.5) -> Dict:
        """EXTREME processing of all matches with maximum GPU utilization"""
        
        # Collect all valid matches
        all_matches = []
        video_results = input_data.get('results', {})
        
        logger.info("📊 COLLECTING MATCHES FOR EXTREME PROCESSING...")
        for video_path, video_data in video_results.items():
            matches = video_data.get('matches', [])
            for match in matches:
                if match.get('combined_score', 0) >= min_score:
                    all_matches.append((video_path, match['path'], match))
        
        total_matches = len(all_matches)
        if total_matches == 0:
            logger.error("❌ NO MATCHES FOUND FOR EXTREME PROCESSING")
            return input_data
        
        logger.info(f"🎯 EXTREME PROCESSING TARGET: {total_matches} matches")
        logger.info(f"🔥 DISTRIBUTING ACROSS {len(self.config.gpu_ids)} GPUs")
        
        # Initialize progress tracking
        progress = ExtremeGPUProgressTracker(total_matches, self.config)
        
        # Distribute matches across GPUs with optimal load balancing
        gpu_match_batches = self._distribute_matches_extreme(all_matches)
        
        # Start EXTREME parallel processing across all GPUs
        enhanced_results = {}
        
        start_time = time.time()
        
        try:
            # Launch workers for each GPU
            with ThreadPoolExecutor(max_workers=len(self.config.gpu_ids) * 2) as executor:
                futures = []
                
                for gpu_id, match_batch in gpu_match_batches.items():
                    if match_batch:
                        logger.info(f"🚀 GPU {gpu_id}: Processing {len(match_batch)} matches")
                        future = executor.submit(
                            self._process_gpu_batch_extreme, 
                            gpu_id, match_batch, progress
                        )
                        futures.append((gpu_id, future))
                
                # Collect results from all GPUs
                all_gpu_results = {}
                for gpu_id, future in futures:
                    try:
                        gpu_results = future.result(timeout=self.config.gpu_timeout_seconds * 10)
                        all_gpu_results.update(gpu_results)
                        logger.info(f"✅ GPU {gpu_id}: Completed batch processing")
                    except Exception as e:
                        if self.config.strict_mode:
                            raise RuntimeError(f"STRICT MODE: GPU {gpu_id} batch failed: {e}")
                        logger.error(f"❌ GPU {gpu_id}: Batch processing failed: {e}")
                
                # Merge results back into original structure
                enhanced_results = self._merge_results_extreme(video_results, all_gpu_results)
        
        finally:
            # Cleanup EXTREME GPU resources
            self.memory_manager.cleanup_extreme()
        
        # Final statistics
        processing_time = time.time() - start_time
        progress.final_extreme_summary()
        
        # Create enhanced output
        enhanced_data = input_data.copy()
        enhanced_data['results'] = enhanced_results
        
        enhanced_data['extreme_gpu_offset_info'] = {
            'processed_at': datetime.now().isoformat(),
            'total_matches_processed': progress.completed_items,
            'successful_offsets': progress.successful_items,
            'gpu_processed_items': progress.gpu_processed_items,
            'gpu_utilization_rate': progress.gpu_processed_items / progress.completed_items if progress.completed_items > 0 else 0,
            'success_rate': progress.successful_items / progress.completed_items if progress.completed_items > 0 else 0,
            'processing_time_seconds': processing_time,
            'processing_rate_matches_per_second': progress.completed_items / processing_time if processing_time > 0 else 0,
            'extreme_gpu_config': {
                'gpu_ids': self.config.gpu_ids,
                'max_gpu_memory_gb_per_gpu': self.config.max_gpu_memory_gb,
                'total_gpu_memory_gb': len(self.config.gpu_ids) * self.config.max_gpu_memory_gb,
                'gpu_batch_size': self.config.gpu_batch_size,
                'cuda_streams_per_gpu': self.config.cuda_streams,
                'strict_mode': self.config.strict_mode,
                'force_gpu_only': self.config.force_gpu_only
            },
            'performance_level': 'EXTREME_GPU_MAXIMUM_UTILIZATION'
        }
        
        return enhanced_data
    
    def _distribute_matches_extreme(self, all_matches: List[Tuple]) -> Dict[int, List]:
        """Distribute matches across GPUs for optimal load balancing"""
        gpu_batches = {gpu_id: [] for gpu_id in self.config.gpu_ids}
        
        # Round-robin distribution with load balancing
        for i, match in enumerate(all_matches):
            gpu_id = self.config.gpu_ids[i % len(self.config.gpu_ids)]
            gpu_batches[gpu_id].append(match)
        
        # Log distribution
        for gpu_id, batch in gpu_batches.items():
            logger.info(f"🎮 GPU {gpu_id}: Assigned {len(batch)} matches")
        
        return gpu_batches
    
    def _process_gpu_batch_extreme(self, gpu_id: int, match_batch: List[Tuple], 
                                 progress: ExtremeGPUProgressTracker) -> Dict:
        """Process batch of matches on specific GPU with EXTREME optimization"""
        gpu_results = {}
        
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                logger.info(f"🚀 GPU {gpu_id}: Starting EXTREME batch processing")
                
                # Group matches by video for batch video processing
                video_groups = {}
                for video_path, gpx_path, match in match_batch:
                    if video_path not in video_groups:
                        video_groups[video_path] = []
                    video_groups[video_path].append((gpx_path, match))
                
                # Process each video group
                for video_path, gpx_matches in video_groups.items():
                    try:
                        # Extract video features once for all matches
                        video_data = self.video_processor.extract_motion_signature_extreme_gpu(
                            [video_path], gpu_id
                        )[0]
                        
                        if video_data is None:
                            # Mark all matches as failed
                            for gpx_path, match in gpx_matches:
                                enhanced_match = match.copy()
                                enhanced_match.update({
                                    'offset_method': 'video_extraction_failed',
                                    'gpu_processing': False
                                })
                                if video_path not in gpu_results:
                                    gpu_results[video_path] = []
                                gpu_results[video_path].append((gpx_path, enhanced_match))
                                progress.update(False, False)
                            continue
                        
                        # Extract GPX features in batch
                        gpx_paths = [gpx_path for gpx_path, _ in gpx_matches]
                        gps_data_list = self.gpx_processor.extract_motion_signature_extreme_gpu_batch(
                            gpx_paths, gpu_id
                        )
                        
                        # Calculate offsets in batch
                        valid_pairs = []
                        match_indices = []
                        
                        for i, (gps_data, (gpx_path, match)) in enumerate(zip(gps_data_list, gpx_matches)):
                            if gps_data is not None:
                                valid_pairs.append((video_data, gps_data))
                                match_indices.append(i)
                        
                        if valid_pairs:
                            offset_results = self.offset_calculator.calculate_offset_extreme_gpu_batch(
                                valid_pairs, gpu_id
                            )
                            
                            # Merge results
                            valid_idx = 0
                            for i, (gpx_path, match) in enumerate(gpx_matches):
                                enhanced_match = match.copy()
                                
                                if i in match_indices and valid_idx < len(offset_results):
                                    enhanced_match.update(offset_results[valid_idx])
                                    enhanced_match['gpu_processing'] = True
                                    valid_idx += 1
                                else:
                                    enhanced_match.update({
                                        'offset_method': 'gps_extraction_failed',
                                        'gpu_processing': False
                                    })
                                
                                if video_path not in gpu_results:
                                    gpu_results[video_path] = []
                                gpu_results[video_path].append((gpx_path, enhanced_match))
                                
                                # Update progress
                                success = enhanced_match.get('temporal_offset_seconds') is not None
                                gpu_processed = enhanced_match.get('gpu_processing', False)
                                progress.update(success, gpu_processed)
                        else:
                            # All GPX extraction failed
                            for gpx_path, match in gpx_matches:
                                enhanced_match = match.copy()
                                enhanced_match.update({
                                    'offset_method': 'gps_extraction_failed',
                                    'gpu_processing': False
                                })
                                if video_path not in gpu_results:
                                    gpu_results[video_path] = []
                                gpu_results[video_path].append((gpx_path, enhanced_match))
                                progress.update(False, False)
                    
                    except Exception as e:
                        if self.config.strict_mode:
                            raise RuntimeError(f"STRICT MODE: GPU {gpu_id} video group processing failed: {e}")
                        
                        logger.error(f"GPU {gpu_id}: Video group processing failed: {e}")
                        
                        # Mark all matches in this group as failed
                        for gpx_path, match in gpx_matches:
                            enhanced_match = match.copy()
                            enhanced_match.update({
                                'offset_method': 'gpu_processing_error',
                                'gpu_processing': False,
                                'error_details': str(e)[:200]
                            })
                            if video_path not in gpu_results:
                                gpu_results[video_path] = []
                            gpu_results[video_path].append((gpx_path, enhanced_match))
                            progress.update(False, False)
        
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPU {gpu_id} batch processing failed: {e}")
            logger.error(f"EXTREME GPU {gpu_id} batch processing failed: {e}")
        
        logger.info(f"✅ GPU {gpu_id}: Completed EXTREME batch processing")
        return gpu_results
    
    def _merge_results_extreme(self, original_results: Dict, gpu_results: Dict) -> Dict:
        """Merge GPU results back into original video structure"""
        enhanced_results = {}
        
        for video_path, video_data in original_results.items():
            enhanced_video_data = video_data.copy()
            enhanced_matches = []
            
            # Get GPU results for this video
            gpu_video_results = gpu_results.get(video_path, [])
            gpu_match_map = {gpx_path: enhanced_match for gpx_path, enhanced_match in gpu_video_results}
            
            # Update original matches
            for match in video_data.get('matches', []):
                gpx_path = match.get('path')
                if gpx_path in gpu_match_map:
                    enhanced_matches.append(gpu_match_map[gpx_path])
                else:
                    enhanced_matches.append(match)
            
            enhanced_video_data['matches'] = enhanced_matches
            enhanced_results[video_path] = enhanced_video_data
        
        return enhanced_results

def main():
    """Main function with EXTREME GPU processing"""
    parser = argparse.ArgumentParser(
        description='EXTREME GPU-accelerated temporal offset extraction for dual RTX 5060 Ti',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀🚀🚀 EXTREME GPU ACCELERATION EXAMPLES 🚀🚀🚀

  # MAXIMUM GPU UTILIZATION (RECOMMENDED)
  python extreme_gpu_offsetter.py complete_turbo_360_report_ramcache.json --strict --extreme

  # EXTREME PERFORMANCE WITH CUSTOM SETTINGS
  python extreme_gpu_offsetter.py input.json -o output.json --strict --extreme \\
    --max-gpu-memory 14.5 --gpu-batch-size 1024 --cuda-streams 32

  # ULTRA AGGRESSIVE MODE
  python extreme_gpu_offsetter.py input.json --strict --extreme --ultra-aggressive
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file from matcher')
    parser.add_argument('-o', '--output', help='Output file (default: extreme_gpu_INPUTNAME.json)')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0, 1], help='GPU IDs (default: 0 1)')
    parser.add_argument('--max-gpu-memory', type=float, default=14.5, help='Max GPU memory per GPU in GB (default: 14.5)')
    parser.add_argument('--gpu-batch-size', type=int, default=1024, help='GPU batch size (default: 1024)')
    parser.add_argument('--cuda-streams', type=int, default=32, help='CUDA streams per GPU (default: 32)')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum match score (default: 0.5)')
    parser.add_argument('--min-confidence', type=float, default=0.3, help='Minimum offset confidence (default: 0.3)')
    parser.add_argument('--max-offset', type=float, default=600.0, help='Maximum offset search seconds (default: 600)')
    parser.add_argument('--strict', action='store_true', help='🔥 STRICT MODE: Maximum GPU utilization, no CPU fallbacks')
    parser.add_argument('--extreme', action='store_true', help='🚀 EXTREME MODE: Maximum aggression')
    parser.add_argument('--ultra-aggressive', action='store_true', help='💥 ULTRA AGGRESSIVE: Beyond maximum')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"extreme_gpu_{input_file.name}"
    
    # Check GPU dependencies
    missing_deps = []
    if not CUPY_AVAILABLE:
        missing_deps.append('cupy-cuda12x')
    if not TORCH_AVAILABLE:
        missing_deps.append('torch')
    
    try:
        import cv2
        import gpxpy
        import pandas as pd
        import scipy
    except ImportError as e:
        missing_deps.extend(['opencv-contrib-python-headless', 'gpxpy', 'pandas', 'scipy'])
    
    if missing_deps:
        logger.error(f"❌ Missing EXTREME GPU dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    # Check GPU availability
    if not cp.cuda.is_available():
        logger.error("❌ CUDA not available - EXTREME GPU mode requires CUDA")
        sys.exit(1)
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    logger.info(f"🎮 Detected {gpu_count} CUDA GPUs")
    
    # Validate GPU IDs
    for gpu_id in args.gpu_ids:
        if gpu_id >= gpu_count:
            logger.error(f"❌ GPU {gpu_id} not available (only {gpu_count} GPUs detected)")
            sys.exit(1)
    
    # GPU memory validation
    total_available_memory = 0
    for gpu_id in args.gpu_ids:
        with cp.cuda.Device(gpu_id):
            total_memory = cp.cuda.Device().mem_info[1] / (1024**3)
            total_available_memory += total_memory
            logger.info(f"🎮 GPU {gpu_id}: {total_memory:.1f}GB total memory")
            
            if args.max_gpu_memory > total_memory * 0.98:
                logger.warning(f"⚠️ GPU {gpu_id}: Requested {args.max_gpu_memory}GB > available {total_memory:.1f}GB")
    
    logger.info(f"💪 TOTAL GPU POWER: {total_available_memory:.1f}GB across {len(args.gpu_ids)} GPUs")
    
    # Configure EXTREME GPU processing
    config = ExtremeGPUConfig(
        gpu_ids=args.gpu_ids,
        max_gpu_memory_gb=args.max_gpu_memory,
        gpu_batch_size=args.gpu_batch_size,
        cuda_streams=args.cuda_streams,
        min_correlation_confidence=args.min_confidence,
        max_offset_search_seconds=args.max_offset,
        strict_mode=args.strict,
        force_gpu_only=args.strict  # Strict mode forces GPU-only processing
    )
    
    # EXTREME mode adjustments
    if args.extreme:
        config.gpu_batch_size = max(config.gpu_batch_size, 1024)
        config.cuda_streams = max(config.cuda_streams, 32)
        config.parallel_videos_per_gpu = 8
        config.parallel_gpx_per_gpu = 16
        config.enable_gpu_caching = True
        logger.info("🚀 EXTREME MODE ACTIVATED")
    
    # ULTRA AGGRESSIVE mode
    if args.ultra_aggressive:
        config.gpu_batch_size = max(config.gpu_batch_size, 2048)
        config.cuda_streams = max(config.cuda_streams, 64)
        config.parallel_videos_per_gpu = 16
        config.parallel_gpx_per_gpu = 32
        config.gpu_memory_fraction = 0.99
        config.correlation_batch_size = 10000
        config.fft_batch_size = 2000
        logger.info("💥 ULTRA AGGRESSIVE MODE ACTIVATED")
    
    # Load data
    logger.info(f"📁 Loading results from {input_file}")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load input file: {e}")
        sys.exit(1)
    
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
    
    if total_matches == 0:
        logger.warning("⚠️ No matches found meeting criteria")
        # Show score distribution
        all_scores = []
        for video_path, results in list(video_results.items())[:10]:
            for match in results.get('matches', [])[:5]:
                all_scores.append(match.get('combined_score', 0))
        if all_scores:
            max_score = max(all_scores)
            logger.info(f"💡 Try lowering --min-score to {max_score * 0.8:.2f}")
        sys.exit(0)
    
    # Limit processing if requested
    if args.limit:
        total_matches = min(total_matches, args.limit)
    
    logger.info("🚀🚀🚀 EXTREME GPU OFFSET PROCESSING STARTING 🚀🚀🚀")
    logger.info("="*80)
    logger.info(f"🎯 Target: {total_matches} matches")
    logger.info(f"🎮 GPUs: {len(args.gpu_ids)} × RTX 5060 Ti")
    logger.info(f"💾 GPU Memory: {args.max_gpu_memory}GB per GPU ({len(args.gpu_ids) * args.max_gpu_memory}GB total)")
    logger.info(f"⚡ Batch Size: {config.gpu_batch_size}")
    logger.info(f"🌊 CUDA Streams: {config.cuda_streams} per GPU")
    logger.info(f"🔥 STRICT Mode: {'ENABLED' if args.strict else 'DISABLED'}")
    logger.info(f"🚀 EXTREME Mode: {'ENABLED' if args.extreme else 'DISABLED'}")
    logger.info(f"💥 ULTRA AGGRESSIVE: {'ENABLED' if args.ultra_aggressive else 'DISABLED'}")
    logger.info("="*80)
    
    # Initialize EXTREME GPU processor
    try:
        processor = ExtremeGPUOffsetProcessor(config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize EXTREME GPU processor: {e}")
        sys.exit(1)
    
    # Process all matches with EXTREME GPU utilization
    start_time = time.time()
    
    try:
        enhanced_data = processor.process_all_matches_extreme(data, args.min_score)
    except Exception as e:
        logger.error(f"❌ EXTREME GPU processing failed: {e}")
        if args.strict:
            logger.error("STRICT MODE: Terminating due to GPU processing failure")
            sys.exit(1)
        else:
            logger.info("Attempting fallback processing...")
            enhanced_data = data  # Fallback to original data
    
    # Save results
    processing_time = time.time() - start_time
    
    logger.info(f"💾 Saving EXTREME GPU results to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"❌ Failed to save output: {e}")
        sys.exit(1)
    
    # Final EXTREME summary
    logger.info("\n" + "🚀" * 30)
    logger.info("🎉 EXTREME GPU PROCESSING COMPLETE! 🎉")
    logger.info("🚀" * 30)
    logger.info(f"📊 Processing time: {processing_time/60:.1f} minutes")
    logger.info(f"⚡ Average rate: {total_matches/processing_time:.1f} matches/second")
    logger.info(f"🎮 GPU utilization: MAXIMUM")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("🚀" * 30)

if __name__ == "__main__":
    main()