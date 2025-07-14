#!/usr/bin/env python3
"""
Fixed Production-Ready Optimized GPU Video Synchronization Processor
===================================================================

High-performance dual GPU video/GPS temporal synchronization processor with
compatibility fixes for various CUDA/CuPy versions.

Key Fixes:
- Fixed peer-to-peer access API usage
- Fallback for async memory allocation compatibility
- Enhanced error handling for CUDA operations
- Progressive feature detection and graceful degradation
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
import asyncio
import queue
from typing import Dict, List, Optional, Tuple, Union, AsyncIterator
from datetime import datetime, timedelta
import warnings
import traceback
import mmap
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import psutil
from contextlib import asynccontextmanager
import aiofiles

# Suppress warnings for production
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Configure logging
def setup_production_logging():
    """Setup production-grade logging with performance tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fixed_optimized_processor.log', mode='w')
        ]
    )
    return logging.getLogger('fixed_optimized_processor')

logger = setup_production_logging()

@dataclass
class CompatibleGPUConfig:
    """GPU configuration with compatibility fallbacks"""
    # GPU Hardware Configuration
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    memory_pool_size_gb: float = 14.0  # Per GPU
    cuda_streams_per_gpu: int = 4
    enable_peer_access: bool = True
    
    # Video Processing Configuration
    target_batch_size: int = 32  # Frames per batch
    adaptive_sampling: bool = True
    motion_threshold: float = 0.1
    hardware_decode: bool = True
    
    # Correlation Configuration
    use_fft_correlation: bool = True
    correlation_batch_size: int = 16
    enable_tensor_cores: bool = True
    mixed_precision: bool = True
    
    # I/O Configuration
    use_memory_mapping: bool = True
    async_io_workers: int = 8
    io_buffer_size: int = 1024 * 1024  # 1MB
    
    # Performance Configuration
    target_gpu_utilization: float = 0.95
    memory_bandwidth_target: float = 0.90
    enable_profiling: bool = True
    
    # Processing Limits
    max_video_duration: float = 3600.0  # seconds
    min_correlation_confidence: float = 0.3
    max_offset_search_range: float = 90.0
    
    # Compatibility flags
    use_async_memory: bool = False  # Will be auto-detected
    use_peer_access: bool = False   # Will be auto-detected

class CompatibleMemoryManager:
    """GPU memory management with enhanced compatibility"""
    
    def __init__(self, config: CompatibleGPUConfig):
        self.config = config
        self.memory_pools = {}
        self.cuda_streams = {}
        self.peer_access_enabled = False
        self.async_memory_available = False
        
        self._detect_capabilities()
        self._initialize_gpu_resources()
    
    def _detect_capabilities(self):
        """Detect available CUDA/CuPy capabilities"""
        logger.info("Detecting GPU capabilities...")
        
        # Test async memory allocation
        try:
            with cp.cuda.Device(0):
                # Try to create async memory pool
                async_pool = cp.cuda.MemoryAsyncPool()
                test_alloc = async_pool.malloc(1024)
                async_pool.free(test_alloc.ptr, 1024)
                self.async_memory_available = True
                self.config.use_async_memory = True
                logger.info("Async memory allocation: Available")
        except Exception as e:
            self.async_memory_available = False
            self.config.use_async_memory = False
            logger.warning(f"Async memory allocation: Not available ({e})")
        
        # Test peer-to-peer access
        if len(self.config.gpu_ids) >= 2:
            try:
                # Correct API usage for peer access
                can_access = cp.cuda.runtime.deviceCanAccessPeer(
                    self.config.gpu_ids[1], self.config.gpu_ids[0]
                )
                if can_access:
                    # Enable with correct API
                    cp.cuda.runtime.deviceEnablePeerAccess(self.config.gpu_ids[1])
                    self.peer_access_enabled = True
                    self.config.use_peer_access = True
                    logger.info("Peer-to-peer GPU access: Enabled")
                else:
                    logger.info("Peer-to-peer GPU access: Not supported by hardware")
            except Exception as e:
                logger.warning(f"Peer-to-peer access failed: {e}")
                self.config.use_peer_access = False
    
    def _initialize_gpu_resources(self):
        """Initialize GPU memory pools and CUDA streams with compatibility"""
        logger.info(f"Initializing GPU resources for devices: {self.config.gpu_ids}")
        
        for gpu_id in self.config.gpu_ids:
            try:
                with cp.cuda.Device(gpu_id):
                    # Configure memory pool
                    mempool = cp.get_default_memory_pool()
                    pool_size = int(self.config.memory_pool_size_gb * 1024**3)
                    mempool.set_limit(size=pool_size)
                    
                    # Use async memory pool only if available
                    if self.config.use_async_memory:
                        try:
                            cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
                            logger.info(f"GPU {gpu_id}: Async memory allocation enabled")
                        except Exception as e:
                            logger.warning(f"GPU {gpu_id}: Async allocation failed, using default: {e}")
                    
                    self.memory_pools[gpu_id] = mempool
                    
                    # Create CUDA streams
                    streams = []
                    for i in range(self.config.cuda_streams_per_gpu):
                        try:
                            stream = cp.cuda.Stream(non_blocking=True)
                            streams.append(stream)
                        except Exception as e:
                            logger.warning(f"GPU {gpu_id}: Failed to create stream {i}: {e}")
                            # Create blocking stream as fallback
                            streams.append(cp.cuda.Stream(non_blocking=False))
                    
                    self.cuda_streams[gpu_id] = streams
                    
                    # Pre-warm memory pools safely
                    self._prewarm_memory_pool_safe(gpu_id)
                    
                    # Log GPU capabilities
                    device = cp.cuda.Device(gpu_id)
                    total_memory = device.mem_info[1] / (1024**3)
                    
                    logger.info(f"GPU {gpu_id} initialized: {total_memory:.1f}GB total, "
                               f"{self.config.memory_pool_size_gb:.1f}GB allocated, "
                               f"{len(streams)} streams")
                    
            except Exception as e:
                logger.error(f"Failed to initialize GPU {gpu_id}: {e}")
                if self.config.enable_profiling:
                    logger.error(traceback.format_exc())
    
    def _prewarm_memory_pool_safe(self, gpu_id: int):
        """Safely pre-allocate common buffer sizes"""
        common_sizes = [
            1920 * 1080 * 3 * 4,  # 1080p RGBA
            1024 * 1024 * 4,      # 1MB buffer
        ]
        
        for size in common_sizes:
            try:
                with cp.cuda.Device(gpu_id):
                    buffer = cp.cuda.alloc(size)
                    # Use standard free for compatibility
                    cp.cuda.runtime.free(buffer.ptr)
            except Exception as e:
                logger.debug(f"GPU {gpu_id}: Failed to prewarm size {size}: {e}")
                continue
    
    def get_stream(self, gpu_id: int, stream_idx: int = 0) -> cp.cuda.Stream:
        """Get CUDA stream for specific GPU with fallback"""
        streams = self.cuda_streams.get(gpu_id, [])
        if stream_idx < len(streams):
            return streams[stream_idx]
        # Return default stream as fallback
        return cp.cuda.Stream()
    
    def get_memory_utilization(self, gpu_id: int) -> Dict[str, float]:
        """Get memory utilization with error handling"""
        try:
            with cp.cuda.Device(gpu_id):
                mempool = self.memory_pools.get(gpu_id)
                if mempool:
                    used_bytes = mempool.used_bytes()
                    total_bytes = mempool.total_bytes()
                else:
                    used_bytes = total_bytes = 0
                
                device_total = cp.cuda.Device().mem_info[1]
                device_free = cp.cuda.Device().mem_info[0]
                device_used = device_total - device_free
                
                return {
                    'pool_utilization': used_bytes / total_bytes if total_bytes > 0 else 0,
                    'device_utilization': device_used / device_total,
                    'pool_used_gb': used_bytes / (1024**3),
                    'device_used_gb': device_used / (1024**3)
                }
        except Exception as e:
            logger.debug(f"Memory utilization check failed for GPU {gpu_id}: {e}")
            return {'pool_utilization': 0, 'device_utilization': 0, 'pool_used_gb': 0, 'device_used_gb': 0}
    
    def cleanup(self):
        """Safe cleanup of GPU resources"""
        logger.info("Cleaning up GPU resources")
        for gpu_id in self.config.gpu_ids:
            try:
                with cp.cuda.Device(gpu_id):
                    if gpu_id in self.memory_pools:
                        self.memory_pools[gpu_id].free_all_blocks()
                    
                    # Synchronize all streams
                    for stream in self.cuda_streams.get(gpu_id, []):
                        try:
                            stream.synchronize()
                        except Exception as e:
                            logger.debug(f"Stream sync error: {e}")
                    
                    cp.cuda.Device().synchronize()
            except Exception as e:
                logger.debug(f"GPU {gpu_id} cleanup error: {e}")

class CompatibleVideoProcessor:
    """Hardware-accelerated video processing with compatibility fallbacks"""
    
    def __init__(self, config: CompatibleGPUConfig, memory_manager: CompatibleMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.motion_detectors = {}
        self.optical_flow_processors = {}
        
        self._initialize_video_processors()
    
    def _initialize_video_processors(self):
        """Initialize video processing with fallbacks"""
        for gpu_id in self.config.gpu_ids:
            try:
                with cp.cuda.Device(gpu_id):
                    # Try to initialize GPU-accelerated components
                    cv2.cuda.setDevice(gpu_id)
                    
                    # Motion detector with fallback
                    try:
                        motion_detector = cv2.cuda.createBackgroundSubtractorMOG2(
                            detectShadows=True, varThreshold=50
                        )
                        self.motion_detectors[gpu_id] = motion_detector
                        logger.info(f"GPU {gpu_id}: Hardware motion detection enabled")
                    except Exception as e:
                        logger.warning(f"GPU {gpu_id}: Hardware motion detection unavailable: {e}")
                    
                    # Optical flow processor with fallback
                    try:
                        flow_processor = cv2.cuda_FarnebackOpticalFlow.create(
                            numLevels=5, pyrScale=0.5, winSize=15,
                            numIters=3, polyN=5, polySigma=1.2
                        )
                        self.optical_flow_processors[gpu_id] = flow_processor
                        logger.info(f"GPU {gpu_id}: Hardware optical flow enabled")
                    except Exception as e:
                        logger.warning(f"GPU {gpu_id}: Hardware optical flow unavailable: {e}")
                        
            except Exception as e:
                logger.warning(f"GPU {gpu_id}: Video processor initialization failed: {e}")
    
    async def extract_motion_signature_batch(self, video_paths: List[str], gpu_id: int) -> Dict[str, Dict]:
        """Extract motion signatures with error handling"""
        results = {}
        
        try:
            with cp.cuda.Device(gpu_id):
                stream = self.memory_manager.get_stream(gpu_id, 0)
                
                for video_path in video_paths:
                    try:
                        result = await self._process_single_video_safe(video_path, gpu_id, stream)
                        if result:
                            results[video_path] = result
                    except Exception as e:
                        logger.debug(f"Failed to process {video_path}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Batch processing failed for GPU {gpu_id}: {e}")
        
        return results
    
    async def _process_single_video_safe(self, video_path: str, gpu_id: int, stream: cp.cuda.Stream) -> Optional[Dict]:
        """Safe video processing with comprehensive error handling"""
        
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                return None
            
            duration = frame_count / fps
            if duration > self.config.max_video_duration:
                return None
            
            # Detect video type
            is_360 = self._detect_360_video(width, height, video_path)
            
            # Process with safe GPU operations
            motion_data = await self._extract_motion_safe(cap, fps, width, height, is_360, gpu_id, stream)
            
            if not motion_data or len(motion_data.get('motion_values', [])) < 5:
                return None
            
            return {
                **motion_data,
                'video_info': {
                    'width': width, 'height': height, 'fps': fps,
                    'duration': duration, 'is_360': is_360
                }
            }
            
        except Exception as e:
            logger.debug(f"Video processing error for {video_path}: {e}")
            return None
        finally:
            if cap:
                cap.release()
    
    def _detect_360_video(self, width: int, height: int, video_path: str) -> bool:
        """Enhanced 360-degree video detection"""
        aspect_ratio = width / height
        filename = Path(video_path).name.lower()
        
        # Aspect ratio indicators
        if 1.8 <= aspect_ratio <= 2.2 or 0.45 <= aspect_ratio <= 0.55:
            return True
        
        # Filename indicators
        keywords_360 = ['360', 'vr', 'spherical', 'equirect', 'panoramic', 'insta360', 'theta']
        if any(kw in filename for kw in keywords_360):
            return True
        
        # Resolution indicators
        if width >= 3840 and height >= 1920:
            return True
        
        return False
    
    async def _extract_motion_safe(self, cap, fps: float, width: int, height: int, 
                                 is_360: bool, gpu_id: int, stream: cp.cuda.Stream) -> Optional[Dict]:
        """Safe motion extraction with fallbacks"""
        
        motion_values = []
        timestamps = []
        
        # Configure processing parameters
        target_fps = min(fps, 30.0)
        resize_factor = 0.5 if is_360 else 0.75
        frame_interval = max(1, int(fps / target_fps))
        
        target_width = int(width * resize_factor)
        target_height = int(height * resize_factor)
        
        frame_idx = 0
        prev_gpu_gray = None
        
        try:
            while len(motion_values) < 1000:  # Limit processing
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        # Resize frame
                        resized = cv2.resize(frame, (target_width, target_height))
                        
                        # Safe GPU processing
                        motion = await self._calculate_motion_safe(
                            resized, prev_gpu_gray, is_360, gpu_id, stream
                        )
                        
                        if motion is not None and motion > 0:
                            motion_values.append(motion)
                            timestamps.append(frame_idx / fps)
                            prev_gpu_gray = resized  # Store for next iteration
                    
                    except Exception as e:
                        logger.debug(f"Frame processing error: {e}")
                        continue
                
                frame_idx += 1
            
            return {
                'motion_values': np.array(motion_values),
                'timestamps': np.array(timestamps),
                'frame_count': len(motion_values)
            }
            
        except Exception as e:
            logger.debug(f"Motion extraction error: {e}")
            return None
    
    async def _calculate_motion_safe(self, current_frame, prev_frame, is_360: bool, 
                                   gpu_id: int, stream: cp.cuda.Stream) -> Optional[float]:
        """Safe motion calculation with GPU fallbacks"""
        
        if prev_frame is None:
            return None
        
        try:
            with cp.cuda.Device(gpu_id):
                # Convert to grayscale
                if len(current_frame.shape) == 3:
                    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                else:
                    curr_gray = current_frame
                
                if len(prev_frame.shape) == 3:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                else:
                    prev_gray = prev_frame
                
                # Try GPU processing first
                try:
                    # Upload to GPU
                    gpu_curr = cp.asarray(curr_gray, dtype=cp.float32)
                    gpu_prev = cp.asarray(prev_gray, dtype=cp.float32)
                    
                    # Calculate difference on GPU
                    if is_360:
                        # Focus on equatorial region for 360 videos
                        h, w = gpu_curr.shape
                        eq_start = h // 5
                        eq_end = 4 * h // 5
                        
                        curr_eq = gpu_curr[eq_start:eq_end, :]
                        prev_eq = gpu_prev[eq_start:eq_end, :]
                        
                        diff = cp.abs(curr_eq - prev_eq)
                        motion = float(cp.mean(diff))
                    else:
                        diff = cp.abs(gpu_curr - gpu_prev)
                        motion = float(cp.mean(diff))
                    
                    return motion
                    
                except Exception as gpu_error:
                    logger.debug(f"GPU motion calculation failed: {gpu_error}")
                    # Fallback to CPU
                    diff = cv2.absdiff(curr_gray.astype(np.float32), prev_gray.astype(np.float32))
                    return float(np.mean(diff))
        
        except Exception as e:
            logger.debug(f"Motion calculation error: {e}")
            return None

class CompatibleGPSProcessor:
    """GPS processing with enhanced error handling"""
    
    def __init__(self, config: CompatibleGPUConfig):
        self.config = config
    
    async def extract_gps_signature_batch(self, gpx_paths: List[str]) -> Dict[str, Dict]:
        """Extract GPS signatures with concurrent processing"""
        results = {}
        
        # Process with limited concurrency
        semaphore = asyncio.Semaphore(self.config.async_io_workers)
        
        async def process_single(gpx_path):
            async with semaphore:
                return await self._process_single_gpx_safe(gpx_path)
        
        tasks = [process_single(gpx_path) for gpx_path in gpx_paths]
        gpx_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for gpx_path, result in zip(gpx_paths, gpx_results):
            if isinstance(result, Exception):
                logger.debug(f"GPX processing failed for {gpx_path}: {result}")
                continue
            
            if result:
                results[gpx_path] = result
        
        return results
    
    async def _process_single_gpx_safe(self, gpx_path: str) -> Optional[Dict]:
        """Safe GPS processing with error handling"""
        try:
            async with aiofiles.open(gpx_path, 'r', encoding='utf-8') as f:
                gpx_content = await f.read()
            
            # Parse in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._parse_gpx_safe, gpx_content)
            
            return result
            
        except Exception as e:
            logger.debug(f"GPX processing error for {gpx_path}: {e}")
            return None
    
    def _parse_gpx_safe(self, gpx_content: str) -> Optional[Dict]:
        """Safe GPX parsing with vectorized calculations"""
        try:
            gpx = gpxpy.parse(gpx_content)
            
            # Extract points
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
            
            # Sort by time
            points.sort(key=lambda p: p['time'])
            
            # Vectorized speed calculation
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
            
            # Time differences and speed
            time_diffs = np.diff(times)
            time_diffs = np.maximum(time_diffs, 0.1)  # Minimum 0.1s
            
            speeds = distances / time_diffs
            
            # Filter realistic speeds (< 500 km/h)
            valid_mask = speeds < 140  # 140 m/s ≈ 500 km/h
            speeds = speeds[valid_mask]
            speed_times = times[:-1][valid_mask]
            
            if len(speeds) < 5:
                return None
            
            # Resample to uniform grid
            start_time = speed_times[0]
            end_time = speed_times[-1]
            duration = end_time - start_time
            
            if duration < 10:  # Minimum 10 seconds
                return None
            
            # Create uniform time grid (1 second intervals)
            target_times = np.arange(0, duration, 1.0)
            relative_times = speed_times - start_time
            
            # Interpolate speeds
            uniform_speeds = np.interp(target_times, relative_times, speeds)
            
            return {
                'speed': uniform_speeds,
                'timestamps': target_times,
                'duration': duration,
                'start_time': start_time,
                'point_count': len(speeds),
                'gps_info': {
                    'raw_points': len(points),
                    'valid_speeds': len(speeds),
                    'max_speed': float(np.max(speeds)),
                    'avg_speed': float(np.mean(speeds))
                }
            }
            
        except Exception as e:
            logger.debug(f"GPX parsing error: {e}")
            return None

class CompatibleCorrelator:
    """GPU correlation with compatibility and fallbacks"""
    
    def __init__(self, config: CompatibleGPUConfig, memory_manager: CompatibleMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
    
    async def correlate_batch_optimized(self, video_data_batch: List[Dict], 
                                      gps_data_batch: List[Dict], gpu_id: int) -> List[Dict]:
        """Safe batch correlation processing"""
        
        results = []
        
        try:
            with cp.cuda.Device(gpu_id):
                stream = self.memory_manager.get_stream(gpu_id, 1)
                
                # Process in smaller batches
                batch_size = min(self.config.correlation_batch_size, 8)  # Conservative batch size
                
                for i in range(0, len(video_data_batch), batch_size):
                    batch_video = video_data_batch[i:i+batch_size]
                    batch_gps = gps_data_batch[i:i+batch_size]
                    
                    batch_results = await self._process_correlation_batch_safe(
                        batch_video, batch_gps, gpu_id, stream
                    )
                    
                    results.extend(batch_results)
        except Exception as e:
            logger.error(f"Correlation batch failed for GPU {gpu_id}: {e}")
            # Return empty results for failed cases
            results.extend([self._create_failed_result() for _ in video_data_batch])
        
        return results
    
    async def _process_correlation_batch_safe(self, video_batch: List[Dict], gps_batch: List[Dict],
                                            gpu_id: int, stream: cp.cuda.Stream) -> List[Dict]:
        """Safe correlation batch processing"""
        
        results = []
        
        for video_data, gps_data in zip(video_batch, gps_batch):
            try:
                result = await self._correlate_single_pair_safe(
                    video_data, gps_data, gpu_id, stream
                )
                results.append(result)
                
            except Exception as e:
                logger.debug(f"Correlation error: {e}")
                results.append(self._create_failed_result())
        
        return results
    
    async def _correlate_single_pair_safe(self, video_data: Dict, gps_data: Dict,
                                        gpu_id: int, stream: cp.cuda.Stream) -> Dict:
        """Safe GPU correlation with CPU fallback"""
        
        result = {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'safe_correlation',
            'sync_quality': 'poor',
            'gpu_id': gpu_id
        }
        
        try:
            # Extract signals
            video_signal = video_data.get('motion_values')
            gps_signal = gps_data.get('speed')
            
            if video_signal is None or gps_signal is None:
                return result
            
            if len(video_signal) < 5 or len(gps_signal) < 5:
                return result
            
            # Try GPU correlation first
            try:
                correlation_result = await self._gpu_correlation_safe(
                    video_signal, gps_signal, gpu_id, stream
                )
                
                if correlation_result['success']:
                    result.update({
                        'temporal_offset_seconds': correlation_result['offset'],
                        'offset_confidence': correlation_result['confidence'],
                        'sync_quality': self._assess_sync_quality(correlation_result['confidence']),
                        'offset_method': 'gpu_fft_safe'
                    })
                    return result
            
            except Exception as gpu_error:
                logger.debug(f"GPU correlation failed: {gpu_error}")
            
            # Fallback to CPU correlation
            try:
                correlation_result = self._cpu_correlation_fallback(video_signal, gps_signal)
                
                if correlation_result['success']:
                    result.update({
                        'temporal_offset_seconds': correlation_result['offset'],
                        'offset_confidence': correlation_result['confidence'],
                        'sync_quality': self._assess_sync_quality(correlation_result['confidence']),
                        'offset_method': 'cpu_fallback'
                    })
            
            except Exception as cpu_error:
                logger.debug(f"CPU correlation failed: {cpu_error}")
        
        except Exception as e:
            logger.debug(f"Correlation error: {e}")
            result['offset_method'] = 'correlation_failed'
        
        return result
    
    async def _gpu_correlation_safe(self, signal1, signal2, gpu_id: int, stream: cp.cuda.Stream) -> Dict:
        """Safe GPU FFT correlation"""
        
        try:
            with cp.cuda.Device(gpu_id):
                # Normalize signals
                s1 = np.array(signal1, dtype=np.float32)
                s2 = np.array(signal2, dtype=np.float32)
                
                s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
                s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
                
                # Transfer to GPU
                gpu_s1 = cp.asarray(s1)
                gpu_s2 = cp.asarray(s2)
                
                # FFT correlation
                max_len = len(gpu_s1) + len(gpu_s2) - 1
                fft_size = 1 << (max_len - 1).bit_length()
                
                # Pad signals
                gpu_s1_padded = cp.pad(gpu_s1, (0, fft_size - len(gpu_s1)))
                gpu_s2_padded = cp.pad(gpu_s2, (0, fft_size - len(gpu_s2)))
                
                # FFT
                fft1 = cp.fft.fft(gpu_s1_padded)
                fft2 = cp.fft.fft(gpu_s2_padded)
                
                # Cross-correlation
                cross = cp.conj(fft1) * fft2
                correlation = cp.fft.ifft(cross).real
                
                # Find peak
                peak_idx = cp.argmax(cp.abs(correlation))
                peak_value = correlation[peak_idx]
                
                # Calculate offset and confidence
                offset_samples = int(peak_idx) - len(gpu_s1) + 1
                
                # Normalize confidence
                correlation_abs = cp.abs(correlation)
                confidence = float(correlation_abs[peak_idx] / cp.mean(correlation_abs))
                confidence = min(confidence / 10.0, 1.0)  # Scale down
                
                return {
                    'success': True,
                    'offset': float(offset_samples),
                    'confidence': confidence
                }
                
        except Exception as e:
            logger.debug(f"GPU correlation error: {e}")
            return {'success': False, 'offset': 0, 'confidence': 0.0}
    
    def _cpu_correlation_fallback(self, signal1, signal2) -> Dict:
        """CPU correlation fallback"""
        
        try:
            from scipy import signal as scipy_signal
            
            # Normalize
            s1 = np.array(signal1, dtype=np.float32)
            s2 = np.array(signal2, dtype=np.float32)
            
            s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
            s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
            
            # Cross-correlation
            correlation = scipy_signal.correlate(s1, s2, mode='full')
            
            # Find peak
            peak_idx = np.argmax(np.abs(correlation))
            peak_value = correlation[peak_idx]
            
            # Calculate offset and confidence
            offset_samples = int(peak_idx) - len(s1) + 1
            confidence = float(np.abs(peak_value) / np.mean(np.abs(correlation)))
            confidence = min(confidence / 10.0, 1.0)
            
            return {
                'success': True,
                'offset': float(offset_samples),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.debug(f"CPU correlation error: {e}")
            return {'success': False, 'offset': 0, 'confidence': 0.0}
    
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
    
    def _create_failed_result(self) -> Dict:
        """Create failed result"""
        return {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'failed',
            'sync_quality': 'poor'
        }

class CompatiblePipelineProcessor:
    """Main pipeline with enhanced compatibility"""
    
    def __init__(self, config: CompatibleGPUConfig):
        self.config = config
        
        # Initialize components
        self.memory_manager = CompatibleMemoryManager(config)
        self.video_processor = CompatibleVideoProcessor(config, self.memory_manager)
        self.gps_processor = CompatibleGPSProcessor(config)
        self.correlator = CompatibleCorrelator(config, self.memory_manager)
        
        logger.info("Compatible pipeline processor initialized")
    
    async def process_matches_batch(self, matches: List[Tuple[str, str, Dict]]) -> List[Dict]:
        """Process matches with enhanced error handling"""
        
        start_time = time.time()
        
        # Split workload
        video_paths = [match[0] for match in matches]
        gps_paths = [match[1] for match in matches]
        original_matches = [match[2] for match in matches]
        
        # Distribute across GPUs
        gpu_batches = self._distribute_workload_safe(matches)
        
        # Process video and GPS concurrently
        video_results = {}
        gps_results = {}
        
        # Video processing
        video_tasks = []
        for gpu_id, gpu_matches in gpu_batches.items():
            if gpu_matches:
                gpu_video_paths = [m[0] for m in gpu_matches]
                task = self.video_processor.extract_motion_signature_batch(gpu_video_paths, gpu_id)
                video_tasks.append(task)
        
        for task in video_tasks:
            try:
                results = await task
                video_results.update(results)
            except Exception as e:
                logger.error(f"Video processing task failed: {e}")
        
        # GPS processing
        try:
            gps_results = await self.gps_processor.extract_gps_signature_batch(gps_paths)
        except Exception as e:
            logger.error(f"GPS processing failed: {e}")
        
        # Correlation
        correlation_results = []
        for gpu_id, gpu_matches in gpu_batches.items():
            if gpu_matches:
                batch_video_data = []
                batch_gps_data = []
                
                for video_path, gps_path, _ in gpu_matches:
                    video_data = video_results.get(video_path, {})
                    gps_data = gps_results.get(gps_path, {})
                    
                    batch_video_data.append(video_data)
                    batch_gps_data.append(gps_data)
                
                try:
                    correlations = await self.correlator.correlate_batch_optimized(
                        batch_video_data, batch_gps_data, gpu_id
                    )
                    correlation_results.extend(correlations)
                except Exception as e:
                    logger.error(f"Correlation failed for GPU {gpu_id}: {e}")
                    correlation_results.extend([
                        self.correlator._create_failed_result() 
                        for _ in gpu_matches
                    ])
        
        # Combine results
        final_results = []
        for i, (video_path, gps_path, original_match) in enumerate(matches):
            enhanced_match = original_match.copy()
            
            if i < len(correlation_results):
                enhanced_match.update(correlation_results[i])
            
            enhanced_match.update({
                'processing_method': 'compatible_pipeline',
                'video_data_available': video_path in video_results,
                'gps_data_available': gps_path in gps_results,
                'processing_time': time.time() - start_time
            })
            
            final_results.append(enhanced_match)
        
        return final_results
    
    def _distribute_workload_safe(self, matches: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Safe workload distribution"""
        gpu_batches = {gpu_id: [] for gpu_id in self.config.gpu_ids}
        
        for i, match in enumerate(matches):
            gpu_id = self.config.gpu_ids[i % len(self.config.gpu_ids)]
            gpu_batches[gpu_id].append(match)
        
        return gpu_batches
    
    def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up compatible pipeline")
        self.memory_manager.cleanup()

async def main_async():
    """Main async function with enhanced error handling"""
    
    parser = argparse.ArgumentParser(
        description='Fixed Compatible GPU Video Synchronization Processor'
    )
    
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--memory-per-gpu', type=float, default=14.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cuda-streams', type=int, default=4)
    parser.add_argument('--min-score', type=float, default=0.3)
    parser.add_argument('--max-matches', type=int, default=10)
    parser.add_argument('--async-io-workers', type=int, default=8)
    parser.add_argument('--enable-profiling', action='store_true')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"fixed_optimized_{input_file.name}"
    
    # Create configuration
    config = CompatibleGPUConfig(
        gpu_ids=args.gpu_ids,
        memory_pool_size_gb=args.memory_per_gpu,
        target_batch_size=args.batch_size,
        cuda_streams_per_gpu=args.cuda_streams,
        enable_profiling=args.enable_profiling,
        async_io_workers=args.async_io_workers
    )
    
    # Validate GPU availability
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"Detected {gpu_count} CUDA GPUs")
        
        for gpu_id in args.gpu_ids:
            if gpu_id >= gpu_count:
                logger.error(f"GPU {gpu_id} not available")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        sys.exit(1)
    
    # Load and process data
    logger.info(f"Loading input data from {input_file}")
    try:
        async with aiofiles.open(input_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        sys.exit(1)
    
    # Collect matches
    all_matches = []
    for video_path, video_data in data.get('results', {}).items():
        matches = video_data.get('matches', [])
        sorted_matches = sorted(matches, key=lambda x: x.get('combined_score', 0), reverse=True)
        top_matches = sorted_matches[:args.max_matches]
        
        for match in top_matches:
            if match.get('combined_score', 0) >= args.min_score:
                gpx_path = match.get('path', '')
                if Path(video_path).exists() and Path(gpx_path).exists():
                    all_matches.append((video_path, gpx_path, match))
    
    logger.info(f"Processing {len(all_matches)} matches with compatible pipeline")
    
    if not all_matches:
        logger.error("No valid matches found")
        sys.exit(1)
    
    # Process with compatible pipeline
    processor = None
    try:
        processor = CompatiblePipelineProcessor(config)
        
        start_time = time.time()
        
        # Process in chunks
        chunk_size = config.target_batch_size * len(config.gpu_ids)
        all_results = []
        
        for i in range(0, len(all_matches), chunk_size):
            chunk = all_matches[i:i+chunk_size]
            
            logger.info(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} matches)")
            
            chunk_results = await processor.process_matches_batch(chunk)
            all_results.extend(chunk_results)
        
        processing_time = time.time() - start_time
        
        # Create output
        enhanced_data = data.copy()
        enhanced_results = {}
        
        # Map results back
        result_index = 0
        for video_path, video_data in data.get('results', {}).items():
            enhanced_video_data = video_data.copy()
            enhanced_matches = []
            
            original_matches = video_data.get('matches', [])
            processed_count = 0
            
            for original_match in original_matches:
                if (original_match.get('combined_score', 0) >= args.min_score and 
                    processed_count < args.max_matches and
                    result_index < len(all_results)):
                    
                    enhanced_matches.append(all_results[result_index])
                    result_index += 1
                    processed_count += 1
                else:
                    enhanced_matches.append(original_match)
            
            enhanced_video_data['matches'] = enhanced_matches
            enhanced_results[video_path] = enhanced_video_data
        
        enhanced_data['results'] = enhanced_results
        
        # Add metadata
        enhanced_data['compatible_processing_info'] = {
            'processing_completed_at': datetime.now().isoformat(),
            'total_processing_time_seconds': processing_time,
            'matches_processed': len(all_results),
            'processing_rate_per_minute': len(all_results) / (processing_time / 60),
            'compatibility_features': [
                'safe_gpu_operations',
                'fallback_mechanisms',
                'enhanced_error_handling',
                'api_compatibility_fixes'
            ],
            'configuration': {
                'gpu_ids': config.gpu_ids,
                'memory_pool_size_gb': config.memory_pool_size_gb,
                'async_memory_enabled': config.use_async_memory,
                'peer_access_enabled': config.use_peer_access
            }
        }
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(enhanced_data, indent=2, default=str))
        
        # Final report
        successful_results = [r for r in all_results if r.get('temporal_offset_seconds') is not None]
        
        logger.info("COMPATIBLE PROCESSING COMPLETE")
        logger.info(f"Total matches processed: {len(all_results)}")
        logger.info(f"Successful synchronizations: {len(successful_results)}")
        logger.info(f"Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        logger.info(f"Processing time: {processing_time:.1f}s ({processing_time/60:.1f}m)")
        logger.info(f"Processing rate: {len(all_results)/(processing_time/60):.1f} matches/minute")
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.enable_profiling:
            logger.error(traceback.format_exc())
        sys.exit(1)
        
    finally:
        if processor:
            processor.cleanup()

def main():
    """Main entry point"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Processing interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()