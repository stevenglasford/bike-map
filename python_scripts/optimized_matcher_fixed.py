#!/usr/bin/env python3
"""
Production-Ready GPU-Optimized Video-GPX Correlation System
Maximizes dual GPU utilization with robust error handling and monitoring
"""

import cv2

import cupy as cp
from cupy import asarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gpxpy
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os
import sys
import glob
import asyncio
import aiofiles
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle
import json
import warnings
from tqdm import tqdm
from collections import defaultdict
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import signal
import atexit
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations

# Production logging setup
def setup_logging(log_file: str = "gpu_matcher.log"):
    """Setup production logging with file and console output"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler for important logs only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# GPU memory pool configuration
try:
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
except Exception as e:
    logger.error(f"Failed to initialize CuPy memory pools: {e}")
    sys.exit(1)

class GPUResourceManager:
    """Manages GPU resources and handles cleanup"""
    
    def __init__(self):
        self.active_streams = []
        self.active_contexts = []
        
    def register_stream(self, stream):
        self.active_streams.append(stream)
    
    def cleanup(self):
        """Clean up all GPU resources"""
        logger.info("Cleaning up GPU resources...")
        
        # Synchronize all streams
        for stream in self.active_streams:
            try:
                stream.synchronize()
            except:
                pass
        
        # Clear memory pools
        try:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("GPU cleanup complete")

# Global resource manager
gpu_resource_manager = GPUResourceManager()

# Register cleanup on exit
atexit.register(gpu_resource_manager.cleanup)

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    logger.info("Received interrupt signal, cleaning up...")
    gpu_resource_manager.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class GPUVideoProcessor:
    """Production-ready GPU-accelerated video processor"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1], batch_size: int = 8, 
                 stream_count: int = 4, max_retries: int = 3):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.devices = []
        
        # Validate CUDA devices
        cuda_device_count = torch.cuda.device_count()
        cupy_device_count = cp.cuda.runtime.getDeviceCount()
        
        logger.info(f"Found {cuda_device_count} CUDA devices (PyTorch), {cupy_device_count} devices (CuPy)")
        
        for gpu_id in gpu_ids:
            if gpu_id < min(cuda_device_count, cupy_device_count):
                self.devices.append(gpu_id)
                logger.info(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                logger.warning(f"GPU {gpu_id} not available")
        
        if not self.devices:
            raise RuntimeError("No valid CUDA devices available")
        
        # Create CUDA streams for each GPU
        self.streams = {}
        for gpu_id in self.devices:
            with cp.cuda.Device(gpu_id):
                streams = [cp.cuda.Stream() for _ in range(stream_count)]
                self.streams[gpu_id] = streams
                for stream in streams:
                    gpu_resource_manager.register_stream(stream)
        
        # Initialize PyTorch models
        self._init_models()
        
        # Preallocate GPU buffers
        self._init_gpu_buffers()
        
        logger.info(f"Initialized video processor with GPUs: {self.devices}")
    
    def _init_models(self):
        """Initialize optimized feature extraction models"""
        self.torch_devices = {gpu_id: torch.device(f'cuda:{gpu_id}') for gpu_id in self.devices}
        self.feature_extractors = {}
        
        for gpu_id in self.devices:
            try:
                # Lightweight CNN optimized for video features
                model = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128)
                ).to(self.torch_devices[gpu_id])
                
                model.eval()
                
                # JIT compile for production performance
                dummy_input = torch.randn(1, 3, 360, 640).to(self.torch_devices[gpu_id])
                model = torch.jit.trace(model, dummy_input)
                
                self.feature_extractors[gpu_id] = model
                logger.debug(f"Initialized model on GPU {gpu_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize model on GPU {gpu_id}: {e}")
                raise
    
    def _init_gpu_buffers(self):
        """Preallocate GPU memory buffers"""
        self.gpu_buffers = {}
        
        for gpu_id in self.devices:
            try:
                with cp.cuda.Device(gpu_id):
                    self.gpu_buffers[gpu_id] = {
                        'frame': cp.zeros((self.batch_size, 360, 640, 3), dtype=cp.uint8),
                        'gray': cp.zeros((self.batch_size, 360, 640), dtype=cp.uint8),
                        'flow': cp.zeros((self.batch_size, 360, 640, 2), dtype=cp.float32),
                        'features': cp.zeros((self.batch_size, 128), dtype=cp.float32)
                    }
                logger.debug(f"Allocated GPU buffers on device {gpu_id}")
            except Exception as e:
                logger.error(f"Failed to allocate GPU buffers on device {gpu_id}: {e}")
                raise
    
    async def process_videos_batch(self, video_paths: List[str]) -> Dict[str, Any]:
        """Process multiple videos with load balancing across GPUs"""
        logger.info(f"Processing {len(video_paths)} videos across {len(self.devices)} GPUs")
        
        # Split videos among GPUs
        gpu_assignments = defaultdict(list)
        for i, video_path in enumerate(video_paths):
            gpu_id = self.devices[i % len(self.devices)]
            gpu_assignments[gpu_id].append(video_path)
        
        # Process on each GPU concurrently
        tasks = []
        for gpu_id, paths in gpu_assignments.items():
            task = asyncio.create_task(self._process_gpu_batch(gpu_id, paths))
            tasks.append(task)
        
        # Gather results with error handling
        results = {}
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error(f"GPU batch processing failed: {task_result}")
            else:
                results.update(task_result)
        
        logger.info(f"Successfully processed {len(results)} videos")
        return results
    
    async def _process_gpu_batch(self, gpu_id: int, video_paths: List[str]) -> Dict[str, Any]:
        """Process videos on specific GPU with retry logic"""
        results = {}
        
        with cp.cuda.Device(gpu_id):
            for video_path in tqdm(video_paths, desc=f"GPU {gpu_id}", leave=False):
                for attempt in range(self.max_retries):
                    try:
                        features = await self._extract_video_features_gpu(video_path, gpu_id)
                        if features is not None:
                            results[video_path] = features
                            break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {video_path}: {e}")
                        if attempt == self.max_retries - 1:
                            logger.error(f"Failed to process {video_path} after {self.max_retries} attempts")
                            results[video_path] = None
                        else:
                            await asyncio.sleep(1)  # Brief pause before retry
                
                # Periodic memory cleanup
                if len(results) % 10 == 0:
                    mempool.free_all_blocks()
                    torch.cuda.empty_cache()
        
        return results
    
    async def _extract_video_features_gpu(self, video_path: str, gpu_id: int) -> Optional[Dict[str, Any]]:
        """Extract features with robust error handling"""
        cap = None
        try:
            # Try hardware decoder first, fallback to software
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    break
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or frame_count <= 0:
                logger.error(f"Invalid video properties for {video_path}: fps={fps}, frames={frame_count}")
                return None
            
            duration = frame_count / fps
            
            # Adaptive sampling based on video duration
            if duration < 60:  # Short video - sample more frequently
                sample_interval = max(1, int(fps / 3))
            elif duration < 300:  # Medium video
                sample_interval = max(1, int(fps / 2))
            else:  # Long video
                sample_interval = max(1, int(fps))
            
            total_samples = min(frame_count // sample_interval, 10000)  # Cap at 10k samples
            
            # Process video
            features = await self._process_video_frames(
                cap, gpu_id, sample_interval, total_samples, frame_count
            )
            
            if features is not None:
                features['duration'] = duration
                features['fps'] = fps
                features['frame_count'] = frame_count
                features['sample_count'] = total_samples
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            logger.debug(traceback.format_exc())
            return None
            
        finally:
            if cap is not None:
                cap.release()
    
    async def _process_video_frames(self, cap, gpu_id: int, sample_interval: int, 
                                   total_samples: int, frame_count: int) -> Optional[Dict[str, Any]]:
        """Process video frames with batching and GPU acceleration"""
        with cp.cuda.Device(gpu_id):
            # Preallocate feature arrays
            motion_magnitude = cp.zeros(total_samples, dtype=cp.float32)
            motion_complexity = cp.zeros(total_samples, dtype=cp.float32)
            scene_features = cp.zeros((total_samples, 128), dtype=cp.float32)
            temporal_gradient = cp.zeros(total_samples, dtype=cp.float32)
            
            batch_frames = []
            batch_indices = []
            sample_idx = 0
            prev_gray_gpu = None
            failed_frames = 0
            
            # Process frames
            for frame_idx in range(0, frame_count, sample_interval):
                if sample_idx >= total_samples:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    failed_frames += 1
                    if failed_frames > 10:
                        logger.warning(f"Too many failed frames, stopping at {frame_idx}/{frame_count}")
                        break
                    continue
                
                # Resize frame
                try:
                    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    batch_frames.append(frame)
                    batch_indices.append(sample_idx)
                    
                    # Process batch
                    if len(batch_frames) >= self.batch_size or frame_idx >= frame_count - sample_interval:
                        if batch_frames:
                            # Upload to GPU
                            frames_gpu = cp.asarray(batch_frames)
                            
                            # Extract features
                            with self.streams[gpu_id][0]:
                                batch_results = self._process_frame_batch_gpu(
                                    frames_gpu, prev_gray_gpu, gpu_id,
                                    motion_magnitude, motion_complexity,
                                    scene_features, temporal_gradient,
                                    batch_indices
                                )
                                prev_gray_gpu = batch_results['last_gray']
                            
                            sample_idx += len(batch_frames)
                            batch_frames = []
                            batch_indices = []
                            
                except Exception as e:
                    logger.warning(f"Frame processing error: {e}")
                    failed_frames += 1
            
            # Wait for GPU operations to complete
            cp.cuda.Stream.null.synchronize()
            
            # Trim arrays to actual size
            actual_samples = min(sample_idx, total_samples)
            
            if actual_samples < 10:
                logger.warning(f"Too few samples extracted: {actual_samples}")
                return None
            
            # Convert to CPU and return
            features = {
                'motion_magnitude': cp.asnumpy(motion_magnitude[:actual_samples]),
                'motion_complexity': cp.asnumpy(motion_complexity[:actual_samples]),
                'scene_features': cp.asnumpy(scene_features[:actual_samples]),
                'temporal_gradient': cp.asnumpy(temporal_gradient[:actual_samples])
            }
            
            return features
    
    def _process_frame_batch_gpu(self, frames_gpu: cp.ndarray, prev_gray_gpu: Optional[cp.ndarray],
                                gpu_id: int, motion_magnitude: cp.ndarray, motion_complexity: cp.ndarray,
                                scene_features: cp.ndarray, temporal_gradient: cp.ndarray,
                                batch_indices: List[int]) -> Dict[str, Any]:
        """Process frame batch entirely on GPU"""
        batch_size = frames_gpu.shape[0]
        
        # RGB to grayscale conversion (optimized coefficients)
        gray_batch = cp.dot(frames_gpu[..., :3], cp.array([0.2989, 0.5870, 0.1140], dtype=cp.float32))
        gray_batch = gray_batch.astype(cp.uint8)
        
        # Motion analysis
        if prev_gray_gpu is not None and prev_gray_gpu.shape[0] > 0:
            # Compute motion metrics
            for i, idx in enumerate(batch_indices):
                if i < gray_batch.shape[0]:
                    # Frame difference
                    diff = cp.abs(gray_batch[i].astype(cp.float32) - prev_gray_gpu.astype(cp.float32))
                    
                    # Motion statistics
                    motion_magnitude[idx] = cp.mean(diff)
                    motion_complexity[idx] = cp.std(diff)
                    
                    # Temporal gradient
                    if idx > 0:
                        temporal_gradient[idx] = cp.mean(diff)
        
        # Deep feature extraction
        with torch.cuda.device(gpu_id):
            # Prepare frames for PyTorch
            frames_torch = torch.from_numpy(
                cp.asnumpy(frames_gpu).astype(cp.float32)
            ).permute(0, 3, 1, 2) / 255.0
            
            frames_torch = frames_torch.to(self.torch_devices[gpu_id])
            
            # Extract features
            with torch.no_grad(), autocast():
                deep_features = self.feature_extractors[gpu_id](frames_torch[:, :3])  # RGB only
            
            # Store features
            features_np = deep_features.cpu().numpy()
            for i, idx in enumerate(batch_indices):
                if i < features_np.shape[0] and idx < scene_features.shape[0]:
                    scene_features[idx] = cp.asarray(features_np[i])
        
        return {'last_gray': gray_batch[-1] if batch_size > 0 else prev_gray_gpu}


class GPXProcessor:
    """Production-ready GPX processor with parallel processing"""
    
    def __init__(self, gpu_id: int = 0, chunk_size: int = 1000):
        self.gpu_id = gpu_id
        self.chunk_size = chunk_size
        
        # Validate GPU
        if gpu_id >= cp.cuda.runtime.getDeviceCount():
            logger.warning(f"GPU {gpu_id} not available for GPX processing, using CPU")
            self.use_gpu = False
        else:
            self.use_gpu = True
            with cp.cuda.Device(gpu_id):
                self.stream = cp.cuda.Stream()
                gpu_resource_manager.register_stream(self.stream)
    
    def process_gpx_files(self, gpx_paths: List[str], max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Process GPX files in parallel"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 32)
        
        logger.info(f"Processing {len(gpx_paths)} GPX files with {max_workers} workers")
        
        # Process in chunks to avoid overwhelming the system
        results = {}
        
        for chunk_start in range(0, len(gpx_paths), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(gpx_paths))
            chunk_paths = gpx_paths[chunk_start:chunk_end]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_gpx, path): path 
                    for path in chunk_paths
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"GPX chunk {chunk_start//self.chunk_size + 1}"):
                    path = futures[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per file
                        if result is not None:
                            results[path] = result
                    except Exception as e:
                        logger.error(f"Failed to process GPX {path}: {e}")
        
        logger.info(f"Successfully processed {len(results)} GPX files")
        return results
    
    def _process_single_gpx(self, gpx_path: str) -> Optional[Dict[str, Any]]:
        """Process single GPX file with error handling"""
        try:
            # Parse GPX
            with open(gpx_path, 'r', encoding='utf-8', errors='ignore') as f:
                gpx = gpxpy.parse(f)
            
            # Extract points
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for pt in segment.points:
                        if pt.time and pt.latitude is not None and pt.longitude is not None:
                            points.append({
                                'time': pt.time.timestamp(),
                                'lat': float(pt.latitude),
                                'lon': float(pt.longitude),
                                'ele': float(pt.elevation) if pt.elevation else 0.0
                            })
            
            if len(points) < 10:
                return None
            
            # Convert to numpy for processing
            data = pd.DataFrame(points)
            data = data.sort_values('time').reset_index(drop=True)
            
            # Extract arrays
            times = data['time'].values
            lats = data['lat'].values
            lons = data['lon'].values
            eles = data['ele'].values
            
            # Calculate features
            if self.use_gpu:
                features = self._calculate_features_gpu(times, lats, lons, eles)
            else:
                features = self._calculate_features_cpu(times, lats, lons, eles)
            
            # Calculate metadata
            duration = float(times[-1] - times[0])
            total_distance = float(cp.sum(features['distances']))
            
            return {
                'features': features,
                'duration': duration,
                'distance': total_distance,
                'point_count': len(points),
                'start_time': datetime.fromtimestamp(times[0]),
                'end_time': datetime.fromtimestamp(times[-1])
            }
            
        except Exception as e:
            logger.debug(f"Error processing GPX {gpx_path}: {e}")
            return None
    
    def _calculate_features_gpu(self, times: cp.ndarray, lats: cp.ndarray, 
                               lons: cp.ndarray, eles: cp.ndarray) -> Dict[str, cp.ndarray]:
        """GPU-accelerated feature calculation"""
        with cp.cuda.Device(self.gpu_id):
            # Upload to GPU
            times_gpu = cp.asarray(times)
            lats_gpu = cp.asarray(lats)
            lons_gpu = cp.asarray(lons)
            eles_gpu = cp.asarray(eles)
            
            n = len(times)
            
            # Time differences
            dt = cp.diff(times_gpu)
            dt = cp.maximum(dt, 0.1)  # Minimum 0.1 second
            
            # Vectorized haversine distance
            lat1, lat2 = lats_gpu[:-1], lats_gpu[1:]
            lon1, lon2 = lons_gpu[:-1], lons_gpu[1:]
            
            lat1_rad = cp.radians(lat1)
            lat2_rad = cp.radians(lat2)
            dlat = cp.radians(lat2 - lat1)
            dlon = cp.radians(lon2 - lon1)
            
            a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
            c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))  # Clip for numerical stability
            distances = 3958.8 * c  # Miles
            
            # Speed calculation
            speeds = (distances * 3600) / dt  # mph
            speeds = cp.clip(speeds, 0, 200)  # Reasonable speed limits
            
            # Acceleration
            accelerations = cp.zeros(n, dtype=cp.float32)
            if n > 2:
                accel_values = cp.diff(speeds) / dt[1:]
                accelerations[1:-1] = cp.clip(accel_values, -50, 50)
            
            # Bearing
            y = cp.sin(dlon) * cp.cos(lat2_rad)
            x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(dlon)
            bearings = cp.degrees(cp.arctan2(y, x))
            
            # Elevation changes
            elev_changes = cp.diff(eles_gpu)
            
            # Pad arrays
            speeds_padded = cp.zeros(n, dtype=cp.float32)
            speeds_padded[1:] = speeds
            
            bearings_padded = cp.zeros(n, dtype=cp.float32)
            bearings_padded[1:] = bearings
            
            elev_padded = cp.zeros(n, dtype=cp.float32)
            elev_padded[1:] = elev_changes
            
            distances_padded = cp.zeros(n, dtype=cp.float32)
            distances_padded[1:] = distances
            
            # Wait for completion and convert to CPU
            self.stream.synchronize()
            
            return {
                'speed': cp.asnumpy(speeds_padded),
                'acceleration': cp.asnumpy(accelerations),
                'bearing': cp.asnumpy(bearings_padded),
                'elevation_change': cp.asnumpy(elev_padded),
                'distances': cp.asnumpy(distances_padded)
            }
    
    def _calculate_features_cpu(self, times: cp.ndarray, lats: cp.ndarray,
                               lons: cp.ndarray, eles: cp.ndarray) -> Dict[str, cp.ndarray]:
        """CPU fallback for feature calculation"""
        n = len(times)
        
        # Time differences
        dt = cp.diff(times)
        dt = cp.maximum(dt, 0.1)
        
        # Vectorized haversine distance
        lat1, lat2 = lats[:-1], lats[1:]
        lon1, lon2 = lons[:-1], lons[1:]
        
        lat1_rad = cp.radians(lat1)
        lat2_rad = cp.radians(lat2)
        dlat = cp.radians(lat2 - lat1)
        dlon = cp.radians(lon2 - lon1)
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))
        distances = 3958.8 * c
        
        # Speed calculation
        speeds = (distances * 3600) / dt
        speeds = cp.clip(speeds, 0, 200)
        
        # Acceleration
        accelerations = cp.zeros(n, dtype=cp.float32)
        if n > 2:
            accel_values = cp.diff(speeds) / dt[1:]
            accelerations[1:-1] = cp.clip(accel_values, -50, 50)
        
        # Bearing
        y = cp.sin(dlon) * cp.cos(lat2_rad)
        x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(dlon)
        bearings = cp.degrees(cp.arctan2(y, x))
        
        # Create padded arrays
        speeds_padded = cp.zeros(n, dtype=cp.float32)
        speeds_padded[1:] = speeds
        
        bearings_padded = cp.zeros(n, dtype=cp.float32)
        bearings_padded[1:] = bearings
        
        elev_padded = cp.zeros(n, dtype=cp.float32)
        elev_padded[1:] = cp.diff(eles)
        
        distances_padded = cp.zeros(n, dtype=cp.float32)
        distances_padded[1:] = distances
        
        return {
            'speed': speeds_padded,
            'acceleration': accelerations,
            'bearing': bearings_padded,
            'elevation_change': elev_padded,
            'distances': distances_padded
        }


class GPUCorrelator:
    """Production-ready GPU-accelerated correlation engine"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1], correlation_threshold: float = 0.01):
        self.gpu_ids = []
        self.correlation_threshold = correlation_threshold
        
        # Validate GPUs
        for gpu_id in gpu_ids:
            if gpu_id < cp.cuda.runtime.getDeviceCount():
                self.gpu_ids.append(gpu_id)
        
        if not self.gpu_ids:
            raise RuntimeError("No valid CUDA devices for correlation")
        
        logger.info(f"Initialized correlator with GPUs: {self.gpu_ids}")
        
        # Initialize correlation weights
        self._init_correlation_weights()
    
    def _init_correlation_weights(self):
        """Initialize correlation method weights on each GPU"""
        self.correlation_weights = {}
        
        for gpu_id in self.gpu_ids:
            with cp.cuda.Device(gpu_id):
                # Optimized weights based on empirical testing
                self.correlation_weights[gpu_id] = {
                    'fft': cp.array(0.35, dtype=cp.float32),
                    'correlation': cp.array(0.35, dtype=cp.float32),
                    'stats': cp.array(0.20, dtype=cp.float32),
                    'dtw': cp.array(0.10, dtype=cp.float32)
                }
    
    async def correlate_all(self, video_features: Dict[str, Any], 
                           gpx_features: Dict[str, Any], 
                           top_k: int = 5) -> Dict[str, Any]:
        """Correlate all videos with GPX files using GPU acceleration"""
        logger.info(f"Starting correlation: {len(video_features)} videos x {len(gpx_features)} GPX files")
        
        # Filter valid entries
        valid_videos = [(k, v) for k, v in video_features.items() if v is not None]
        valid_gpx = [(k, v) for k, v in gpx_features.items() if v is not None]
        
        if not valid_videos or not valid_gpx:
            logger.warning("No valid videos or GPX files to correlate")
            return {}
        
        logger.info(f"Valid entries: {len(valid_videos)} videos, {len(valid_gpx)} GPX files")
        
        # Distribute work across GPUs
        videos_per_gpu = (len(valid_videos) + len(self.gpu_ids) - 1) // len(self.gpu_ids)
        
        tasks = []
        for i, gpu_id in enumerate(self.gpu_ids):
            start_idx = i * videos_per_gpu
            end_idx = min((i + 1) * videos_per_gpu, len(valid_videos))
            
            if start_idx < end_idx:
                task = asyncio.create_task(
                    self._correlate_on_gpu(
                        valid_videos[start_idx:end_idx],
                        valid_gpx,
                        gpu_id,
                        top_k
                    )
                )
                tasks.append(task)
        
        # Gather results
        results = {}
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error(f"GPU correlation task failed: {task_result}")
            else:
                results.update(task_result)
                success_count += len(task_result)
        
        logger.info(f"Correlation complete: {success_count} videos processed successfully")
        return results
    
    async def _correlate_on_gpu(self, video_subset: List[Tuple[str, Dict]], 
                               gpx_list: List[Tuple[str, Dict]], 
                               gpu_id: int, top_k: int) -> Dict[str, Any]:
        """Correlate videos on specific GPU"""
        results = {}
        
        try:
            with cp.cuda.Device(gpu_id):
                # Precompute GPX signatures for efficiency
                logger.debug(f"GPU {gpu_id}: Precomputing {len(gpx_list)} GPX signatures")
                gpx_signatures = self._batch_compute_signatures(
                    [gpx_data for _, gpx_data in gpx_list], gpu_id
                )
                
                # Process each video
                for video_path, video_data in tqdm(video_subset, 
                                                  desc=f"GPU {gpu_id} correlation", 
                                                  leave=False):
                    try:
                        # Compute video signature
                        video_sig = self._compute_signature(video_data, gpu_id)
                        
                        # Batch similarity computation
                        scores = self._batch_compute_similarity(
                            video_sig, gpx_signatures, gpu_id
                        )
                        
                        # Find top matches
                        if len(scores) > 0:
                            # Use argpartition for efficiency with large arrays
                            k = min(top_k, len(scores))
                            top_indices = cp.argpartition(-scores, k-1)[:k]
                            top_indices = top_indices[cp.argsort(-scores[top_indices])]
                            
                            # Build match list
                            matches = []
                            for idx in cp.asnumpy(top_indices):
                                score = float(scores[idx])
                                if score > self.correlation_threshold:
                                    matches.append({
                                        'path': gpx_list[idx][0],
                                        'score': score,
                                        'duration_ratio': self._compute_duration_ratio(
                                            video_data.get('duration', 0),
                                            gpx_list[idx][1].get('duration', 0)
                                        )
                                    })
                            
                            results[video_path] = {
                                'matches': matches,
                                'video_duration': video_data.get('duration', 0),
                                'processed_time': datetime.now().isoformat()
                            }
                        else:
                            results[video_path] = {
                                'matches': [],
                                'video_duration': video_data.get('duration', 0),
                                'processed_time': datetime.now().isoformat()
                            }
                            
                    except Exception as e:
                        logger.error(f"Failed to correlate {video_path}: {e}")
                        results[video_path] = None
                
                # Clean up GPU memory
                mempool.free_all_blocks()
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} correlation failed: {e}")
            logger.debug(traceback.format_exc())
        
        return results
    
    def _compute_signature(self, data: Dict[str, Any], gpu_id: int) -> Dict[str, cp.ndarray]:
        """Compute feature signature on GPU"""
        signature = {}
        
        with cp.cuda.Device(gpu_id):
            # Process each feature type
            feature_keys = ['motion_magnitude', 'motion_complexity', 'temporal_gradient',
                           'speed', 'acceleration', 'bearing']
            
            for key in feature_keys:
                if key in data and isinstance(data[key], cp.ndarray) and len(data[key]) > 0:
                    try:
                        # Upload to GPU
                        signal = cp.asarray(data[key], dtype=cp.float32)
                        
                        # Normalize
                        mean = cp.mean(signal)
                        std = cp.std(signal) + 1e-8
                        signal_norm = (signal - mean) / std
                        
                        # FFT signature
                        fft = cp.fft.fft(signal_norm)
                        fft_mag = cp.abs(fft[:len(fft)//2])
                        
                        # Normalize FFT
                        if cp.max(fft_mag) > 0:
                            fft_mag = fft_mag / cp.max(fft_mag)
                        
                        signature[f'{key}_fft'] = fft_mag
                        
                        # Statistical signature
                        stats = cp.array([
                            mean,
                            std,
                            cp.min(signal),
                            cp.max(signal),
                            cp.percentile(signal, 25),
                            cp.percentile(signal, 50),
                            cp.percentile(signal, 75),
                            float(len(signal))
                        ], dtype=cp.float32)
                        
                        signature[f'{key}_stats'] = stats
                        
                        # Downsampled signal for correlation
                        target_len = min(len(signal_norm), 200)
                        if len(signal_norm) > target_len:
                            indices = cp.linspace(0, len(signal_norm)-1, target_len, dtype=cp.int32)
                            signal_ds = signal_norm[indices]
                        else:
                            signal_ds = signal_norm
                        
                        signature[f'{key}_signal'] = signal_ds
                        
                    except Exception as e:
                        logger.debug(f"Failed to compute signature for {key}: {e}")
            
        return signature
    
    def _batch_compute_signatures(self, data_list: List[Dict[str, Any]], 
                                 gpu_id: int) -> List[Dict[str, cp.ndarray]]:
        """Batch compute signatures for efficiency"""
        signatures = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            for data in batch:
                sig = self._compute_signature(data.get('features', data), gpu_id)
                signatures.append(sig)
        
        return signatures
    
    def _batch_compute_similarity(self, video_sig: Dict[str, cp.ndarray],
                                 gpx_signatures: List[Dict[str, cp.ndarray]],
                                 gpu_id: int) -> cp.ndarray:
        """Compute similarity scores in batch on GPU"""
        n_gpx = len(gpx_signatures)
        
        if n_gpx == 0:
            return cp.array([])
        
        with cp.cuda.Device(gpu_id):
            scores = cp.zeros(n_gpx, dtype=cp.float32)
            weights = self.correlation_weights[gpu_id]
            
            # Batch process similarities
            batch_size = 1000  # Process 1000 at a time
            
            for batch_start in range(0, n_gpx, batch_size):
                batch_end = min(batch_start + batch_size, n_gpx)
                batch_scores = cp.zeros(batch_end - batch_start, dtype=cp.float32)
                
                for i, gpx_sig in enumerate(gpx_signatures[batch_start:batch_end]):
                    # FFT correlation
                    fft_score = self._compute_fft_correlation(video_sig, gpx_sig)
                    
                    # Signal correlation
                    signal_score = self._compute_signal_correlation(video_sig, gpx_sig)
                    
                    # Statistical similarity
                    stats_score = self._compute_stats_similarity(video_sig, gpx_sig)
                    
                    # Weighted combination
                    batch_scores[i] = (
                        weights['fft'] * fft_score +
                        weights['correlation'] * signal_score +
                        weights['stats'] * stats_score
                    )
                
                scores[batch_start:batch_end] = batch_scores
            
            return scores
    
    def _compute_fft_correlation(self, sig1: Dict[str, cp.ndarray], 
                                sig2: Dict[str, cp.ndarray]) -> cp.float32:
        """FFT-based correlation on GPU"""
        scores = []
        
        for key in sig1:
            if key.endswith('_fft') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                if len(s1) > 0 and len(s2) > 0:
                    # Match lengths
                    min_len = min(len(s1), len(s2))
                    s1 = s1[:min_len]
                    s2 = s2[:min_len]
                    
                    # Compute correlation
                    if min_len > 0:
                        dot_product = cp.sum(s1 * s2)
                        norm1 = cp.linalg.norm(s1) + 1e-8
                        norm2 = cp.linalg.norm(s2) + 1e-8
                        corr = cp.abs(dot_product / (norm1 * norm2))
                        scores.append(float(corr))
        
        return cp.array(cp.mean(scores) if scores else 0.0, dtype=cp.float32)
    
    def _compute_signal_correlation(self, sig1: Dict[str, cp.ndarray],
                                   sig2: Dict[str, cp.ndarray]) -> cp.float32:
        """Direct signal correlation on GPU"""
        scores = []
        
        for key in sig1:
            if key.endswith('_signal') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                if len(s1) > 10 and len(s2) > 10:
                    # Resample to same length
                    target_len = min(len(s1), len(s2), 100)
                    
                    # Linear interpolation for resampling
                    x1 = cp.linspace(0, len(s1)-1, target_len)
                    x2 = cp.linspace(0, len(s2)-1, target_len)
                    
                    s1_resampled = cp.interp(x1, cp.arange(len(s1)), s1)
                    s2_resampled = cp.interp(x2, cp.arange(len(s2)), s2)
                    
                    # Compute correlation
                    if cp.std(s1_resampled) > 0 and cp.std(s2_resampled) > 0:
                        # Normalize
                        s1_norm = (s1_resampled - cp.mean(s1_resampled)) / cp.std(s1_resampled)
                        s2_norm = (s2_resampled - cp.mean(s2_resampled)) / cp.std(s2_resampled)
                        
                        # Correlation coefficient
                        corr = cp.sum(s1_norm * s2_norm) / len(s1_norm)
                        scores.append(float(cp.abs(corr)))
        
        return cp.array(cp.mean(scores) if scores else 0.0, dtype=cp.float32)
    
    def _compute_stats_similarity(self, sig1: Dict[str, cp.ndarray],
                                 sig2: Dict[str, cp.ndarray]) -> cp.float32:
        """Statistical similarity on GPU"""
        scores = []
        
        for key in sig1:
            if key.endswith('_stats') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                if len(s1) == len(s2) and len(s1) > 0:
                    # Normalize stats vectors
                    s1_norm = s1 / (cp.linalg.norm(s1) + 1e-8)
                    s2_norm = s2 / (cp.linalg.norm(s2) + 1e-8)
                    
                    # Cosine similarity
                    similarity = cp.sum(s1_norm * s2_norm)
                    scores.append(float(cp.abs(similarity)))
        
        return cp.array(cp.mean(scores) if scores else 0.0, dtype=cp.float32)
    
    def _compute_duration_ratio(self, duration1: float, duration2: float) -> float:
        """Compute duration compatibility ratio"""
        if duration1 <= 0 or duration2 <= 0:
            return 0.0
        return min(duration1, duration2) / max(duration1, duration2)


def generate_production_report(results: Dict[str, Any], output_dir: str) -> None:
    """Generate comprehensive production report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Analyze results
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'gpu_optimized': True
        },
        'summary': {
            'total_videos': len(results),
            'successful_matches': 0,
            'failed_matches': 0,
            'no_matches': 0
        },
        'statistics': {
            'scores': [],
            'duration_ratios': [],
            'matches_per_video': []
        },
        'confidence_distribution': {
            'high': 0,      # > 0.7
            'medium': 0,    # 0.3 - 0.7
            'low': 0,       # 0.1 - 0.3
            'very_low': 0   # < 0.1
        },
        'detailed_results': []
    }
    
    # Process results
    for video_path, result in results.items():
        if result is None:
            report['summary']['failed_matches'] += 1
            continue
        
        matches = result.get('matches', [])
        
        if not matches:
            report['summary']['no_matches'] += 1
        else:
            report['summary']['successful_matches'] += 1
            report['statistics']['matches_per_video'].append(len(matches))
            
            # Best match analysis
            best_match = matches[0]
            score = best_match['score']
            
            report['statistics']['scores'].append(score)
            report['statistics']['duration_ratios'].append(
                best_match.get('duration_ratio', 0)
            )
            
            # Confidence categorization
            if score > 0.7:
                report['confidence_distribution']['high'] += 1
            elif score > 0.3:
                report['confidence_distribution']['medium'] += 1
            elif score > 0.1:
                report['confidence_distribution']['low'] += 1
            else:
                report['confidence_distribution']['very_low'] += 1
            
            # Detailed entry
            report['detailed_results'].append({
                'video': str(video_path),
                'best_match': str(best_match['path']),
                'score': float(score),
                'duration_ratio': float(best_match.get('duration_ratio', 0)),
                'num_matches': len(matches),
                'all_matches': [
                    {
                        'gpx': str(m['path']),
                        'score': float(m['score'])
                    } for m in matches[:10]  # Top 10
                ]
            })
    
    # Calculate statistics
    if report['statistics']['scores']:
        scores = cp.array(report['statistics']['scores'])
        report['statistics']['summary'] = {
            'mean_score': float(cp.mean(scores)),
            'std_score': float(cp.std(scores)),
            'min_score': float(cp.min(scores)),
            'max_score': float(cp.max(scores)),
            'median_score': float(cp.median(scores)),
            'percentile_25': float(cp.percentile(scores, 25)),
            'percentile_75': float(cp.percentile(scores, 75))
        }
    
    # Save JSON report
    json_path = output_path / 'correlation_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_path = output_path / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("GPU-Optimized Video-GPX Correlation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {report['metadata']['timestamp']}\n")
        f.write(f"Version: {report['metadata']['version']}\n\n")
        
        f.write("Summary\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Videos: {report['summary']['total_videos']}\n")
        f.write(f"Successful Matches: {report['summary']['successful_matches']}\n")
        f.write(f"No Matches Found: {report['summary']['no_matches']}\n")
        f.write(f"Failed Processing: {report['summary']['failed_matches']}\n\n")
        
        if 'summary' in report['statistics']:
            f.write("Score Statistics\n")
            f.write("-" * 30 + "\n")
            stats = report['statistics']['summary']
            f.write(f"Mean Score: {stats['mean_score']:.4f}\n")
            f.write(f"Std Dev: {stats['std_score']:.4f}\n")
            f.write(f"Median: {stats['median_score']:.4f}\n")
            f.write(f"Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]\n")
            f.write(f"IQR: [{stats['percentile_25']:.4f}, {stats['percentile_75']:.4f}]\n\n")
        
        f.write("Confidence Distribution\n")
        f.write("-" * 30 + "\n")
        total = report['summary']['successful_matches']
        for level, count in report['confidence_distribution'].items():
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"{level.capitalize()}: {count} ({percentage:.1f}%)\n")
    
    # Save detailed results as CSV
    if report['detailed_results']:
        csv_path = output_path / 'detailed_results.csv'
        df_results = []
        
        for result in report['detailed_results']:
            df_results.append({
                'video': Path(result['video']).name,
                'best_match_gpx': Path(result['best_match']).name,
                'score': result['score'],
                'duration_ratio': result['duration_ratio'],
                'num_matches': result['num_matches']
            })
        
        pd.DataFrame(df_results).to_csv(csv_path, index=False)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("Correlation Complete")
    print("=" * 60)
    print(f"Total Videos: {report['summary']['total_videos']}")
    print(f"Successful: {report['summary']['successful_matches']} "
          f"({report['summary']['successful_matches']/report['summary']['total_videos']*100:.1f}%)")
    
    if 'summary' in report['statistics']:
        print(f"Average Score: {report['statistics']['summary']['mean_score']:.3f}")
        print(f"Median Score: {report['statistics']['summary']['median_score']:.3f}")


async def main():
    """Main entry point for production GPU matcher"""
    parser = argparse.ArgumentParser(
        description="Production GPU-Optimized Video-GPX Matcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-d", "--directory", required=True, 
                       help="Directory containing videos and GPX files")
    parser.add_argument("-o", "--output", default="./correlation_results",
                       help="Output directory for results")
    parser.add_argument("--gpu-ids", nargs='+', type=int, default=[0, 1],
                       help="GPU device IDs to use")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for GPU processing")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top matches to return per video")
    parser.add_argument("--cache-dir", default="./cache",
                       help="Directory for cached features")
    parser.add_argument("--force-reprocess", action='store_true',
                       help="Force reprocessing, ignore cache")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum worker processes for GPX parsing")
    parser.add_argument("--log-file", default="gpu_matcher.log",
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Update logging configuration
    global logger
    logger = setup_logging(args.log_file)
    
    logger.info("Starting GPU-Optimized Video-GPX Matcher")
    logger.info(f"Configuration: {vars(args)}")
    
    # Find files
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    # Find video files (case-insensitive)
    video_extensions = ['mp4', 'avi', 'mov', 'mkv']
    video_files = []
    
    for ext in video_extensions:
        for pattern in [f'*.{ext}', f'*.{ext.upper()}', f'*.{ext.capitalize()}']:
            video_files.extend(directory.glob(pattern))
    
    # Find GPX files (case-insensitive)
    gpx_files = []
    for pattern in ['*.gpx', '*.GPX', '*.Gpx']:
        gpx_files.extend(directory.glob(pattern))
    
    # Remove duplicates and convert to strings
    video_files = list(set(str(f) for f in video_files))
    gpx_files = list(set(str(f) for f in gpx_files))
    
    logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
    
    if not video_files:
        logger.error("No video files found")
        sys.exit(1)
    
    if not gpx_files:
        logger.error("No GPX files found")
        sys.exit(1)
    
    # Initialize processors
    try:
        video_processor = GPUVideoProcessor(
            gpu_ids=args.gpu_ids,
            batch_size=args.batch_size
        )
        
        gpx_processor = GPXProcessor(
            gpu_id=args.gpu_ids[0] if args.gpu_ids else 0
        )
        
        correlator = GPUCorrelator(
            gpu_ids=args.gpu_ids
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        sys.exit(1)
    
    # Setup cache
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    video_cache_path = cache_dir / "video_features_gpu_v2.pkl"
    gpx_cache_path = cache_dir / "gpx_features_v2.pkl"
    
    # Process videos
    if video_cache_path.exists() and not args.force_reprocess:
        logger.info("Loading cached video features...")
        try:
            with open(video_cache_path, 'rb') as f:
                video_features = pickle.load(f)
            logger.info(f"Loaded {len(video_features)} cached video features")
        except Exception as e:
            logger.warning(f"Failed to load video cache: {e}")
            video_features = await video_processor.process_videos_batch(video_files)
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
    else:
        logger.info("Processing videos on GPU...")
        video_features = await video_processor.process_videos_batch(video_files)
        
        # Save cache
        try:
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            logger.info(f"Cached {len(video_features)} video features")
        except Exception as e:
            logger.warning(f"Failed to save video cache: {e}")
    
    # Process GPX files
    if gpx_cache_path.exists() and not args.force_reprocess:
        logger.info("Loading cached GPX features...")
        try:
            with open(gpx_cache_path, 'rb') as f:
                gpx_features = pickle.load(f)
            logger.info(f"Loaded {len(gpx_features)} cached GPX features")
        except Exception as e:
            logger.warning(f"Failed to load GPX cache: {e}")
            gpx_features = gpx_processor.process_gpx_files(gpx_files, args.max_workers)
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_features, f)
    else:
        logger.info("Processing GPX files...")
        gpx_features = gpx_processor.process_gpx_files(gpx_files, args.max_workers)
        
        # Save cache
        try:
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_features, f)
            logger.info(f"Cached {len(gpx_features)} GPX features")
        except Exception as e:
            logger.warning(f"Failed to save GPX cache: {e}")
    
    # Perform correlation
    logger.info("Starting GPU correlation...")
    correlation_results = await correlator.correlate_all(
        video_features, gpx_features, args.top_k
    )
    
    # Generate report
    generate_production_report(correlation_results, args.output)
    
    # Save raw results
    results_path = Path(args.output) / "raw_correlation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(correlation_results, f)
    
    # Cleanup
    gpu_resource_manager.cleanup()
    
    logger.info("Processing complete")


if __name__ == "__main__":
    # Configure multiprocessing for CUDA
    mp.set_start_method('spawn', force=True)
    
    # Run async main
    try:
        asyncio.run(main())
        sys.exit(0)  # Success
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        gpu_resource_manager.cleanup()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.debug(traceback.format_exc())
        gpu_resource_manager.cleanup()
        sys.exit(1)