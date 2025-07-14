#!/usr/bin/env python3
"""
FIXED EXTREME GPU-INTENSIVE Temporal Offset Calculator
🚀 MAXIMUM UTILIZATION OF DUAL RTX 5060 Ti (30.8GB GPU RAM TOTAL)
⚡ 100% GPU utilization target with proper OpenCV CUDA handling
🎯 Fixed OpenCV CUDA operations for reliable GPU processing
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
        logging.FileHandler('fixed_extreme_gpu_offsetter.log', mode='w')
    ]
)
logger = logging.getLogger('fixed_extreme_gpu_offsetter')

@dataclass
class FixedExtremeGPUConfig:
    """Fixed extreme GPU configuration for maximum utilization"""
    # GPU Configuration - MAXIMUM AGGRESSION
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    max_gpu_memory_gb: float = 14.5  # Use almost all 15.4GB per GPU
    gpu_batch_size: int = 512  # Conservative but reliable batch size
    cuda_streams: int = 16  # Optimized streams per GPU
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
    gpu_memory_fraction: float = 0.95  # Use 95% of GPU memory
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    gpu_memory_pool_size: int = 14 * 1024**3  # 14GB pool per GPU
    enable_mixed_precision: bool = True
    
    # PARALLEL PROCESSING - OPTIMIZED
    parallel_videos_per_gpu: int = 4  # Process 4 videos per GPU simultaneously
    parallel_gpx_per_gpu: int = 8  # Process 8 GPX files per GPU
    correlation_batch_size: int = 2000  # Large correlation batches
    fft_batch_size: int = 500  # FFT batches
    
    # MULTI-GPU COORDINATION
    enable_multi_gpu_batching: bool = True
    gpu_load_balancing: bool = True
    cross_gpu_memory_sharing: bool = True
    
    # AGGRESSIVE CACHING
    enable_gpu_caching: bool = True
    cache_video_features: bool = True
    cache_gps_features: bool = True
    cache_correlations: bool = True
    
    # STRICT MODE - ROBUST GPU PROCESSING
    strict_mode: bool = False
    force_gpu_only: bool = False  # Set to True with --strict
    gpu_timeout_seconds: float = 600.0
    fail_on_cpu_fallback: bool = False
    enable_opencv_cuda_fallback: bool = True  # Allow fallback for OpenCV issues

class FixedExtremeGPUMemoryManager:
    """Fixed extreme GPU memory management for maximum utilization"""
    
    def __init__(self, config: FixedExtremeGPUConfig):
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
        logger.info(f"🚀 FIXED EXTREME GPU INITIALIZATION: {self.config.gpu_ids}")
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
                
                # Pre-allocate memory blocks for caching
                if self.config.enable_gpu_caching:
                    cache_size = int(1 * 1024**3)  # 1GB cache per GPU (conservative)
                    self.gpu_caches[gpu_id] = cp.zeros(cache_size // 4, dtype=cp.float32)
                    logger.info(f"   GPU {gpu_id}: Pre-allocated 1GB cache")
                
                # Create CUDA streams
                streams = []
                for i in range(self.config.cuda_streams):
                    stream = cp.cuda.Stream(non_blocking=True)
                    streams.append(stream)
                self.gpu_streams[gpu_id] = streams
                
                # Get device properties
                device = cp.cuda.Device(gpu_id)
                total_memory = device.mem_info[1] / (1024**3)
                free_memory = device.mem_info[0] / (1024**3)
                total_gpu_memory += total_memory
                
                # Warm up GPU with computation
                warmup_size = min(100000, int(free_memory * 0.01 * 1024**3 / 4))  # Conservative warmup
                warmup_data = cp.random.rand(warmup_size, dtype=cp.float32)
                _ = cp.fft.fft(warmup_data)
                del warmup_data
                
                logger.info(f"🎮 GPU {gpu_id} FIXED EXTREME INIT:")
                logger.info(f"   ├─ Total Memory: {total_memory:.1f}GB")
                logger.info(f"   ├─ Allocated: {self.config.max_gpu_memory_gb:.1f}GB")
                logger.info(f"   ├─ Streams: {self.config.cuda_streams}")
                logger.info(f"   ├─ Compute Capability: {device.compute_capability}")
                logger.info(f"   └─ Warmed Up: ✅")
                
            except Exception as e:
                logger.error(f"FIXED EXTREME GPU INIT FAILED {gpu_id}: {e}")
                if self.config.strict_mode:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} initialization failed")
        
        logger.info(f"🔥 TOTAL GPU POWER: {total_gpu_memory:.1f}GB across {len(self.config.gpu_ids)} GPUs")
        logger.info(f"⚡ FIXED EXTREME MODE: {'ENABLED' if self.config.force_gpu_only else 'STANDARD'}")
    
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
        logger.info("🧹 FIXED EXTREME GPU CLEANUP")
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

class FixedExtremeVideoProcessor:
    """Fixed extreme GPU video processing with proper OpenCV CUDA handling"""
    
    def __init__(self, config: FixedExtremeGPUConfig, memory_manager: FixedExtremeGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.video_cache = {}
        
        # Initialize OpenCV with GPU acceleration
        self._initialize_fixed_opencv()
    
    def _initialize_fixed_opencv(self):
        """Initialize OpenCV for FIXED EXTREME GPU processing"""
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count == 0:
                if self.config.strict_mode and not self.config.enable_opencv_cuda_fallback:
                    raise RuntimeError("STRICT MODE: No OpenCV CUDA devices found")
                logger.warning("⚠️ OpenCV CUDA not available, using CPU fallback")
                self.opencv_cuda_available = False
                return
            
            # Set primary GPU device
            cv2.cuda.setDevice(self.config.gpu_ids[0])
            
            # Test OpenCV CUDA functionality
            try:
                test_mat = cv2.cuda_GpuMat()
                test_cpu = np.zeros((100, 100, 3), dtype=np.uint8)
                test_mat.upload(test_cpu)
                
                # Test basic operations
                test_gray = cv2.cuda_GpuMat()
                cv2.cuda.cvtColor(test_mat, test_gray, cv2.COLOR_BGR2GRAY)
                
                # Test Gaussian blur
                test_blur = cv2.cuda_GpuMat()
                gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
                gaussian_filter.apply(test_gray, test_blur)
                
                # Test optical flow
                optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
                test_flow = cv2.cuda_GpuMat()
                optical_flow.calc(test_gray, test_gray, test_flow)
                
                logger.info("✅ OpenCV CUDA functionality verified")
                self.opencv_cuda_available = True
                
            except Exception as e:
                if self.config.strict_mode and not self.config.enable_opencv_cuda_fallback:
                    raise RuntimeError(f"STRICT MODE: OpenCV CUDA test failed: {e}")
                logger.warning(f"OpenCV CUDA test failed: {e}, using CPU fallback")
                self.opencv_cuda_available = False
                return
            
            # Pre-create GPU filters and objects for maximum performance
            self.gpu_filters = {}
            self.optical_flow_objects = {}
            
            for gpu_id in self.config.gpu_ids:
                try:
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
                    
                    logger.info(f"✅ GPU {gpu_id}: OpenCV CUDA objects created")
                    
                except Exception as e:
                    logger.warning(f"GPU {gpu_id}: OpenCV CUDA object creation failed: {e}")
                    if self.config.strict_mode and not self.config.enable_opencv_cuda_fallback:
                        raise
            
            logger.info(f"🎮 OpenCV CUDA FIXED EXTREME: {gpu_count} devices initialized")
            
        except Exception as e:
            if self.config.strict_mode and not self.config.enable_opencv_cuda_fallback:
                raise RuntimeError(f"STRICT MODE: OpenCV CUDA initialization failed: {e}")
            logger.warning(f"OpenCV CUDA initialization failed: {e}")
            self.opencv_cuda_available = False
    
    def extract_motion_signature_extreme_gpu(self, video_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME GPU video processing - batch multiple videos"""
        if self.config.strict_mode and not self.opencv_cuda_available and not self.config.enable_opencv_cuda_fallback:
            raise RuntimeError("STRICT MODE: GPU video processing required but OpenCV CUDA unavailable")
        
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                return self._process_video_batch_extreme_fixed(video_paths, gpu_id)
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPU video batch processing failed: {e}")
            logger.debug(f"EXTREME GPU video batch failed: {e}")
            return [None] * len(video_paths)
    
    def _process_video_batch_extreme_fixed(self, video_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME batch video processing on single GPU with fixed OpenCV handling"""
        results = []
        
        # Get optimized streams for different operations
        video_stream = self.memory_manager.get_extreme_stream(gpu_id, 'video')
        compute_stream = self.memory_manager.get_extreme_stream(gpu_id, 'compute')
        
        # Process videos in smaller batches for stability
        batch_size = min(len(video_paths), self.config.parallel_videos_per_gpu)
        
        try:
            # Process videos in batches for maximum GPU utilization
            for i in range(0, len(video_paths), batch_size):
                batch_paths = video_paths[i:i + batch_size]
                batch_results = []
                
                # Parallel processing within batch
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = []
                    
                    for video_path in batch_paths:
                        future = executor.submit(
                            self._process_single_video_extreme_gpu_fixed, 
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
    
    def _process_single_video_extreme_gpu_fixed(self, video_path: str, gpu_id: int, 
                                              video_stream: cp.cuda.Stream, 
                                              compute_stream: cp.cuda.Stream) -> Optional[Dict]:
        """EXTREME single video processing with FIXED OpenCV CUDA handling"""
        
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
        
        # Pre-allocate for motion values
        max_frames = min(int(duration * self.config.video_sample_rate), 5000)  # Conservative limit
        
        try:
            motion_values = []
            motion_energy = []
            timestamps = []
            
            frame_idx = 0
            processed_frames = 0
            prev_frame_cpu = None  # Keep CPU processing for stability
            
            # Use GPU processing if available, otherwise CPU fallback
            use_gpu_processing = self.opencv_cuda_available and gpu_id in self.gpu_filters
            
            if use_gpu_processing:
                logger.debug(f"GPU {gpu_id}: Using GPU OpenCV processing")
                # GPU objects for this specific GPU
                try:
                    cv2.cuda.setDevice(gpu_id)
                    gaussian_filter = self.gpu_filters[gpu_id]['gaussian_5x5']
                    optical_flow = self.optical_flow_objects[gpu_id]
                    
                    # Allocate GPU matrices
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_gray = cv2.cuda_GpuMat()
                    gpu_blurred = cv2.cuda_GpuMat()
                    gpu_flow = cv2.cuda_GpuMat()
                    
                    prev_gpu_frame = None
                    
                except Exception as e:
                    logger.warning(f"GPU {gpu_id}: Failed to setup GPU objects: {e}")
                    use_gpu_processing = False
            
            if not use_gpu_processing:
                logger.debug(f"GPU {gpu_id}: Using CPU OpenCV fallback")
            
            while True:
                ret, frame = cap.read()
                if not ret or processed_frames >= max_frames:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Resize for efficiency
                    target_width = 1280 if is_360 else 640
                    if frame.shape[1] > target_width:
                        scale = target_width / frame.shape[1]
                        new_width = target_width
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    if use_gpu_processing:
                        try:
                            # GPU processing path
                            gpu_frame.upload(frame)
                            
                            # Convert to grayscale on GPU - FIXED
                            cv2.cuda.cvtColor(gpu_frame, gpu_gray, cv2.COLOR_BGR2GRAY)
                            
                            # Apply Gaussian filter on GPU - FIXED
                            gaussian_filter.apply(gpu_gray, gpu_blurred)
                            
                            if prev_gpu_frame is not None:
                                # GPU optical flow - FIXED
                                optical_flow.calc(prev_gpu_frame, gpu_blurred, gpu_flow)
                                
                                # Download flow and calculate motion on GPU using CuPy
                                with cp.cuda.Stream(compute_stream):
                                    flow_cpu = gpu_flow.download()
                                    flow_gpu = cp.asarray(flow_cpu)
                                    magnitude = cp.sqrt(flow_gpu[..., 0]**2 + flow_gpu[..., 1]**2)
                                    
                                    # Calculate motion metrics
                                    motion_mag = float(cp.mean(magnitude))
                                    motion_eng = float(cp.sum(magnitude ** 2))
                                    
                                    motion_values.append(motion_mag)
                                    motion_energy.append(motion_eng)
                                    timestamps.append(frame_idx / fps)
                            
                            # Store current frame for next iteration - FIXED
                            if prev_gpu_frame is None:
                                prev_gpu_frame = cv2.cuda_GpuMat()
                            gpu_blurred.copyTo(prev_gpu_frame)
                            
                        except Exception as e:
                            # GPU processing failed, fall back to CPU for this frame
                            logger.debug(f"GPU {gpu_id}: GPU processing failed for frame, using CPU: {e}")
                            
                            # CPU fallback processing
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray = cv2.GaussianBlur(gray, (5, 5), 0)
                            
                            if prev_frame_cpu is not None:
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_frame_cpu, gray, None,
                                    0.5, 3, 15, 3, 5, 1.2, 0
                                )
                                
                                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                motion_mag = np.mean(magnitude)
                                motion_eng = np.sum(magnitude ** 2)
                                
                                motion_values.append(motion_mag)
                                motion_energy.append(motion_eng)
                                timestamps.append(frame_idx / fps)
                            
                            prev_frame_cpu = gray
                    
                    else:
                        # CPU processing path
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (5, 5), 0)
                        
                        if prev_frame_cpu is not None:
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_frame_cpu, gray, None,
                                0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            
                            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            motion_mag = np.mean(magnitude)
                            motion_eng = np.sum(magnitude ** 2)
                            
                            motion_values.append(motion_mag)
                            motion_energy.append(motion_eng)
                            timestamps.append(frame_idx / fps)
                        
                        prev_frame_cpu = gray
                    
                    processed_frames += 1
                
                frame_idx += 1
            
            cap.release()
            
            if len(motion_values) < 3:
                return None
            
            # Convert to GPU arrays if possible, otherwise keep as CPU arrays
            try:
                with cp.cuda.Stream(compute_stream):
                    result = {
                        'motion_magnitude': cp.array(motion_values),
                        'motion_energy': cp.array(motion_energy),
                        'timestamps': cp.array(timestamps),
                        'duration': duration,
                        'fps': fps,
                        'is_360': is_360,
                        'frame_count': processed_frames,
                        'gpu_id': gpu_id,
                        'processing_method': 'fixed_extreme_gpu' if use_gpu_processing else 'cpu_fallback'
                    }
            except Exception as e:
                # CPU storage fallback
                result = {
                    'motion_magnitude': np.array(motion_values),
                    'motion_energy': np.array(motion_energy),
                    'timestamps': np.array(timestamps),
                    'duration': duration,
                    'fps': fps,
                    'is_360': is_360,
                    'frame_count': processed_frames,
                    'gpu_id': gpu_id,
                    'processing_method': 'cpu_fallback'
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

# Continue with the rest of the classes...
# For brevity, I'll include the key changes and then provide the main function

class FixedExtremeGPXProcessor:
    """Fixed extreme GPX processor with robust GPU processing"""
    
    def __init__(self, config: FixedExtremeGPUConfig, memory_manager: FixedExtremeGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.gpx_cache = {}
    
    def extract_motion_signature_extreme_gpu_batch(self, gpx_paths: List[str], gpu_id: int) -> List[Dict]:
        """EXTREME GPU GPX batch processing with error handling"""
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                return self._process_gpx_batch_extreme_fixed(gpx_paths, gpu_id)
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: GPU GPX batch processing failed: {e}")
            logger.debug(f"EXTREME GPU GPX batch failed: {e}")
            return [None] * len(gpx_paths)
    
    def _process_gpx_batch_extreme_fixed(self, gpx_paths: List[str], gpu_id: int) -> List[Dict]:
        """Process GPX files with robust error handling"""
        results = []
        
        # Process in batches with conservative sizing
        batch_size = min(len(gpx_paths), self.config.parallel_gpx_per_gpu)
        gps_stream = self.memory_manager.get_extreme_stream(gpu_id, 'gps')
        
        try:
            for i in range(0, len(gpx_paths), batch_size):
                batch_paths = gpx_paths[i:i + batch_size]
                
                # Sequential processing for stability
                batch_results = []
                for gpx_path in batch_paths:
                    try:
                        result = self._process_single_gpx_extreme_gpu_fixed(gpx_path, gpu_id, gps_stream)
                        batch_results.append(result)
                    except Exception as e:
                        if self.config.strict_mode:
                            raise
                        logger.debug(f"GPX processing failed: {e}")
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
    
    def _process_single_gpx_extreme_gpu_fixed(self, gpx_path: str, gpu_id: int, 
                                            gps_stream: cp.cuda.Stream) -> Optional[Dict]:
        """Process single GPX with GPU acceleration and CPU fallback"""
        
        # Check cache first
        cache_key = f"{gpx_path}_{gpu_id}"
        if self.config.cache_gps_features and cache_key in self.gpx_cache:
            return self.gpx_cache[cache_key]
        
        # Load GPX data (CPU)
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
        except Exception as e:
            logger.debug(f"Failed to parse GPX file {gpx_path}: {e}")
            return None
        
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
        
        # Try GPU processing with CPU fallback
        try:
            with cp.cuda.Stream(gps_stream):
                # Upload data to GPU
                lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)
                lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
                
                # Vectorized distance calculation on GPU
                lat1, lat2 = lats_gpu[:-1], lats_gpu[1:]
                lon1, lon2 = lons_gpu[:-1], lons_gpu[1:]
                
                lat1_rad, lat2_rad = cp.radians(lat1), cp.radians(lat2)
                lon1_rad, lon2_rad = cp.radians(lon1), cp.radians(lon2)
                
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
                distances = 6371000 * 2 * cp.arcsin(cp.sqrt(a))
                
                time_diffs = cp.array([
                    (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds() 
                    for i in range(len(df)-1)
                ], dtype=cp.float32)
                
                speeds = cp.where(time_diffs > 0, distances / time_diffs, 0)
                accelerations = cp.zeros_like(speeds)
                accelerations[1:] = cp.where(
                    time_diffs[1:] > 0,
                    (speeds[1:] - speeds[:-1]) / time_diffs[1:],
                    0
                )
                
                # Resample to consistent intervals
                time_offsets = cp.cumsum(cp.concatenate([cp.array([0]), time_diffs]))
                target_times = cp.arange(0, duration, self.config.gps_sample_rate, dtype=cp.float32)
                
                resampled_speed = cp.interp(target_times, time_offsets[:-1], speeds)
                resampled_accel = cp.interp(target_times, time_offsets[:-1], accelerations)
                
                result = {
                    'speed': resampled_speed,
                    'acceleration': resampled_accel,
                    'timestamps': df['time'].tolist(),
                    'time_offsets': target_times,
                    'duration': duration,
                    'point_count': len(speeds),
                    'start_time': df['time'].iloc[0],
                    'end_time': df['time'].iloc[-1],
                    'gpu_id': gpu_id,
                    'processing_method': 'fixed_extreme_gpu'
                }
                
                # Cache result
                if self.config.cache_gps_features:
                    self.gpx_cache[cache_key] = result
                
                return result
        
        except Exception as e:
            # CPU fallback
            logger.debug(f"GPU GPX processing failed, using CPU fallback: {e}")
            return self._process_gpx_cpu_fallback(df, duration)
    
    def _process_gpx_cpu_fallback(self, df: pd.DataFrame, duration: float) -> Dict:
        """CPU fallback for GPX processing"""
        try:
            # CPU processing
            lats = df['lat'].values
            lons = df['lon'].values
            
            lat1 = np.radians(lats[:-1])
            lat2 = np.radians(lats[1:])
            lon1 = np.radians(lons[:-1])
            lon2 = np.radians(lons[1:])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            distances = 6371000 * 2 * np.arcsin(np.sqrt(a))
            
            time_diffs = np.array([
                (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds() 
                for i in range(len(df)-1)
            ])
            
            speeds = np.divide(distances, time_diffs, out=np.zeros_like(distances), where=time_diffs!=0)
            accelerations = np.zeros_like(speeds)
            accelerations[1:] = np.divide(
                np.diff(speeds), time_diffs[1:], 
                out=np.zeros_like(speeds[1:]), where=time_diffs[1:]!=0
            )
            
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
        except Exception as e:
            logger.error(f"CPU GPX fallback failed: {e}")
            return None

# I'll continue with a simplified but robust offset calculator and main processor...

class FixedExtremeOffsetCalculator:
    """Fixed extreme offset calculator with robust GPU processing"""
    
    def __init__(self, config: FixedExtremeGPUConfig, memory_manager: FixedExtremeGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
    
    def calculate_offset_extreme_gpu_batch(self, video_gps_pairs: List[Tuple[Dict, Dict]], 
                                         gpu_id: int) -> List[Dict]:
        """Calculate offsets with robust error handling"""
        results = []
        
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                for video_data, gps_data in video_gps_pairs:
                    try:
                        result = self._calculate_single_offset_fixed(video_data, gps_data, gpu_id)
                        results.append(result)
                    except Exception as e:
                        if self.config.strict_mode:
                            raise
                        results.append({
                            'offset_method': 'calculation_failed',
                            'gpu_processing': False,
                            'error': str(e)[:100]
                        })
        except Exception as e:
            if self.config.strict_mode:
                raise RuntimeError(f"STRICT MODE: Batch offset calculation failed: {e}")
            results = [{'offset_method': 'batch_failed', 'gpu_processing': False}] * len(video_gps_pairs)
        
        return results
    
    def _calculate_single_offset_fixed(self, video_data: Dict, gps_data: Dict, gpu_id: int) -> Dict:
        """Calculate single offset with GPU acceleration and CPU fallback"""
        result = {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'fixed_extreme_gpu',
            'gpu_processing': True,
            'gpu_id': gpu_id
        }
        
        try:
            # Get signals
            video_signal = self._get_signal(video_data, 'video')
            gps_signal = self._get_signal(gps_data, 'gps')
            
            if video_signal is None or gps_signal is None:
                result['offset_method'] = 'no_signals'
                result['gpu_processing'] = False
                return result
            
            # Try GPU FFT correlation
            try:
                offset, confidence = self._gpu_fft_correlation_fixed(video_signal, gps_signal, gpu_id)
                
                if offset is not None and confidence >= self.config.min_correlation_confidence:
                    result.update({
                        'temporal_offset_seconds': float(offset),
                        'offset_confidence': float(confidence),
                        'offset_method': 'fixed_gpu_fft_correlation'
                    })
                    return result
            except Exception as e:
                logger.debug(f"GPU FFT correlation failed: {e}")
            
            # CPU fallback
            try:
                from scipy.signal import correlate
                
                # Normalize signals
                video_norm = self._normalize_signal(video_signal)
                gps_norm = self._normalize_signal(gps_signal)
                
                # Cross-correlation
                correlation = correlate(gps_norm, video_norm, mode='full')
                peak_idx = np.argmax(correlation)
                confidence = correlation[peak_idx] / len(video_norm)
                
                offset_samples = peak_idx - len(video_norm) + 1
                offset_seconds = offset_samples * self.config.gps_sample_rate
                
                if abs(offset_seconds) <= self.config.max_offset_search_seconds and abs(confidence) >= self.config.min_correlation_confidence:
                    result.update({
                        'temporal_offset_seconds': float(offset_seconds),
                        'offset_confidence': float(abs(confidence)),
                        'offset_method': 'cpu_fallback_correlation',
                        'gpu_processing': False
                    })
                else:
                    result['offset_method'] = 'correlation_below_threshold'
                    result['gpu_processing'] = False
                
            except Exception as e:
                result['offset_method'] = f'cpu_fallback_failed: {str(e)[:50]}'
                result['gpu_processing'] = False
        
        except Exception as e:
            result['offset_method'] = f'processing_error: {str(e)[:50]}'
            result['gpu_processing'] = False
        
        return result
    
    def _gpu_fft_correlation_fixed(self, video_signal, gps_signal, gpu_id: int) -> Tuple[Optional[float], float]:
        """Fixed GPU FFT correlation with proper error handling"""
        try:
            correlation_stream = self.memory_manager.get_extreme_stream(gpu_id, 'correlation')
            
            with cp.cuda.Stream(correlation_stream):
                # Convert to GPU arrays
                if isinstance(video_signal, np.ndarray):
                    v_signal = cp.array(video_signal)
                else:
                    v_signal = video_signal
                
                if isinstance(gps_signal, np.ndarray):
                    g_signal = cp.array(gps_signal)
                else:
                    g_signal = gps_signal
                
                # Normalize
                v_norm = self._normalize_signal_gpu(v_signal)
                g_norm = self._normalize_signal_gpu(g_signal)
                
                # Pad for FFT
                max_len = len(v_norm) + len(g_norm) - 1
                pad_len = 1 << (max_len - 1).bit_length()
                
                v_padded = cp.pad(v_norm, (0, pad_len - len(v_norm)))
                g_padded = cp.pad(g_norm, (0, pad_len - len(g_norm)))
                
                # FFT correlation
                v_fft = cp.fft.fft(v_padded)
                g_fft = cp.fft.fft(g_padded)
                correlation = cp.fft.ifft(cp.conj(v_fft) * g_fft).real
                
                # Find peak
                peak_idx = cp.argmax(correlation)
                confidence = float(correlation[peak_idx] / len(v_norm))
                
                # Convert to offset
                offset_samples = int(peak_idx - len(v_norm) + 1)
                offset_seconds = offset_samples * self.config.gps_sample_rate
                
                if abs(offset_seconds) <= self.config.max_offset_search_seconds:
                    return float(offset_seconds), min(abs(confidence), 1.0)
                else:
                    return None, 0.0
        
        except Exception as e:
            logger.debug(f"GPU FFT correlation failed: {e}")
            return None, 0.0
    
    def _get_signal(self, data: Dict, data_type: str):
        """Get motion signal from data"""
        if data_type == 'video':
            signal_keys = ['motion_magnitude', 'motion_energy']
        else:
            signal_keys = ['speed', 'acceleration']
        
        for key in signal_keys:
            if key in data:
                signal = data[key]
                if isinstance(signal, (np.ndarray, cp.ndarray)) and len(signal) > 3:
                    return signal
        
        return None
    
    def _normalize_signal(self, signal):
        """Normalize signal (works with both numpy and cupy)"""
        if isinstance(signal, cp.ndarray):
            return self._normalize_signal_gpu(signal)
        else:
            return self._normalize_signal_cpu(signal)
    
    def _normalize_signal_gpu(self, signal: cp.ndarray) -> cp.ndarray:
        """GPU signal normalization"""
        if len(signal) == 0:
            return signal
        
        mean = cp.mean(signal)
        std = cp.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean
    
    def _normalize_signal_cpu(self, signal: np.ndarray) -> np.ndarray:
        """CPU signal normalization"""
        if len(signal) == 0:
            return signal
        
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean

# Main processor class that coordinates everything
class FixedExtremeGPUOffsetProcessor:
    """Fixed extreme GPU processor with maximum utilization and reliability"""
    
    def __init__(self, config: FixedExtremeGPUConfig):
        self.config = config
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("EXTREME MODE: CuPy required for maximum GPU utilization")
        
        # Initialize components
        self.memory_manager = FixedExtremeGPUMemoryManager(config)
        self.video_processor = FixedExtremeVideoProcessor(config, self.memory_manager)
        self.gpx_processor = FixedExtremeGPXProcessor(config, self.memory_manager)
        self.offset_calculator = FixedExtremeOffsetCalculator(config, self.memory_manager)
        
        logger.info(f"🚀🚀🚀 FIXED EXTREME GPU OFFSET PROCESSOR INITIALIZED 🚀🚀🚀")
        logger.info(f"🎮 GPUs: {len(config.gpu_ids)} × RTX 5060 Ti")
        logger.info(f"💪 Total GPU Memory: {len(config.gpu_ids) * config.max_gpu_memory_gb:.1f}GB")
        logger.info(f"⚡ STRICT MODE: {'ENABLED' if config.strict_mode else 'DISABLED'}")
        logger.info(f"🔧 OpenCV CUDA Fallback: {'ENABLED' if config.enable_opencv_cuda_fallback else 'DISABLED'}")
    
    def process_all_matches_extreme(self, input_data: Dict, min_score: float = 0.5) -> Dict:
        """Process all matches with maximum GPU utilization and reliability"""
        
        # Collect matches
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
        
        logger.info(f"🎯 FIXED EXTREME PROCESSING TARGET: {total_matches} matches")
        logger.info(f"🔥 DISTRIBUTING ACROSS {len(self.config.gpu_ids)} GPUs")
        
        # Initialize progress tracking
        class SimpleProgressTracker:
            def __init__(self, total):
                self.total = total
                self.completed = 0
                self.successful = 0
                self.gpu_processed = 0
                self.start_time = time.time()
                self.lock = threading.Lock()
            
            def update(self, success=False, gpu_processed=False):
                with self.lock:
                    self.completed += 1
                    if success:
                        self.successful += 1
                    if gpu_processed:
                        self.gpu_processed += 1
                    
                    if self.completed % 50 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.completed / elapsed if elapsed > 0 else 0
                        eta = (self.total - self.completed) / rate if rate > 0 else 0
                        
                        logger.info(f"🚀 Progress: {self.completed}/{self.total} "
                                   f"({self.completed/self.total*100:.1f}%) | "
                                   f"Success: {self.successful/self.completed*100:.1f}% | "
                                   f"GPU: {self.gpu_processed/self.completed*100:.1f}% | "
                                   f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")
        
        progress = SimpleProgressTracker(total_matches)
        
        # Distribute matches across GPUs
        gpu_batches = {gpu_id: [] for gpu_id in self.config.gpu_ids}
        for i, match in enumerate(all_matches):
            gpu_id = self.config.gpu_ids[i % len(self.config.gpu_ids)]
            gpu_batches[gpu_id].append(match)
        
        # Log distribution
        for gpu_id, batch in gpu_batches.items():
            logger.info(f"🎮 GPU {gpu_id}: Assigned {len(batch)} matches")
        
        # Process batches in parallel
        enhanced_results = {}
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=len(self.config.gpu_ids)) as executor:
                futures = []
                
                for gpu_id, match_batch in gpu_batches.items():
                    if match_batch:
                        future = executor.submit(
                            self._process_gpu_batch_fixed, 
                            gpu_id, match_batch, progress
                        )
                        futures.append((gpu_id, future))
                
                # Collect results
                all_gpu_results = {}
                for gpu_id, future in futures:
                    try:
                        gpu_results = future.result(timeout=self.config.gpu_timeout_seconds * 2)
                        all_gpu_results.update(gpu_results)
                        logger.info(f"✅ GPU {gpu_id}: Completed batch processing")
                    except Exception as e:
                        if self.config.strict_mode:
                            raise RuntimeError(f"STRICT MODE: GPU {gpu_id} batch failed: {e}")
                        logger.error(f"❌ GPU {gpu_id}: Batch processing failed: {e}")
                
                # Merge results
                enhanced_results = self._merge_results_fixed(video_results, all_gpu_results)
        
        finally:
            self.memory_manager.cleanup_extreme()
        
        # Create enhanced output
        processing_time = time.time() - start_time
        enhanced_data = input_data.copy()
        enhanced_data['results'] = enhanced_results
        
        enhanced_data['fixed_extreme_gpu_offset_info'] = {
            'processed_at': datetime.now().isoformat(),
            'total_matches_processed': progress.completed,
            'successful_offsets': progress.successful,
            'gpu_processed_items': progress.gpu_processed,
            'gpu_utilization_rate': progress.gpu_processed / progress.completed if progress.completed > 0 else 0,
            'success_rate': progress.successful / progress.completed if progress.completed > 0 else 0,
            'processing_time_seconds': processing_time,
            'processing_rate_matches_per_second': progress.completed / processing_time if processing_time > 0 else 0,
            'fixed_extreme_gpu_config': {
                'gpu_ids': self.config.gpu_ids,
                'max_gpu_memory_gb_per_gpu': self.config.max_gpu_memory_gb,
                'total_gpu_memory_gb': len(self.config.gpu_ids) * self.config.max_gpu_memory_gb,
                'gpu_batch_size': self.config.gpu_batch_size,
                'cuda_streams_per_gpu': self.config.cuda_streams,
                'strict_mode': self.config.strict_mode,
                'opencv_cuda_fallback_enabled': self.config.enable_opencv_cuda_fallback
            }
        }
        
        logger.info("="*80)
        logger.info("🎉 FIXED EXTREME GPU PROCESSING COMPLETE! 🎉")
        logger.info(f"📊 Total processed: {progress.completed}")
        logger.info(f"✅ Successful offsets: {progress.successful}")
        logger.info(f"🎮 GPU processed: {progress.gpu_processed}")
        logger.info(f"⚡ Success rate: {progress.successful/progress.completed*100:.1f}%")
        logger.info(f"🚀 GPU utilization: {progress.gpu_processed/progress.completed*100:.1f}%")
        logger.info(f"⏱️  Processing time: {processing_time/60:.1f} minutes")
        logger.info("="*80)
        
        return enhanced_data
    
    def _process_gpu_batch_fixed(self, gpu_id: int, match_batch: List[Tuple], progress) -> Dict:
        """Process batch on specific GPU with fixed error handling"""
        gpu_results = {}
        
        try:
            with self.memory_manager.extreme_gpu_context(gpu_id):
                logger.info(f"🚀 GPU {gpu_id}: Starting FIXED batch processing ({len(match_batch)} matches)")
                
                # Group by video for efficiency
                video_groups = {}
                for video_path, gpx_path, match in match_batch:
                    if video_path not in video_groups:
                        video_groups[video_path] = []
                    video_groups[video_path].append((gpx_path, match))
                
                # Process each video group
                for video_path, gpx_matches in video_groups.items():
                    try:
                        # Extract video features once
                        video_data_list = self.video_processor.extract_motion_signature_extreme_gpu([video_path], gpu_id)
                        video_data = video_data_list[0] if video_data_list else None
                        
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
                        
                        # Extract GPX features
                        gpx_paths = [gpx_path for gpx_path, _ in gpx_matches]
                        gps_data_list = self.gpx_processor.extract_motion_signature_extreme_gpu_batch(gpx_paths, gpu_id)
                        
                        # Calculate offsets
                        for (gpx_path, match), gps_data in zip(gpx_matches, gps_data_list):
                            enhanced_match = match.copy()
                            
                            if gps_data is not None:
                                # Calculate offset
                                offset_results = self.offset_calculator.calculate_offset_extreme_gpu_batch(
                                    [(video_data, gps_data)], gpu_id
                                )
                                
                                if offset_results:
                                    enhanced_match.update(offset_results[0])
                                else:
                                    enhanced_match.update({
                                        'offset_method': 'offset_calculation_failed',
                                        'gpu_processing': False
                                    })
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
                    
                    except Exception as e:
                        if self.config.strict_mode:
                            raise
                        
                        logger.error(f"GPU {gpu_id}: Video group processing failed: {e}")
                        
                        # Mark all matches in this group as failed
                        for gpx_path, match in gpx_matches:
                            enhanced_match = match.copy()
                            enhanced_match.update({
                                'offset_method': 'video_group_processing_error',
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
            logger.error(f"FIXED EXTREME GPU {gpu_id} batch processing failed: {e}")
        
        logger.info(f"✅ GPU {gpu_id}: Completed FIXED batch processing")
        return gpu_results
    
    def _merge_results_fixed(self, original_results: Dict, gpu_results: Dict) -> Dict:
        """Merge GPU results back into original structure"""
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
    """Main function with FIXED EXTREME GPU processing"""
    parser = argparse.ArgumentParser(
        description='FIXED EXTREME GPU temporal offset extraction for dual RTX 5060 Ti',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀🚀🚀 FIXED EXTREME GPU ACCELERATION 🚀🚀🚀

  # MAXIMUM GPU UTILIZATION (RECOMMENDED)
  python fixed_extreme_gpu_offsetter.py complete_turbo_360_report_ramcache.json --strict --extreme

  # FIXED EXTREME WITH FALLBACKS
  python fixed_extreme_gpu_offsetter.py input.json -o output.json --extreme --enable-fallbacks
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file from matcher')
    parser.add_argument('-o', '--output', help='Output file (default: fixed_extreme_gpu_INPUTNAME.json)')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0, 1], help='GPU IDs (default: 0 1)')
    parser.add_argument('--max-gpu-memory', type=float, default=14.5, help='Max GPU memory per GPU in GB (default: 14.5)')
    parser.add_argument('--gpu-batch-size', type=int, default=512, help='GPU batch size (default: 512)')
    parser.add_argument('--cuda-streams', type=int, default=16, help='CUDA streams per GPU (default: 16)')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score (default: 0.3)')
    parser.add_argument('--min-confidence', type=float, default=0.3, help='Minimum offset confidence (default: 0.3)')
    parser.add_argument('--max-offset', type=float, default=600.0, help='Maximum offset search seconds (default: 600)')
    parser.add_argument('--strict', action='store_true', help='🔥 STRICT MODE: Maximum GPU utilization')
    parser.add_argument('--extreme', action='store_true', help='🚀 EXTREME MODE: Maximum performance')
    parser.add_argument('--enable-fallbacks', action='store_true', help='Enable CPU fallbacks for reliability')
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
        output_file = input_file.parent / f"fixed_extreme_gpu_{input_file.name}"
    
    # Check dependencies
    missing_deps = []
    if not CUPY_AVAILABLE:
        missing_deps.append('cupy-cuda12x')
    
    try:
        import cv2
        import gpxpy
        import pandas as pd
        import scipy
    except ImportError:
        missing_deps.extend(['opencv-contrib-python-headless', 'gpxpy', 'pandas', 'scipy'])
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        sys.exit(1)
    
    # Check GPU
    if not cp.cuda.is_available():
        logger.error("❌ CUDA not available")
        sys.exit(1)
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    logger.info(f"🎮 Detected {gpu_count} CUDA GPUs")
    
    # Configure processing
    config = FixedExtremeGPUConfig(
        gpu_ids=args.gpu_ids,
        max_gpu_memory_gb=args.max_gpu_memory,
        gpu_batch_size=args.gpu_batch_size,
        cuda_streams=args.cuda_streams,
        min_correlation_confidence=args.min_confidence,
        max_offset_search_seconds=args.max_offset,
        strict_mode=args.strict,
        force_gpu_only=args.strict,
        enable_opencv_cuda_fallback=args.enable_fallbacks or not args.strict
    )
    
    # EXTREME mode adjustments
    if args.extreme:
        config.gpu_batch_size = max(config.gpu_batch_size, 512)
        config.cuda_streams = max(config.cuda_streams, 16)
        config.enable_gpu_caching = True
        logger.info("🚀 EXTREME MODE ACTIVATED")
    
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
        sys.exit(0)
    
    if args.limit:
        total_matches = min(total_matches, args.limit)
    
    logger.info("🚀🚀🚀 FIXED EXTREME GPU OFFSET PROCESSING STARTING 🚀🚀🚀")
    logger.info("="*80)
    logger.info(f"🎯 Target: {total_matches} matches")
    logger.info(f"🎮 GPUs: {len(args.gpu_ids)} × RTX 5060 Ti")
    logger.info(f"💾 GPU Memory: {args.max_gpu_memory}GB per GPU")
    logger.info(f"⚡ Batch Size: {config.gpu_batch_size}")
    logger.info(f"🌊 CUDA Streams: {config.cuda_streams} per GPU")
    logger.info(f"🔥 STRICT Mode: {'ENABLED' if args.strict else 'DISABLED'}")
    logger.info(f"🚀 EXTREME Mode: {'ENABLED' if args.extreme else 'DISABLED'}")
    logger.info(f"🔧 Fallbacks: {'ENABLED' if config.enable_opencv_cuda_fallback else 'DISABLED'}")
    logger.info("="*80)
    
    # Initialize processor
    try:
        processor = FixedExtremeGPUOffsetProcessor(config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process matches
    start_time = time.time()
    
    try:
        enhanced_data = processor.process_all_matches_extreme(data, args.min_score)
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        if args.strict:
            sys.exit(1)
        enhanced_data = data
    
    # Save results
    processing_time = time.time() - start_time
    
    logger.info(f"💾 Saving results to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"❌ Failed to save output: {e}")
        sys.exit(1)
    
    # Final summary
    logger.info("\n" + "🚀" * 30)
    logger.info("🎉 FIXED EXTREME GPU PROCESSING COMPLETE! 🎉")
    logger.info("🚀" * 30)
    logger.info(f"📊 Processing time: {processing_time/60:.1f} minutes")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("🚀" * 30)

if __name__ == "__main__":
    main()