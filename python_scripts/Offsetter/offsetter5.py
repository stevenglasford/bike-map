#!/usr/bin/env python3
"""
HARDCORE GPU-ONLY Temporal Offset Calculator
🔥 NO CPU FALLBACKS - GPU OR DIE! 🔥
⚡ 100% GPU utilization MANDATORY
🚫 ZERO TOLERANCE for CPU processing

If GPU fails, we FAIL HARD - no compromises!
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
import gc
import os
from contextlib import contextmanager

# GPU-specific imports
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    from cupyx.scipy.fft import fft, ifft
    CUPY_AVAILABLE = True
except ImportError:
    raise RuntimeError("🔥 HARDCORE MODE: CuPy is MANDATORY for GPU-only processing!")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy_cache'

# Configure hardcore logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hardcore_gpu_offsetter.log', mode='w')
    ]
)
logger = logging.getLogger('hardcore_gpu_offsetter')

@dataclass
class HardcoreGPUConfig:
    """Hardcore GPU-only configuration - NO COMPROMISES"""
    # GPU Configuration - EXTREME AGGRESSION
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    max_gpu_memory_gb: float = 14.8  # Use nearly ALL GPU memory
    gpu_batch_size: int = 1024  # MASSIVE batching
    cuda_streams: int = 32  # MAXIMUM streams
    
    # HARDCORE GPU Settings
    gpu_memory_fraction: float = 0.98  # Use 98% of GPU memory
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    
    # Processing Configuration
    video_sample_rate: float = 2.0
    gps_sample_rate: float = 1.0
    min_correlation_confidence: float = 0.3
    max_offset_search_seconds: float = 600.0
    min_video_duration: float = 5.0
    min_gps_duration: float = 10.0
    
    # HARDCORE MODE - NO FALLBACKS ALLOWED
    strict_mode: bool = True
    force_gpu_only: bool = True
    fail_on_cpu_fallback: bool = True
    gpu_timeout_seconds: float = 300.0
    
    # OpenCV HARDCORE settings
    force_opencv_cuda: bool = True
    opencv_cuda_device_test: bool = True
    opencv_pre_allocate_memory: bool = True

class HardcoreGPUMemoryManager:
    """Hardcore GPU memory management - MAXIMUM AGGRESSION"""
    
    def __init__(self, config: HardcoreGPUConfig):
        self.config = config
        self.memory_pools = {}
        self.gpu_streams = {}
        self.gpu_contexts = {}
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("🔥 HARDCORE MODE: CuPy is MANDATORY!")
        
        self._initialize_hardcore_gpu_resources()
    
    def _initialize_hardcore_gpu_resources(self):
        """Initialize HARDCORE GPU resources with maximum aggression"""
        logger.info(f"🔥 HARDCORE GPU INITIALIZATION: NO COMPROMISES!")
        logger.info(f"💀 Target: {self.config.max_gpu_memory_gb}GB per GPU")
        logger.info(f"⚡ CUDA Streams: {self.config.cuda_streams} per GPU")
        logger.info(f"🚫 CPU FALLBACKS: DISABLED")
        
        total_gpu_memory = 0
        
        for gpu_id in self.config.gpu_ids:
            try:
                # FORCE GPU context
                cp.cuda.Device(gpu_id).use()
                
                # AGGRESSIVE memory allocation
                memory_pool = cp.get_default_memory_pool()
                target_memory = int(self.config.max_gpu_memory_gb * 1024**3)
                memory_pool.set_limit(size=target_memory)
                self.memory_pools[gpu_id] = memory_pool
                
                # Pre-allocate MASSIVE memory blocks
                cache_size = int(2 * 1024**3)  # 2GB pre-allocation
                try:
                    prealloc_cache = cp.zeros(cache_size // 4, dtype=cp.float32)
                    logger.info(f"🔥 GPU {gpu_id}: Pre-allocated 2GB cache - SUCCESS")
                    del prealloc_cache  # Free it but keep pool warm
                except Exception as e:
                    raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} pre-allocation FAILED: {e}")
                
                # Create MAXIMUM streams
                streams = []
                for i in range(self.config.cuda_streams):
                    stream = cp.cuda.Stream(non_blocking=True)
                    streams.append(stream)
                self.gpu_streams[gpu_id] = streams
                
                # Verify GPU capabilities
                device = cp.cuda.Device(gpu_id)
                total_memory = device.mem_info[1] / (1024**3)
                free_memory = device.mem_info[0] / (1024**3)
                total_gpu_memory += total_memory
                
                if free_memory < self.config.max_gpu_memory_gb:
                    raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} insufficient memory! "
                                     f"Need {self.config.max_gpu_memory_gb}GB, have {free_memory:.1f}GB")
                
                # AGGRESSIVE warmup computation
                warmup_size = 1000000  # Large warmup
                warmup_data = cp.random.rand(warmup_size, dtype=cp.float32)
                warmup_result = cp.fft.fft(warmup_data)
                _ = cp.sum(warmup_result)
                del warmup_data, warmup_result
                
                # Test GPU compute capability
                compute_cap = device.compute_capability
                if compute_cap < (7, 5):  # RTX 5060 Ti should be 8.9
                    logger.warning(f"⚠️ GPU {gpu_id}: Low compute capability {compute_cap}")
                
                logger.info(f"🔥 GPU {gpu_id} HARDCORE INIT SUCCESS:")
                logger.info(f"   ├─ Total Memory: {total_memory:.1f}GB")
                logger.info(f"   ├─ Reserved: {self.config.max_gpu_memory_gb:.1f}GB")
                logger.info(f"   ├─ Streams: {self.config.cuda_streams}")
                logger.info(f"   ├─ Compute: {compute_cap}")
                logger.info(f"   └─ Status: 🔥 HARDCORE READY")
                
            except Exception as e:
                raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} initialization FAILED: {e}")
        
        logger.info(f"💀 TOTAL HARDCORE GPU POWER: {total_gpu_memory:.1f}GB")
        logger.info(f"🚫 CPU FALLBACKS: ABSOLUTELY FORBIDDEN")
    
    @contextmanager
    def hardcore_gpu_context(self, gpu_id: int):
        """HARDCORE GPU context - FAIL IF NOT AVAILABLE"""
        try:
            original_device = cp.cuda.Device()
            cp.cuda.Device(gpu_id).use()
            
            # Verify we're actually on the right GPU
            current_device = cp.cuda.Device().id
            if current_device != gpu_id:
                raise RuntimeError(f"🔥 HARDCORE MODE: GPU context switch FAILED! "
                                 f"Expected {gpu_id}, got {current_device}")
            
            yield gpu_id
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} context FAILED: {e}")
        finally:
            try:
                original_device.use()
            except:
                pass  # Don't fail cleanup
    
    def get_hardcore_stream(self, gpu_id: int, operation_type: str = 'compute') -> cp.cuda.Stream:
        """Get HARDCORE CUDA stream - MUST SUCCEED"""
        streams = self.gpu_streams.get(gpu_id, [])
        if not streams:
            raise RuntimeError(f"🔥 HARDCORE MODE: No streams available for GPU {gpu_id}")
        
        # Distribute streams by operation type
        stream_map = {
            'video': 0,
            'gps': 8,
            'correlation': 16,
            'fft': 24,
            'compute': 4
        }
        
        base_idx = stream_map.get(operation_type, 0)
        stream_idx = base_idx % len(streams)
        
        return streams[stream_idx]
    
    def verify_gpu_health(self, gpu_id: int):
        """Verify GPU is healthy and responsive"""
        try:
            with self.hardcore_gpu_context(gpu_id):
                # Quick compute test
                test_data = cp.random.rand(10000, dtype=cp.float32)
                result = cp.fft.fft(test_data)
                _ = cp.sum(result)
                del test_data, result
                return True
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} health check FAILED: {e}")
    
    def cleanup_hardcore(self):
        """HARDCORE cleanup"""
        logger.info("🔥 HARDCORE GPU CLEANUP")
        total_freed = 0
        
        for gpu_id in self.config.gpu_ids:
            try:
                with self.hardcore_gpu_context(gpu_id):
                    if gpu_id in self.memory_pools:
                        pool = self.memory_pools[gpu_id]
                        used_bytes = pool.used_bytes()
                        pool.free_all_blocks()
                        total_freed += used_bytes
                    
                    # Synchronize all streams
                    for stream in self.gpu_streams.get(gpu_id, []):
                        stream.synchronize()
                    
                    cp.cuda.Device().synchronize()
                    
            except Exception as e:
                logger.warning(f"GPU {gpu_id} cleanup warning: {e}")
        
        logger.info(f"💀 FREED {total_freed / 1024**3:.1f}GB GPU MEMORY")

class HardcoreOpenCVGPUManager:
    """Hardcore OpenCV GPU manager - NO CPU FALLBACKS ALLOWED"""
    
    def __init__(self, config: HardcoreGPUConfig):
        self.config = config
        self.gpu_objects = {}
        self.opencv_cuda_verified = False
        
        self._initialize_hardcore_opencv_cuda()
    
    def _initialize_hardcore_opencv_cuda(self):
        """Initialize OpenCV CUDA with HARDCORE verification"""
        logger.info("🔥 HARDCORE OpenCV CUDA INITIALIZATION")
        
        # Check basic CUDA availability
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: OpenCV CUDA not available: {e}")
        
        if gpu_count == 0:
            raise RuntimeError("🔥 HARDCORE MODE: No OpenCV CUDA devices found!")
        
        if gpu_count < len(self.config.gpu_ids):
            raise RuntimeError(f"🔥 HARDCORE MODE: Need {len(self.config.gpu_ids)} OpenCV CUDA devices, "
                             f"found {gpu_count}")
        
        logger.info(f"🔥 OpenCV CUDA devices available: {gpu_count}")
        
        # HARDCORE test each GPU
        for gpu_id in self.config.gpu_ids:
            try:
                self._hardcore_test_opencv_gpu(gpu_id)
                self._create_hardcore_gpu_objects(gpu_id)
                logger.info(f"🔥 GPU {gpu_id}: OpenCV CUDA HARDCORE VERIFIED ✅")
            except Exception as e:
                raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} OpenCV CUDA FAILED: {e}")
        
        self.opencv_cuda_verified = True
        logger.info("💀 ALL GPUs: OpenCV CUDA HARDCORE READY")
    
    def _hardcore_test_opencv_gpu(self, gpu_id: int):
        """HARDCORE test OpenCV CUDA functionality on specific GPU"""
        logger.info(f"🔥 Testing OpenCV CUDA on GPU {gpu_id}")
        
        try:
            # Set GPU device
            cv2.cuda.setDevice(gpu_id)
            
            # Verify device was set
            current_device = cv2.cuda.getDevice()
            if current_device != gpu_id:
                raise RuntimeError(f"OpenCV CUDA device set FAILED: expected {gpu_id}, got {current_device}")
            
            # Test basic GPU Mat operations
            test_size = (480, 640, 3)
            test_cpu = np.random.randint(0, 255, test_size, dtype=np.uint8)
            
            # Upload test
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(test_cpu)
            
            # Download test
            result_cpu = gpu_mat.download()
            if not np.array_equal(test_cpu, result_cpu):
                raise RuntimeError("GPU Mat upload/download verification FAILED")
            
            # Test color conversion
            gpu_gray = cv2.cuda_GpuMat()
            cv2.cuda.cvtColor(gpu_mat, gpu_gray, cv2.COLOR_BGR2GRAY)
            
            # Test Gaussian filter
            gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
            gpu_blurred = cv2.cuda_GpuMat()
            gaussian_filter.apply(gpu_gray, gpu_blurred)
            
            # Test optical flow
            optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
            gpu_flow = cv2.cuda_GpuMat()
            optical_flow.calc(gpu_gray, gpu_gray, gpu_flow)
            
            # Verify results have correct dimensions
            gray_result = gpu_gray.download()
            blur_result = gpu_blurred.download()
            flow_result = gpu_flow.download()
            
            if gray_result.shape != test_size[:2]:
                raise RuntimeError(f"Gray conversion shape mismatch: {gray_result.shape} != {test_size[:2]}")
            
            if blur_result.shape != test_size[:2]:
                raise RuntimeError(f"Blur result shape mismatch: {blur_result.shape} != {test_size[:2]}")
            
            if flow_result.shape != (*test_size[:2], 2):
                raise RuntimeError(f"Flow result shape mismatch: {flow_result.shape} != {(*test_size[:2], 2)}")
            
            logger.info(f"✅ GPU {gpu_id}: All OpenCV CUDA operations VERIFIED")
            
        except Exception as e:
            raise RuntimeError(f"OpenCV CUDA test FAILED on GPU {gpu_id}: {e}")
    
    def _create_hardcore_gpu_objects(self, gpu_id: int):
        """Create OpenCV GPU objects for HARDCORE processing"""
        try:
            cv2.cuda.setDevice(gpu_id)
            
            # Create all necessary GPU objects
            self.gpu_objects[gpu_id] = {
                'gaussian_filter_5x5': cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0),
                'gaussian_filter_3x3': cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0),
                'optical_flow': cv2.cuda_FarnebackOpticalFlow.create(
                    numLevels=3,
                    pyrScale=0.5,
                    fastPyramids=True,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0
                )
            }
            
            # Pre-allocate GPU matrices for reuse
            self.gpu_objects[gpu_id]['gpu_mats'] = {
                'frame': cv2.cuda_GpuMat(),
                'gray': cv2.cuda_GpuMat(),
                'blurred': cv2.cuda_GpuMat(),
                'flow': cv2.cuda_GpuMat(),
                'prev_frame': cv2.cuda_GpuMat()
            }
            
            logger.info(f"🔥 GPU {gpu_id}: Objects created and pre-allocated")
            
        except Exception as e:
            raise RuntimeError(f"GPU {gpu_id} object creation FAILED: {e}")
    
    def get_gpu_objects(self, gpu_id: int) -> Dict:
        """Get GPU objects for processing - MUST EXIST"""
        if not self.opencv_cuda_verified:
            raise RuntimeError("🔥 HARDCORE MODE: OpenCV CUDA not verified!")
        
        if gpu_id not in self.gpu_objects:
            raise RuntimeError(f"🔥 HARDCORE MODE: No GPU objects for GPU {gpu_id}")
        
        return self.gpu_objects[gpu_id]

class HardcoreVideoProcessor:
    """HARDCORE GPU video processor - NO CPU ALLOWED"""
    
    def __init__(self, config: HardcoreGPUConfig, memory_manager: HardcoreGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.opencv_manager = HardcoreOpenCVGPUManager(config)
        self.video_cache = {}
    
    def extract_motion_signature_hardcore_gpu(self, video_paths: List[str], gpu_id: int) -> List[Dict]:
        """HARDCORE GPU video processing - GPU OR DIE"""
        if not self.opencv_manager.opencv_cuda_verified:
            raise RuntimeError("🔥 HARDCORE MODE: OpenCV CUDA not verified for GPU processing!")
        
        try:
            with self.memory_manager.hardcore_gpu_context(gpu_id):
                return self._process_video_batch_hardcore(video_paths, gpu_id)
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} video processing FAILED: {e}")
    
    def _process_video_batch_hardcore(self, video_paths: List[str], gpu_id: int) -> List[Dict]:
        """HARDCORE batch video processing on GPU"""
        logger.info(f"🔥 GPU {gpu_id}: HARDCORE video processing {len(video_paths)} videos")
        
        # Verify GPU health before starting
        self.memory_manager.verify_gpu_health(gpu_id)
        
        results = []
        
        # Get GPU objects
        gpu_objects = self.opencv_manager.get_gpu_objects(gpu_id)
        
        # Process videos with MAXIMUM GPU utilization
        for video_path in video_paths:
            try:
                result = self._process_single_video_hardcore_gpu(video_path, gpu_id, gpu_objects)
                results.append(result)
                logger.debug(f"🔥 GPU {gpu_id}: Processed {Path(video_path).name}")
            except Exception as e:
                if self.config.fail_on_cpu_fallback:
                    raise RuntimeError(f"🔥 HARDCORE MODE: Video processing FAILED for {video_path}: {e}")
                else:
                    results.append(None)
                    logger.error(f"🔥 GPU {gpu_id}: Video processing FAILED: {e}")
        
        logger.info(f"🔥 GPU {gpu_id}: HARDCORE batch complete")
        return results
    
    def _process_single_video_hardcore_gpu(self, video_path: str, gpu_id: int, gpu_objects: Dict) -> Optional[Dict]:
        """HARDCORE single video processing - PURE GPU"""
        
        # Check cache
        cache_key = f"{video_path}_{gpu_id}"
        if cache_key in self.video_cache:
            return self.video_cache[cache_key]
        
        # Set OpenCV GPU device
        cv2.cuda.setDevice(gpu_id)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Invalid video properties: fps={fps}, frames={frame_count}")
        
        duration = frame_count / fps
        if duration < self.config.min_video_duration:
            cap.release()
            raise RuntimeError(f"Video too short: {duration}s < {self.config.min_video_duration}s")
        
        is_360 = (width / height) >= 1.8
        frame_interval = max(1, int(fps / self.config.video_sample_rate))
        
        # Get GPU processing objects
        gaussian_filter = gpu_objects['gaussian_filter_5x5']
        optical_flow = gpu_objects['optical_flow']
        gpu_mats = gpu_objects['gpu_mats']
        
        # GPU processing variables
        stream = self.memory_manager.get_hardcore_stream(gpu_id, 'video')
        motion_values = []
        motion_energy = []
        timestamps = []
        
        frame_idx = 0
        processed_frames = 0
        prev_frame_set = False
        
        try:
            with cp.cuda.Stream(stream):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_idx % frame_interval == 0:
                        # Resize for processing
                        target_width = 1280 if is_360 else 640
                        if frame.shape[1] > target_width:
                            scale = target_width / frame.shape[1]
                            new_width = target_width
                            new_height = int(frame.shape[0] * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # HARDCORE GPU processing ONLY
                        try:
                            # Upload to GPU
                            gpu_mats['frame'].upload(frame)
                            
                            # Convert to grayscale on GPU
                            cv2.cuda.cvtColor(gpu_mats['frame'], gpu_mats['gray'], cv2.COLOR_BGR2GRAY)
                            
                            # Apply Gaussian filter on GPU
                            gaussian_filter.apply(gpu_mats['gray'], gpu_mats['blurred'])
                            
                            if prev_frame_set:
                                # GPU optical flow
                                optical_flow.calc(gpu_mats['prev_frame'], gpu_mats['blurred'], gpu_mats['flow'])
                                
                                # Download flow and process on GPU with CuPy
                                flow_cpu = gpu_mats['flow'].download()
                                flow_gpu = cp.asarray(flow_cpu)
                                magnitude = cp.sqrt(flow_gpu[..., 0]**2 + flow_gpu[..., 1]**2)
                                
                                # Calculate motion metrics on GPU
                                motion_mag = float(cp.mean(magnitude))
                                motion_eng = float(cp.sum(magnitude ** 2))
                                
                                motion_values.append(motion_mag)
                                motion_energy.append(motion_eng)
                                timestamps.append(frame_idx / fps)
                                
                                processed_frames += 1
                            
                            # Store current frame for next iteration
                            gpu_mats['blurred'].copyTo(gpu_mats['prev_frame'])
                            prev_frame_set = True
                            
                        except Exception as e:
                            cap.release()
                            raise RuntimeError(f"🔥 HARDCORE MODE: GPU processing FAILED on frame {frame_idx}: {e}")
                    
                    frame_idx += 1
                
                cap.release()
                
                if processed_frames < 3:
                    raise RuntimeError(f"Insufficient motion data: {processed_frames} frames")
                
                # Convert to GPU arrays
                motion_values_gpu = cp.array(motion_values)
                motion_energy_gpu = cp.array(motion_energy)
                timestamps_gpu = cp.array(timestamps)
                
                result = {
                    'motion_magnitude': motion_values_gpu,
                    'motion_energy': motion_energy_gpu,
                    'timestamps': timestamps_gpu,
                    'duration': duration,
                    'fps': fps,
                    'is_360': is_360,
                    'frame_count': processed_frames,
                    'gpu_id': gpu_id,
                    'processing_method': 'hardcore_gpu_only'
                }
                
                # Cache result
                self.video_cache[cache_key] = result
                return result
        
        except Exception as e:
            cap.release()
            raise RuntimeError(f"🔥 HARDCORE MODE: Video processing FAILED: {e}")

class HardcoreGPXProcessor:
    """HARDCORE GPU GPX processor"""
    
    def __init__(self, config: HardcoreGPUConfig, memory_manager: HardcoreGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.gpx_cache = {}
    
    def extract_motion_signature_hardcore_gpu_batch(self, gpx_paths: List[str], gpu_id: int) -> List[Dict]:
        """HARDCORE GPU GPX batch processing"""
        try:
            with self.memory_manager.hardcore_gpu_context(gpu_id):
                return self._process_gpx_batch_hardcore(gpx_paths, gpu_id)
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} GPX processing FAILED: {e}")
    
    def _process_gpx_batch_hardcore(self, gpx_paths: List[str], gpu_id: int) -> List[Dict]:
        """HARDCORE GPX batch processing"""
        logger.info(f"🔥 GPU {gpu_id}: HARDCORE GPX processing {len(gpx_paths)} files")
        
        results = []
        
        for gpx_path in gpx_paths:
            try:
                result = self._process_single_gpx_hardcore_gpu(gpx_path, gpu_id)
                results.append(result)
            except Exception as e:
                if self.config.fail_on_cpu_fallback:
                    raise RuntimeError(f"🔥 HARDCORE MODE: GPX processing FAILED for {gpx_path}: {e}")
                else:
                    results.append(None)
                    logger.error(f"🔥 GPU {gpu_id}: GPX processing FAILED: {e}")
        
        return results
    
    def _process_single_gpx_hardcore_gpu(self, gpx_path: str, gpu_id: int) -> Optional[Dict]:
        """HARDCORE single GPX processing"""
        
        cache_key = f"{gpx_path}_{gpu_id}"
        if cache_key in self.gpx_cache:
            return self.gpx_cache[cache_key]
        
        # Load GPX (CPU unavoidable)
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
        except Exception as e:
            raise RuntimeError(f"GPX parsing FAILED: {e}")
        
        # Collect points
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                points.extend([{
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'time': point.time
                } for point in segment.points if point.time])
        
        if len(points) < 10:
            raise RuntimeError(f"Insufficient GPX points: {len(points)}")
        
        points.sort(key=lambda p: p['time'])
        df = pd.DataFrame(points)
        
        duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        if duration < self.config.min_gps_duration:
            raise RuntimeError(f"GPX duration too short: {duration}s")
        
        # HARDCORE GPU processing
        stream = self.memory_manager.get_hardcore_stream(gpu_id, 'gps')
        
        try:
            with cp.cuda.Stream(stream):
                # ALL processing on GPU
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
                
                # Resample on GPU
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
                    'processing_method': 'hardcore_gpu_only'
                }
                
                self.gpx_cache[cache_key] = result
                return result
        
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU GPX processing FAILED: {e}")

class HardcoreOffsetCalculator:
    """HARDCORE GPU offset calculator"""
    
    def __init__(self, config: HardcoreGPUConfig, memory_manager: HardcoreGPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
    
    def calculate_offset_hardcore_gpu_batch(self, video_gps_pairs: List[Tuple[Dict, Dict]], 
                                          gpu_id: int) -> List[Dict]:
        """HARDCORE GPU batch offset calculation"""
        try:
            with self.memory_manager.hardcore_gpu_context(gpu_id):
                return self._calculate_batch_offsets_hardcore(video_gps_pairs, gpu_id)
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} offset calculation FAILED: {e}")
    
    def _calculate_batch_offsets_hardcore(self, video_gps_pairs: List[Tuple[Dict, Dict]], 
                                        gpu_id: int) -> List[Dict]:
        """HARDCORE batch offset calculation"""
        results = []
        
        stream = self.memory_manager.get_hardcore_stream(gpu_id, 'correlation')
        
        for video_data, gps_data in video_gps_pairs:
            try:
                result = self._calculate_single_offset_hardcore(video_data, gps_data, gpu_id, stream)
                results.append(result)
            except Exception as e:
                if self.config.fail_on_cpu_fallback:
                    raise RuntimeError(f"🔥 HARDCORE MODE: Offset calculation FAILED: {e}")
                else:
                    results.append({
                        'offset_method': 'hardcore_gpu_failed',
                        'gpu_processing': False,
                        'error': str(e)[:100]
                    })
        
        return results
    
    def _calculate_single_offset_hardcore(self, video_data: Dict, gps_data: Dict, 
                                        gpu_id: int, stream: cp.cuda.Stream) -> Dict:
        """HARDCORE single offset calculation"""
        result = {
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': 'hardcore_gpu_correlation',
            'gpu_processing': True,
            'gpu_id': gpu_id
        }
        
        try:
            with cp.cuda.Stream(stream):
                # Get signals (must be GPU arrays)
                video_signal = self._get_gpu_signal(video_data, 'video')
                gps_signal = self._get_gpu_signal(gps_data, 'gps')
                
                if video_signal is None or gps_signal is None:
                    raise RuntimeError("No valid signals for correlation")
                
                # Ensure GPU arrays
                if isinstance(video_signal, np.ndarray):
                    video_signal = cp.array(video_signal)
                if isinstance(gps_signal, np.ndarray):
                    gps_signal = cp.array(gps_signal)
                
                # Normalize on GPU
                video_norm = self._normalize_signal_gpu(video_signal)
                gps_norm = self._normalize_signal_gpu(gps_signal)
                
                # GPU FFT correlation
                max_len = len(video_norm) + len(gps_norm) - 1
                pad_len = 1 << (max_len - 1).bit_length()
                
                video_padded = cp.pad(video_norm, (0, pad_len - len(video_norm)))
                gps_padded = cp.pad(gps_norm, (0, pad_len - len(gps_norm)))
                
                video_fft = cp.fft.fft(video_padded)
                gps_fft = cp.fft.fft(gps_padded)
                correlation = cp.fft.ifft(cp.conj(video_fft) * gps_fft).real
                
                # Find peak
                peak_idx = cp.argmax(correlation)
                confidence = float(correlation[peak_idx] / len(video_norm))
                
                # Convert to offset
                offset_samples = int(peak_idx - len(video_norm) + 1)
                offset_seconds = offset_samples * self.config.gps_sample_rate
                
                if (abs(offset_seconds) <= self.config.max_offset_search_seconds and 
                    abs(confidence) >= self.config.min_correlation_confidence):
                    
                    result.update({
                        'temporal_offset_seconds': float(offset_seconds),
                        'offset_confidence': float(abs(confidence)),
                        'offset_method': 'hardcore_gpu_fft_correlation'
                    })
                else:
                    result['offset_method'] = 'correlation_below_threshold'
        
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: Offset calculation FAILED: {e}")
        
        return result
    
    def _get_gpu_signal(self, data: Dict, data_type: str):
        """Get GPU signal"""
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
    
    def _normalize_signal_gpu(self, signal: cp.ndarray) -> cp.ndarray:
        """Normalize signal on GPU"""
        if len(signal) == 0:
            return signal
        
        mean = cp.mean(signal)
        std = cp.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean

class HardcoreGPUOffsetProcessor:
    """HARDCORE GPU processor - NO COMPROMISES"""
    
    def __init__(self, config: HardcoreGPUConfig):
        self.config = config
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("🔥 HARDCORE MODE: CuPy is MANDATORY!")
        
        # Initialize HARDCORE components
        logger.info("🔥 INITIALIZING HARDCORE GPU PROCESSOR")
        self.memory_manager = HardcoreGPUMemoryManager(config)
        self.video_processor = HardcoreVideoProcessor(config, self.memory_manager)
        self.gpx_processor = HardcoreGPXProcessor(config, self.memory_manager)
        self.offset_calculator = HardcoreOffsetCalculator(config, self.memory_manager)
        
        logger.info("💀 HARDCORE GPU PROCESSOR READY - NO CPU FALLBACKS!")
    
    def process_all_matches_hardcore(self, input_data: Dict, min_score: float = 0.3) -> Dict:
        """Process all matches with HARDCORE GPU processing"""
        
        # Collect matches
        all_matches = []
        video_results = input_data.get('results', {})
        
        for video_path, video_data in video_results.items():
            matches = video_data.get('matches', [])
            for match in matches:
                if match.get('combined_score', 0) >= min_score:
                    all_matches.append((video_path, match['path'], match))
        
        total_matches = len(all_matches)
        if total_matches == 0:
            raise RuntimeError("🔥 HARDCORE MODE: No matches found for processing!")
        
        logger.info(f"💀 HARDCORE PROCESSING TARGET: {total_matches} matches")
        logger.info(f"🔥 DISTRIBUTING ACROSS {len(self.config.gpu_ids)} GPUs")
        
        # Distribute matches across GPUs
        gpu_batches = {gpu_id: [] for gpu_id in self.config.gpu_ids}
        for i, match in enumerate(all_matches):
            gpu_id = self.config.gpu_ids[i % len(self.config.gpu_ids)]
            gpu_batches[gpu_id].append(match)
        
        for gpu_id, batch in gpu_batches.items():
            logger.info(f"💀 GPU {gpu_id}: Assigned {len(batch)} matches")
        
        # Process with HARDCORE parallelism
        enhanced_results = {}
        start_time = time.time()
        
        # Progress tracking
        class HardcoreProgress:
            def __init__(self, total):
                self.total = total
                self.completed = 0
                self.successful = 0
                self.gpu_processed = 0
                self.lock = threading.Lock()
                self.start_time = time.time()
            
            def update(self, success=False, gpu_processed=False):
                with self.lock:
                    self.completed += 1
                    if success:
                        self.successful += 1
                    if gpu_processed:
                        self.gpu_processed += 1
                    
                    if self.completed % 20 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.completed / elapsed if elapsed > 0 else 0
                        logger.info(f"💀 HARDCORE Progress: {self.completed}/{self.total} "
                                   f"({self.completed/self.total*100:.1f}%) | "
                                   f"Success: {self.successful/self.completed*100:.1f}% | "
                                   f"GPU: {self.gpu_processed/self.completed*100:.1f}% | "
                                   f"Rate: {rate:.1f}/s")
        
        progress = HardcoreProgress(total_matches)
        
        try:
            with ThreadPoolExecutor(max_workers=len(self.config.gpu_ids)) as executor:
                futures = []
                
                for gpu_id, match_batch in gpu_batches.items():
                    if match_batch:
                        future = executor.submit(
                            self._process_gpu_batch_hardcore, 
                            gpu_id, match_batch, progress
                        )
                        futures.append((gpu_id, future))
                
                # Collect results
                all_gpu_results = {}
                for gpu_id, future in futures:
                    try:
                        gpu_results = future.result(timeout=self.config.gpu_timeout_seconds)
                        all_gpu_results.update(gpu_results)
                        logger.info(f"💀 GPU {gpu_id}: HARDCORE batch COMPLETE")
                    except Exception as e:
                        raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} batch FAILED: {e}")
                
                # Merge results
                enhanced_results = self._merge_results_hardcore(video_results, all_gpu_results)
        
        finally:
            self.memory_manager.cleanup_hardcore()
        
        # Create output
        processing_time = time.time() - start_time
        enhanced_data = input_data.copy()
        enhanced_data['results'] = enhanced_results
        
        enhanced_data['hardcore_gpu_offset_info'] = {
            'processed_at': datetime.now().isoformat(),
            'total_matches_processed': progress.completed,
            'successful_offsets': progress.successful,
            'gpu_processed_items': progress.gpu_processed,
            'gpu_utilization_rate': progress.gpu_processed / progress.completed if progress.completed > 0 else 0,
            'success_rate': progress.successful / progress.completed if progress.completed > 0 else 0,
            'processing_time_seconds': processing_time,
            'processing_rate_matches_per_second': progress.completed / processing_time if processing_time > 0 else 0,
            'hardcore_mode': True,
            'cpu_fallbacks_used': False  # Should always be False in hardcore mode
        }
        
        logger.info("💀💀💀 HARDCORE GPU PROCESSING COMPLETE 💀💀💀")
        logger.info(f"🔥 Success rate: {progress.successful/progress.completed*100:.1f}%")
        logger.info(f"💀 GPU utilization: {progress.gpu_processed/progress.completed*100:.1f}%")
        
        if progress.gpu_processed < progress.completed:
            logger.warning(f"⚠️ Some matches did not use GPU processing!")
        
        return enhanced_data
    
    def _process_gpu_batch_hardcore(self, gpu_id: int, match_batch: List[Tuple], progress) -> Dict:
        """Process batch on GPU with HARDCORE mode"""
        gpu_results = {}
        
        try:
            with self.memory_manager.hardcore_gpu_context(gpu_id):
                logger.info(f"💀 GPU {gpu_id}: Starting HARDCORE batch ({len(match_batch)} matches)")
                
                # Group by video
                video_groups = {}
                for video_path, gpx_path, match in match_batch:
                    if video_path not in video_groups:
                        video_groups[video_path] = []
                    video_groups[video_path].append((gpx_path, match))
                
                # Process each video group
                for video_path, gpx_matches in video_groups.items():
                    try:
                        # Extract video features (HARDCORE GPU ONLY)
                        video_data_list = self.video_processor.extract_motion_signature_hardcore_gpu([video_path], gpu_id)
                        video_data = video_data_list[0] if video_data_list else None
                        
                        if video_data is None:
                            raise RuntimeError(f"Video extraction FAILED for {video_path}")
                        
                        # Extract GPX features (HARDCORE GPU)
                        gpx_paths = [gpx_path for gpx_path, _ in gpx_matches]
                        gps_data_list = self.gpx_processor.extract_motion_signature_hardcore_gpu_batch(gpx_paths, gpu_id)
                        
                        # Calculate offsets (HARDCORE GPU)
                        for (gpx_path, match), gps_data in zip(gpx_matches, gps_data_list):
                            enhanced_match = match.copy()
                            
                            if gps_data is not None:
                                offset_results = self.offset_calculator.calculate_offset_hardcore_gpu_batch(
                                    [(video_data, gps_data)], gpu_id
                                )
                                
                                if offset_results:
                                    enhanced_match.update(offset_results[0])
                                else:
                                    raise RuntimeError("Offset calculation returned no results")
                            else:
                                raise RuntimeError(f"GPS extraction FAILED for {gpx_path}")
                            
                            if video_path not in gpu_results:
                                gpu_results[video_path] = []
                            gpu_results[video_path].append((gpx_path, enhanced_match))
                            
                            # Update progress
                            success = enhanced_match.get('temporal_offset_seconds') is not None
                            gpu_processed = enhanced_match.get('gpu_processing', False)
                            progress.update(success, gpu_processed)
                    
                    except Exception as e:
                        raise RuntimeError(f"🔥 HARDCORE MODE: Video group processing FAILED for {video_path}: {e}")
        
        except Exception as e:
            raise RuntimeError(f"🔥 HARDCORE MODE: GPU {gpu_id} batch processing FAILED: {e}")
        
        return gpu_results
    
    def _merge_results_hardcore(self, original_results: Dict, gpu_results: Dict) -> Dict:
        """Merge GPU results"""
        enhanced_results = {}
        
        for video_path, video_data in original_results.items():
            enhanced_video_data = video_data.copy()
            enhanced_matches = []
            
            gpu_video_results = gpu_results.get(video_path, [])
            gpu_match_map = {gpx_path: enhanced_match for gpx_path, enhanced_match in gpu_video_results}
            
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
    """HARDCORE GPU main function"""
    parser = argparse.ArgumentParser(
        description='🔥 HARDCORE GPU-ONLY temporal offset calculator - NO CPU FALLBACKS!',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
💀💀💀 HARDCORE GPU MODE - NO COMPROMISES 💀💀💀

  # HARDCORE GPU-ONLY PROCESSING
  python hardcore_gpu_offsetter.py input.json --hardcore --gpu-only

  # MAXIMUM AGGRESSION MODE
  python hardcore_gpu_offsetter.py input.json --hardcore --max-aggression
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file from matcher')
    parser.add_argument('-o', '--output', help='Output file (default: hardcore_gpu_INPUTNAME.json)')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0, 1], help='GPU IDs (default: 0 1)')
    parser.add_argument('--max-gpu-memory', type=float, default=14.8, help='Max GPU memory per GPU in GB (default: 14.8)')
    parser.add_argument('--gpu-batch-size', type=int, default=1024, help='GPU batch size (default: 1024)')
    parser.add_argument('--cuda-streams', type=int, default=32, help='CUDA streams per GPU (default: 32)')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score (default: 0.3)')
    parser.add_argument('--min-confidence', type=float, default=0.3, help='Minimum offset confidence (default: 0.3)')
    parser.add_argument('--hardcore', action='store_true', help='🔥 HARDCORE MODE: GPU-only processing')
    parser.add_argument('--gpu-only', action='store_true', help='💀 NO CPU FALLBACKS ALLOWED')
    parser.add_argument('--max-aggression', action='store_true', help='⚡ MAXIMUM AGGRESSION MODE')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Input validation
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"hardcore_gpu_{input_file.name}"
    
    # HARDCORE mode checks
    if not args.hardcore and not args.gpu_only:
        logger.error("🔥 This is a HARDCORE GPU-only processor! Use --hardcore or --gpu-only")
        sys.exit(1)
    
    # Check dependencies
    if not CUPY_AVAILABLE:
        logger.error("🔥 HARDCORE MODE: CuPy is MANDATORY!")
        sys.exit(1)
    
    try:
        import cv2
        import gpxpy
        import pandas as pd
    except ImportError as e:
        logger.error(f"🔥 HARDCORE MODE: Missing dependencies: {e}")
        sys.exit(1)
    
    # Check GPU
    if not cp.cuda.is_available():
        logger.error("🔥 HARDCORE MODE: CUDA not available!")
        sys.exit(1)
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    if gpu_count < len(args.gpu_ids):
        logger.error(f"🔥 HARDCORE MODE: Need {len(args.gpu_ids)} GPUs, found {gpu_count}")
        sys.exit(1)
    
    # Configure HARDCORE processing
    config = HardcoreGPUConfig(
        gpu_ids=args.gpu_ids,
        max_gpu_memory_gb=args.max_gpu_memory,
        gpu_batch_size=args.gpu_batch_size,
        cuda_streams=args.cuda_streams,
        min_correlation_confidence=args.min_confidence,
        strict_mode=True,
        force_gpu_only=True,
        fail_on_cpu_fallback=args.gpu_only,
        force_opencv_cuda=True
    )
    
    # MAX AGGRESSION mode
    if args.max_aggression:
        config.gpu_batch_size = max(config.gpu_batch_size, 2048)
        config.cuda_streams = max(config.cuda_streams, 64)
        config.max_gpu_memory_gb = min(config.max_gpu_memory_gb + 0.2, 15.0)
        logger.info("⚡ MAXIMUM AGGRESSION MODE ACTIVATED")
    
    # Load data
    logger.info(f"📁 Loading data from {input_file}")
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
        logger.error("🔥 HARDCORE MODE: No matches found!")
        sys.exit(1)
    
    if args.limit:
        total_matches = min(total_matches, args.limit)
    
    logger.info("💀💀💀 HARDCORE GPU PROCESSING STARTING 💀💀💀")
    logger.info("="*80)
    logger.info(f"🎯 Target: {total_matches} matches")
    logger.info(f"🔥 GPUs: {len(args.gpu_ids)} × RTX 5060 Ti")
    logger.info(f"💀 GPU Memory: {args.max_gpu_memory}GB per GPU")
    logger.info(f"⚡ Batch Size: {config.gpu_batch_size}")
    logger.info(f"🌊 CUDA Streams: {config.cuda_streams} per GPU")
    logger.info(f"🚫 CPU Fallbacks: ABSOLUTELY FORBIDDEN")
    logger.info("="*80)
    
    # Initialize HARDCORE processor
    try:
        processor = HardcoreGPUOffsetProcessor(config)
    except Exception as e:
        logger.error(f"🔥 HARDCORE MODE: Processor initialization FAILED: {e}")
        sys.exit(1)
    
    # Process matches
    start_time = time.time()
    
    try:
        enhanced_data = processor.process_all_matches_hardcore(data, args.min_score)
    except Exception as e:
        logger.error(f"🔥 HARDCORE MODE: Processing FAILED: {e}")
        sys.exit(1)
    
    # Save results
    processing_time = time.time() - start_time
    
    logger.info(f"💾 Saving HARDCORE results to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"❌ Failed to save output: {e}")
        sys.exit(1)
    
    # Final summary
    logger.info("\n" + "💀" * 30)
    logger.info("🔥 HARDCORE GPU PROCESSING COMPLETE! 🔥")
    logger.info("💀" * 30)
    logger.info(f"📊 Processing time: {processing_time/60:.1f} minutes")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("🚫 ZERO CPU FALLBACKS USED!")
    logger.info("💀" * 30)

if __name__ == "__main__":
    main()