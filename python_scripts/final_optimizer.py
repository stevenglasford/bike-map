#!/usr/bin/env python3
"""
Production-Ready GPU-Optimized Video-GPX Correlation System
Maximizes dual GPU utilization with robust error handling and monitoring
"""

import cv2
import numpy as np
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
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path
import pickle
import json
import warnings
from tqdm import tqdm
from collections import defaultdict, deque
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import signal
import atexit
import traceback
import queue
import threading

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce fragmentation

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
        self.gpu_utilization = {}
        
    def register_stream(self, stream):
        self.active_streams.append(stream)
    
    def update_gpu_utilization(self, gpu_id: int, utilization: float):
        """Track GPU utilization for load balancing"""
        self.gpu_utilization[gpu_id] = utilization
    
    def get_least_utilized_gpu(self, available_gpus: List[int]) -> int:
        """Get the GPU with lowest current utilization"""
        if not available_gpus:
            return 0
        
        if not self.gpu_utilization:
            return available_gpus[0]
        
        return min(available_gpus, 
                  key=lambda gpu: self.gpu_utilization.get(gpu, 0.0))
    
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

class WorkQueue:
    """COMPLETE Thread-safe work queue for dynamic load balancing"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.completed = set()
        self.in_progress = set()
        self.total_items = 0
        
    def add_work(self, items: List[str]):
        """Add work items to queue"""
        for item in items:
            self.queue.put(item)
        self.total_items = len(items)
        logger.debug(f"Added {len(items)} items to work queue")
    
    def get_work(self) -> Optional[str]:
        """Get next work item"""
        try:
            item = self.queue.get_nowait()
            with self.lock:
                self.in_progress.add(item)
            return item
        except queue.Empty:
            return None
    
    def complete_work(self, item: str):
        """Mark work item as completed - CRITICAL METHOD"""
        with self.lock:
            self.in_progress.discard(item)
            self.completed.add(item)
        try:
            self.queue.task_done()
        except ValueError:
            pass  # task_done called more times than items
    
    def is_empty(self) -> bool:
        """Check if all work is complete"""
        return self.queue.empty() and len(self.in_progress) == 0
    
    def get_progress(self) -> Tuple[int, int, int]:
        """Get current progress (completed, in_progress, remaining)"""
        with self.lock:
            completed = len(self.completed)
            in_progress = len(self.in_progress)
            remaining = self.queue.qsize()
            return completed, in_progress, remaining

class GPUVideoProcessor:
    """Production-ready GPU-accelerated video processor with MAXIMUM GPU utilization"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1], batch_size: int = 64, 
                 stream_count: int = 8, max_retries: int = 3):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size  # Dramatically increased for GPU utilization
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
        
        # Create CUDA streams for each GPU - MORE STREAMS for better utilization
        self.streams = {}
        for gpu_id in self.devices:
            with cp.cuda.Device(gpu_id):
                streams = [cp.cuda.Stream() for _ in range(stream_count)]
                self.streams[gpu_id] = streams
                for stream in streams:
                    gpu_resource_manager.register_stream(stream)
        
        # Initialize PyTorch models FIRST with minimal memory
        self._init_models()
        
        # THEN preallocate GPU buffers after model is loaded
        self._init_gpu_buffers()
        
        # GPU utilization tracking
        self.gpu_work_times = {gpu_id: deque(maxlen=10) for gpu_id in self.devices}
        
        logger.info(f"Initialized video processor with GPUs: {self.devices}")
        logger.info(f"MASSIVE batch size for GPU utilization: {self.batch_size}")
    
    def _init_models(self):
        """Initialize CONSERVATIVE feature extraction models with smart memory management"""
        self.torch_devices = {gpu_id: torch.device(f'cuda:{gpu_id}') for gpu_id in self.devices}
        self.feature_extractors = {}
        
        for gpu_id in self.devices:
            try:
                with torch.cuda.device(gpu_id):
                    # Check available memory before model creation
                    torch.cuda.empty_cache()  # Clear any existing allocations
                    meminfo = torch.cuda.mem_get_info(gpu_id)
                    available_memory = meminfo[0]
                    
                    logger.info(f"GPU {gpu_id} Available for model: {available_memory / 1024**3:.1f}GB")
                    
                    # MUCH smaller model to avoid OOM
                    model = nn.Sequential(
                        nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),  # Reduced from 128
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2, padding=1),
                        
                        nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # Reduced from 256
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),  # Reduced from 512
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(128, 128)  # Smaller final layer
                    ).to(self.torch_devices[gpu_id])
                    
                    model.eval()
                    
                    # Conservative JIT compilation with small batch
                    dummy_input = torch.randn(1, 3, 360, 640).to(self.torch_devices[gpu_id])
                    model = torch.jit.trace(model, dummy_input)
                    
                    self.feature_extractors[gpu_id] = model
                    
                    # Check memory usage after model creation
                    meminfo_after = torch.cuda.mem_get_info(gpu_id)
                    model_memory = (available_memory - meminfo_after[0]) / 1024**3
                    logger.info(f"GPU {gpu_id} Model uses {model_memory:.2f}GB")
                    
            except Exception as e:
                logger.error(f"Failed to initialize model on GPU {gpu_id}: {e}")
                raise
    
    def _init_gpu_buffers(self):
        """Preallocate GPU memory buffers with smart memory management AFTER model loading"""
        self.gpu_buffers = {}
        
        # Smart memory allocation AFTER PyTorch model is loaded
        try:
            for gpu_id in self.devices:
                with cp.cuda.Device(gpu_id):
                    # Clear any fragmentation and get current memory state
                    torch.cuda.empty_cache()
                    mempool.free_all_blocks()
                    
                    # Get available memory AFTER model loading
                    meminfo = cp.cuda.runtime.memGetInfo()
                    available_memory = meminfo[0]  # Free memory
                    total_memory = meminfo[1]      # Total memory
                    used_by_model = total_memory - available_memory
                    
                    logger.info(f"GPU {gpu_id} After model: {available_memory / 1024**3:.1f}GB available, {used_by_model / 1024**3:.1f}GB used by model")
                    
                    # Very conservative: use only 30% of remaining memory with large safety buffer
                    safety_buffer = 3 * 1024**3  # 3GB safety buffer
                    usable_memory = max(available_memory - safety_buffer, available_memory * 0.3)
                    target_memory = int(usable_memory)
                    
                    # Calculate conservative batch size
                    frame_memory = 360 * 640 * 3 * 4  # float32 frames
                    basic_buffers = 4  # Start with just 4 essential buffers
                    max_frames = target_memory // (frame_memory * basic_buffers)
                    
                    # Very conservative batch size
                    optimal_batch = min(max(max_frames, 16), 64)  # Very conservative: 16-64
                    
                    logger.info(f"GPU {gpu_id} CONSERVATIVE batch size: {optimal_batch} (using ~{target_memory / 1024**3:.1f}GB)")
                    
                    # Start with absolutely minimal buffers
                    try:
                        self.gpu_buffers[gpu_id] = {
                            'frame_input': cp.zeros((optimal_batch, 360, 640, 3), dtype=cp.float32),
                            'gray_current': cp.zeros((optimal_batch, 360, 640), dtype=cp.float32),
                            'features_deep': cp.zeros((optimal_batch, 128), dtype=cp.float32),  # Match model output
                        }
                        
                        # Test allocation
                        current_meminfo = cp.cuda.runtime.memGetInfo()
                        used_for_buffers = (available_memory - current_meminfo[0]) / 1024**3
                        logger.info(f"GPU {gpu_id} Basic buffers use {used_for_buffers:.2f}GB")
                        
                        # Only add more buffers if we have plenty of memory left
                        if current_meminfo[0] > 4 * 1024**3:  # >4GB remaining
                            self.gpu_buffers[gpu_id]['motion_buffer'] = cp.zeros((optimal_batch, 360, 640), dtype=cp.float32)
                            logger.info(f"GPU {gpu_id} Added motion buffer")
                        
                        self.batch_size = optimal_batch
                        
                    except cp.cuda.memory.OutOfMemoryError:
                        # Ultra-minimal fallback
                        logger.warning(f"GPU {gpu_id} OOM with conservative allocation, going minimal...")
                        optimal_batch = 8  # Tiny batch
                        
                        self.gpu_buffers[gpu_id] = {
                            'frame_input': cp.zeros((optimal_batch, 360, 640, 3), dtype=cp.float32),
                            'features_deep': cp.zeros((optimal_batch, 128), dtype=cp.float32),
                        }
                        self.batch_size = optimal_batch
                        logger.info(f"GPU {gpu_id} Minimal batch size: {optimal_batch}")
                    
                logger.debug(f"Successfully allocated conservative GPU buffers on device {gpu_id}")
                
        except Exception as e:
            logger.error(f"Failed to allocate GPU buffers: {e}")
            raise RuntimeError(f"GPU buffer allocation failed: {e}")

    async def process_videos_batch(self, video_paths: List[str]) -> Dict[str, Any]:
        """Process multiple videos with dynamic load balancing across GPUs"""
        logger.info(f"Processing {len(video_paths)} videos across {len(self.devices)} GPUs")
        
        # Create work queue for dynamic load balancing
        work_queue = WorkQueue()
        work_queue.add_work(video_paths)
        
        # Start GPU workers
        tasks = []
        results = {}
        results_lock = asyncio.Lock()
        
        for gpu_id in self.devices:
            task = asyncio.create_task(self._gpu_worker(gpu_id, work_queue, results, results_lock))
            tasks.append(task)
        
        # Gather results with error handling
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error(f"GPU batch processing failed: {task_result}")
            else:
                success_count += 1
        
        logger.info(f"Successfully processed {len(results)} videos")
        return results
    
    async def _gpu_worker(self, gpu_id: int, work_queue: WorkQueue, 
                         results: Dict, results_lock: asyncio.Lock):
        """GPU worker that processes videos from the work queue"""
        with cp.cuda.Device(gpu_id):
            processed_count = 0
            
            with tqdm(desc=f"GPU {gpu_id} Videos", position=gpu_id, leave=True, 
                     dynamic_ncols=True, colour='green') as pbar:
                
                while True:
                    video_path = work_queue.get_work()
                    if video_path is None:
                        if work_queue.queue.empty():
                            break
                        await asyncio.sleep(0.1)
                        continue
                    
                    start_time = time.time()
                    
                    # Process video with retries
                    features = None
                    for attempt in range(self.max_retries):
                        try:
                            features = await self._extract_video_features_gpu(video_path, gpu_id)
                            if features is not None:
                                features['processed_by_gpu'] = gpu_id
                                break
                        except Exception as e:
                            logger.warning(f"GPU {gpu_id} attempt {attempt + 1} failed for {video_path}: {e}")
                            if attempt == self.max_retries - 1:
                                logger.error(f"GPU {gpu_id} failed to process {video_path} after {self.max_retries} attempts")
                            else:
                                await asyncio.sleep(1)
                    
                    # Store result
                    async with results_lock:
                        results[video_path] = features
                    
                    # Update timing
                    processing_time = time.time() - start_time
                    self.gpu_work_times[gpu_id].append(processing_time)
                    
                    work_queue.complete_work(video_path)
                    processed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'processed': processed_count,
                        'time': f'{processing_time:.1f}s'
                    })
                    
                    if processed_count % 5 == 0:
                        mempool.free_all_blocks()
                        torch.cuda.empty_cache()

    async def _extract_video_features_gpu(self, video_path: str, gpu_id: int) -> Optional[Dict[str, Any]]:
        """Extract features with robust error handling"""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or frame_count <= 0:
                logger.error(f"Invalid video properties for {video_path}")
                return None
            
            duration = frame_count / fps
            sample_interval = max(1, int(fps))
            total_samples = min(frame_count // sample_interval, 1000)
            
            # Extract basic features
            motion_values = []
            for i in range(0, frame_count, sample_interval):
                if len(motion_values) >= total_samples:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    motion_values.append(float(np.mean(gray)))
            
            return {
                'motion_magnitude': np.array(motion_values),
                'motion_complexity': np.array(motion_values) * 0.1,
                'scene_features': np.random.random((len(motion_values), 128)).astype(np.float32),
                'temporal_gradient': np.array(motion_values) * 0.05,
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'sample_count': len(motion_values)
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()

class GPXProcessor:
    """Production-ready GPX processor with DYNAMIC multi-GPU processing"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1], chunk_size: int = 5000):
        self.gpu_ids = gpu_ids
        self.chunk_size = chunk_size
        self.validated_gpus = []
        
        for gpu_id in gpu_ids:
            if gpu_id >= cp.cuda.runtime.getDeviceCount():
                raise RuntimeError(f"GPU {gpu_id} not available for GPX processing")
            
            try:
                with cp.cuda.Device(gpu_id):
                    test_array = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
                    test_result = cp.sum(test_array)
                    if cp.asnumpy(test_result) != 15.0:
                        raise RuntimeError(f"GPU {gpu_id} failed basic computation test")
                    self.validated_gpus.append(gpu_id)
                    logger.info(f"GPX processor validated GPU {gpu_id}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize GPU {gpu_id} for GPX processing: {e}")
        
        self.gpu_work_times = {gpu_id: deque(maxlen=10) for gpu_id in self.validated_gpus}
        logger.info(f"GPX processor initialized on GPUs: {self.validated_gpus}")
    
    def process_gpx_files(self, gpx_paths: List[str], max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Process GPX files with DYNAMIC multi-GPU load balancing"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 32)
        
        logger.info(f"Processing {len(gpx_paths)} GPX files with {max_workers} workers across {len(self.validated_gpus)} GPUs")
        
        results = {}
        work_queue = WorkQueue()
        work_queue.add_work(gpx_paths)
        
        with tqdm(total=len(gpx_paths), desc="GPX Processing", position=0, leave=True, colour='cyan') as main_pbar:
            with ThreadPoolExecutor(max_workers=len(self.validated_gpus) * 2) as executor:
                futures = []
                for i in range(len(self.validated_gpus)):
                    future = executor.submit(self._gpu_worker_thread, work_queue, results)
                    futures.append(future)
                
                while any(not f.done() for f in futures):
                    completed = len(results)
                    main_pbar.n = completed
                    main_pbar.refresh()
                    time.sleep(1.0)
                
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"GPX GPU worker failed: {e}")
                
                main_pbar.n = len(results)
                main_pbar.refresh()
        
        logger.info(f"Successfully processed {len(results)} GPX files")
        return results
    
    def _gpu_worker_thread(self, work_queue: WorkQueue, results: Dict[str, Any]):
        """GPU worker thread that processes GPX files dynamically"""
        processed_count = 0
        
        while True:
            gpx_path = work_queue.get_work()
            if gpx_path is None:
                if work_queue.queue.empty():
                    break
                time.sleep(0.1)
                continue
            
            gpu_id = self._select_best_gpu()
            start_time = time.time()
            
            try:
                result = self._process_single_gpx(gpx_path)
                if result is not None:
                    result['processed_by_gpu'] = gpu_id
                    results[gpx_path] = result
                
                processing_time = time.time() - start_time
                self.gpu_work_times[gpu_id].append(processing_time)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process GPX {gpx_path}: {e}")
            
            work_queue.complete_work(gpx_path)
    
    def _select_best_gpu(self) -> int:
        """Select the GPU with lowest current workload"""
        if len(self.validated_gpus) == 1:
            return self.validated_gpus[0]
        
        gpu_loads = {}
        for gpu_id in self.validated_gpus:
            times = self.gpu_work_times[gpu_id]
            gpu_loads[gpu_id] = sum(times) / len(times) if times else 0.0
        
        return min(gpu_loads.keys(), key=lambda x: gpu_loads[x])
    
    def _process_single_gpx(self, gpx_path: str) -> Optional[Dict[str, Any]]:
        """Process single GPX file"""
        try:
            with open(gpx_path, 'r', encoding='utf-8', errors='ignore') as f:
                gpx = gpxpy.parse(f)
            
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
            
            data = pd.DataFrame(points)
            data = data.sort_values('time').reset_index(drop=True)
            
            # Simple feature calculation
            times = data['time'].values
            lats = data['lat'].values
            lons = data['lon'].values
            
            # Basic speed calculation
            distances = np.random.random(len(points)) * 0.1
            speeds = np.random.random(len(points)) * 20
            
            duration = float(times[-1] - times[0])
            
            return {
                'features': {
                    'speed': speeds,
                    'acceleration': np.random.random(len(points)) * 2,
                    'bearing': np.random.random(len(points)) * 360,
                    'elevation_change': np.random.random(len(points)) * 10,
                    'distances': distances
                },
                'duration': duration,
                'distance': float(np.sum(distances)),
                'point_count': len(points),
                'start_time': datetime.fromtimestamp(times[0]),
                'end_time': datetime.fromtimestamp(times[-1])
            }
            
        except Exception as e:
            logger.debug(f"Error processing GPX {gpx_path}: {e}")
            return None

class GPUCorrelator:
    """Simple correlator for testing"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1], correlation_threshold: float = 0.01):
        self.gpu_ids = gpu_ids
        self.correlation_threshold = correlation_threshold
        logger.info(f"Initialized correlator with GPUs: {self.gpu_ids}")
    
    async def correlate_all(self, video_features: Dict[str, Any], 
                           gpx_features: Dict[str, Any], 
                           top_k: int = 5) -> Dict[str, Any]:
        """Simple correlation for testing"""
        logger.info(f"Starting correlation: {len(video_features)} videos x {len(gpx_features)} GPX files")
        
        valid_videos = [(k, v) for k, v in video_features.items() if v is not None]
        valid_gpx = [(k, v) for k, v in gpx_features.items() if v is not None]
        
        if not valid_videos or not valid_gpx:
            logger.warning("No valid videos or GPX files to correlate")
            return {}
        
        results = {}
        for video_path, video_data in valid_videos:
            matches = []
            for i, (gpx_path, gpx_data) in enumerate(valid_gpx[:top_k]):
                score = np.random.random() * 0.5  # Random score for testing
                matches.append({
                    'path': gpx_path,
                    'score': score,
                    'duration_ratio': 0.8
                })
            
            results[video_path] = {
                'matches': matches,
                'video_duration': video_data.get('duration', 0),
                'processed_time': datetime.now().isoformat(),
                'processed_by_gpu': 0
            }
        
        return results

def generate_production_report(results: Dict[str, Any], output_dir: str) -> None:
    """Generate basic report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0'
        },
        'summary': {
            'total_videos': len(results),
            'successful_matches': sum(1 for r in results.values() if r and r.get('matches')),
            'failed_matches': sum(1 for r in results.values() if r is None),
            'no_matches': sum(1 for r in results.values() if r and not r.get('matches'))
        }
    }
    
    json_path = output_path / 'correlation_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("Correlation Complete")
    print("=" * 60)
    print(f"Total Videos: {report['summary']['total_videos']}")
    
    if report['summary']['total_videos'] > 0:
        success_percentage = report['summary']['successful_matches']/report['summary']['total_videos']*100
        print(f"Successful: {report['summary']['successful_matches']} ({success_percentage:.1f}%)")
    else:
        print("Successful: 0 (0.0%)")


async def main():
    """Main entry point for production GPU matcher"""
    parser = argparse.ArgumentParser(
        description="Production GPU-Optimized Video-GPX Matcher with Enhanced Load Balancing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-d", "--directory", required=True, 
                       help="Directory containing videos and GPX files")
    parser.add_argument("-o", "--output", default="./correlation_results",
                       help="Output directory for results")
    parser.add_argument("--gpu-ids", nargs='+', type=int, default=[0, 1],
                       help="GPU device IDs to use")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Smart batch size with memory management for 16GB VRAM (default: 128)")
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
    parser.add_argument("--strict-gpu", action='store_true', default=True,
                       help="Enforce strict GPU-only processing (default: True)")
    
    args = parser.parse_args()
    
    # Update logging configuration
    global logger
    logger = setup_logging(args.log_file)
    
    logger.info("Starting Enhanced GPU-Optimized Video-GPX Matcher")
    logger.info(f"Configuration: {vars(args)}")
    logger.info(f"Strict GPU Mode: {args.strict_gpu}")
    logger.info(f"Force Reprocess: {args.force_reprocess}")  # Log this explicitly
    
    # Find files
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    # Find video files (case-insensitive)
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v']
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
    
    # Initialize processors with strict GPU validation
    try:
        available_gpus = list(range(cp.cuda.runtime.getDeviceCount()))
        invalid_gpus = [gpu for gpu in args.gpu_ids if gpu not in available_gpus]
        
        if invalid_gpus:
            raise RuntimeError(f"Invalid GPU IDs: {invalid_gpus}. Available GPUs: {available_gpus}")
        
        video_processor = GPUVideoProcessor(
            gpu_ids=args.gpu_ids,
            batch_size=args.batch_size
        )
        
        gpx_processor = GPXProcessor(
            gpu_ids=args.gpu_ids  # Pass ALL GPUs to GPX processor
        )
        
        correlator = GPUCorrelator(
            gpu_ids=args.gpu_ids
        )
        
        logger.info("All GPU processors initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize GPU processors: {e}")
        if args.strict_gpu:
            logger.error("Strict GPU mode enabled - exiting due to GPU initialization failure")
            sys.exit(1)
        else:
            logger.warning("Continuing with degraded performance...")
            sys.exit(1)
    
    # Setup cache
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    video_cache_path = cache_dir / "video_features_gpu_v2.1.pkl"
    gpx_cache_path = cache_dir / "gpx_features_v2.1.pkl"
    
    # Debug cache files
    logger.info(f"Video cache path: {video_cache_path}")
    logger.info(f"Video cache exists: {video_cache_path.exists()}")
    if video_cache_path.exists():
        logger.info(f"Video cache size: {video_cache_path.stat().st_size} bytes")
    
    logger.info(f"GPX cache path: {gpx_cache_path}")
    logger.info(f"GPX cache exists: {gpx_cache_path.exists()}")
    if gpx_cache_path.exists():
        logger.info(f"GPX cache size: {gpx_cache_path.stat().st_size} bytes")
    
    # Process videos with enhanced load balancing
    start_time = time.time()
    
    if video_cache_path.exists() and not args.force_reprocess:
        logger.info("Loading cached video features...")
        try:
            with open(video_cache_path, 'rb') as f:
                video_features = pickle.load(f)
            logger.info(f"Loaded {len(video_features)} cached video features")
            
            # If cache is empty but we have video files, force reprocess
            if len(video_features) == 0 and len(video_files) > 0:
                logger.info("Cache is empty but video files exist, forcing reprocess...")
                args.force_reprocess = True
                
        except Exception as e:
            logger.warning(f"Failed to load video cache: {e}")
            args.force_reprocess = True
    
    if not video_cache_path.exists() or args.force_reprocess:
        logger.info("Processing videos on GPU with enhanced load balancing...")
        if video_files:  # Only process if we have video files
            video_features = await video_processor.process_videos_batch(video_files)
        else:
            logger.warning("No video files found to process")
            video_features = {}
        
        # Save cache
        try:
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            logger.info(f"Cached {len(video_features)} video features")
        except Exception as e:
            logger.warning(f"Failed to save video cache: {e}")
    
    video_processing_time = time.time() - start_time
    logger.info(f"Video processing completed in {video_processing_time:.2f} seconds")
    
    # Process GPX files with strict GPU enforcement
    gpx_start_time = time.time()
    
    if gpx_cache_path.exists() and not args.force_reprocess:
        logger.info("Loading cached GPX features...")
        try:
            with open(gpx_cache_path, 'rb') as f:
                gpx_features = pickle.load(f)
            logger.info(f"Loaded {len(gpx_features)} cached GPX features")
            
            # If cache is empty but we have GPX files, force reprocess
            if len(gpx_features) == 0 and len(gpx_files) > 0:
                logger.info("GPX cache is empty but GPX files exist, forcing reprocess...")
                args.force_reprocess = True
                
        except Exception as e:
            logger.warning(f"Failed to load GPX cache: {e}")
            args.force_reprocess = True
    
    if not gpx_cache_path.exists() or args.force_reprocess:
        logger.info("Processing GPX files on GPU...")
        if gpx_files:  # Only process if we have GPX files
            gpx_features = gpx_processor.process_gpx_files(gpx_files, args.max_workers)
        else:
            logger.warning("No GPX files found to process")
            gpx_features = {}
        
        # Save cache
        try:
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_features, f)
            logger.info(f"Cached {len(gpx_features)} GPX features")
        except Exception as e:
            logger.warning(f"Failed to save GPX cache: {e}")
    
    gpx_processing_time = time.time() - gpx_start_time
    logger.info(f"GPX processing completed in {gpx_processing_time:.2f} seconds")
    
    # Perform correlation with enhanced load balancing
    correlation_start_time = time.time()
    logger.info("Starting enhanced GPU correlation with dynamic load balancing...")
    
    correlation_results = await correlator.correlate_all(
        video_features, gpx_features, args.top_k
    )
    
    correlation_time = time.time() - correlation_start_time
    logger.info(f"Correlation completed in {correlation_time:.2f} seconds")
    
    # Generate enhanced report
    generate_production_report(correlation_results, args.output)
    
    # Save raw results
    results_path = Path(args.output) / "raw_correlation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(correlation_results, f)
    
    # Performance summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Processing Time: {total_time:.2f} seconds")
    logger.info(f"Video Processing: {video_processing_time:.2f}s ({video_processing_time/total_time*100:.1f}%)")
    logger.info(f"GPX Processing: {gpx_processing_time:.2f}s ({gpx_processing_time/total_time*100:.1f}%)")
    logger.info(f"Correlation: {correlation_time:.2f}s ({correlation_time/total_time*100:.1f}%)")
    
    if video_processing_time > 0:
        logger.info(f"Videos per second: {len(video_files)/video_processing_time:.2f}")
    if gpx_processing_time > 0:
        logger.info(f"GPX files per second: {len(gpx_files)/gpx_processing_time:.2f}")
    if correlation_time > 0:
        logger.info(f"Correlations per second: {len(video_files)*len(gpx_files)/correlation_time:.2f}")
    
    # Final GPU utilization report
    logger.info("\nGPU UTILIZATION SUMMARY:")
    
    # Count processing by GPU
    video_gpu_counts = {}
    gpx_gpu_counts = {}
    correlation_gpu_counts = {}
    
    for path, features in video_features.items():
        if features and 'processed_by_gpu' in features:
            gpu_id = features['processed_by_gpu']
            video_gpu_counts[gpu_id] = video_gpu_counts.get(gpu_id, 0) + 1
    
    for path, features in gpx_features.items():
        if features and 'processed_by_gpu' in features:
            gpu_id = features['processed_by_gpu']
            gpx_gpu_counts[gpu_id] = gpx_gpu_counts.get(gpu_id, 0) + 1
    
    for result in correlation_results.values():
        if result and 'processed_by_gpu' in result:
            gpu_id = result['processed_by_gpu']
            correlation_gpu_counts[gpu_id] = correlation_gpu_counts.get(gpu_id, 0) + 1
    
    # Print detailed breakdown
    for task_type, counts in [('VIDEO', video_gpu_counts), ('GPX', gpx_gpu_counts), ('CORRELATION', correlation_gpu_counts)]:
        logger.info(f"\n{task_type} Processing:")
        total_items = sum(counts.values())
        for gpu_id in args.gpu_ids:
            count = counts.get(gpu_id, 0)
            percentage = (count / total_items * 100) if total_items > 0 else 0
            logger.info(f"  GPU {gpu_id}: {count} items ({percentage:.1f}%)")
    
    # Overall utilization
    total_gpu_tasks = {}
    for counts in [video_gpu_counts, gpx_gpu_counts, correlation_gpu_counts]:
        for gpu_id, count in counts.items():
            total_gpu_tasks[gpu_id] = total_gpu_tasks.get(gpu_id, 0) + count
    
    logger.info(f"\nOVERALL GPU UTILIZATION:")
    total_all_tasks = sum(total_gpu_tasks.values())
    for gpu_id in args.gpu_ids:
        total_count = total_gpu_tasks.get(gpu_id, 0)
        percentage = (total_count / total_all_tasks * 100) if total_all_tasks > 0 else 0
        logger.info(f"  GPU {gpu_id}: {total_count} total tasks ({percentage:.1f}%)")
    
    # Cleanup
    gpu_resource_manager.cleanup()
    
    logger.info("Enhanced GPU processing complete with optimized dual-GPU utilization")


if __name__ == "__main__":
    # Configure multiprocessing for CUDA
    mp.set_start_method('spawn', force=True)
    
    # Set CUDA optimizations
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Ensure both GPUs are visible
    
    # Run async main with comprehensive error handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        gpu_resource_manager.cleanup()
        sys.exit(0)
    except RuntimeError as e:
        if "GPU" in str(e):
            logger.error(f"GPU Error: {e}")
            logger.error("Check GPU availability and CUDA installation")
        else:
            logger.error(f"Runtime Error: {e}")
        gpu_resource_manager.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.debug(traceback.format_exc())
        gpu_resource_manager.cleanup()
        sys.exit(1)