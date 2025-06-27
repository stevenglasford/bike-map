#!/usr/bin/env python3
"""
High-Performance Multi-GPU Video Analyzer
=========================================

Optimized for maximum hardware utilization:
- Multi-GPU support (both RTX 5060 Ti GPUs)
- Batch processing for YOLO inference
- Parallel video processing
- Memory-efficient frame caching
- Multi-threaded video decoding

Hardware targets:
- 2x NVIDIA RTX 5060 Ti (16GB VRAM each)
- 128GB RAM
- 16-core AMD Ryzen CPU
"""

import json
import argparse
import logging
import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict, deque
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
from functools import partial

# AI/ML imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch/YOLO required: pip install torch ultralytics")
    sys.exit(1)

# Audio analysis
try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# GPS processing
try:
    import gpxpy
    from geopy.distance import distance as geopy_distance
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

# Performance monitoring
import psutil
import GPUtil

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameBatchDataset(Dataset):
    """Dataset for batch processing video frames"""
    
    def __init__(self, frames: List[np.ndarray]):
        self.frames = frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize for YOLO (640x640 is optimal)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        return torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

class GPUManager:
    """Manages multi-GPU operations and load balancing"""
    
    def __init__(self):
        self.available_gpus = self.detect_gpus()
        self.gpu_loads = {gpu_id: 0 for gpu_id in self.available_gpus}
        self.models = {}
        self.lock = threading.Lock()
        
        logger.info(f"🎮 Detected {len(self.available_gpus)} GPUs: {self.available_gpus}")
        
    def detect_gpus(self) -> List[int]:
        """Detect available CUDA GPUs"""
        if not torch.cuda.is_available():
            logger.error("❌ CUDA not available")
            return []
        
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            try:
                # Test GPU availability
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                gpus.append(i)
            except Exception as e:
                logger.warning(f"⚠️ GPU {i} not available: {e}")
        
        return gpus
    
    def get_least_loaded_gpu(self) -> int:
        """Get GPU ID with lowest current load"""
        with self.lock:
            return min(self.gpu_loads.keys(), key=lambda k: self.gpu_loads[k])
    
    def increment_gpu_load(self, gpu_id: int):
        """Increment load counter for GPU"""
        with self.lock:
            self.gpu_loads[gpu_id] += 1
    
    def decrement_gpu_load(self, gpu_id: int):
        """Decrement load counter for GPU"""
        with self.lock:
            self.gpu_loads[gpu_id] = max(0, self.gpu_loads[gpu_id] - 1)
    
    def load_model_on_gpu(self, gpu_id: int, model_path: str) -> YOLO:
        """Load YOLO model on specific GPU"""
        if gpu_id not in self.models:
            torch.cuda.set_device(gpu_id)
            model = YOLO(model_path)
            model.to(f'cuda:{gpu_id}')
            self.models[gpu_id] = model
            logger.info(f"✅ Loaded YOLO model on GPU {gpu_id}")
        
        return self.models[gpu_id]
    
    def get_gpu_stats(self) -> Dict:
        """Get current GPU utilization stats"""
        stats = {}
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.id in self.available_gpus:
                    stats[gpu.id] = {
                        'utilization': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    }
        except Exception as e:
            logger.debug(f"Could not get GPU stats: {e}")
        
        return stats

class HighPerformanceVideoProcessor:
    """High-performance video processor with multi-GPU support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.setup_directories()
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager()
        if not self.gpu_manager.available_gpus:
            raise RuntimeError("❌ No CUDA GPUs available")
        
        # Load models on all GPUs
        for gpu_id in self.gpu_manager.available_gpus:
            self.gpu_manager.load_model_on_gpu(gpu_id, config.get('yolo_model', 'yolo11x.pt'))
        
        # Performance settings
        self.batch_size = config.get('batch_size', 32)
        self.max_workers = min(config.get('max_workers', 8), len(self.gpu_manager.available_gpus) * 4)
        self.frame_cache_size = config.get('frame_cache_size', 1000)  # Frames to cache in memory
        self.memory_limit_gb = config.get('memory_limit_gb', 64)  # Use 64GB of your 128GB
        
        # Statistics
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_detections': 0,
            'total_frames_processed': 0,
            'processing_times': [],
            'gpu_usage': defaultdict(list)
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"🚀 High-performance processor initialized:")
        logger.info(f"   GPUs: {len(self.gpu_manager.available_gpus)}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Memory limit: {self.memory_limit_gb}GB")
    
    def setup_directories(self):
        """Create output directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        subdirs = ['object_tracking', 'stoplight_detection', 'traffic_counting', 
                  'scene_complexity', 'audio_analysis', 'processing_reports']
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def monitor_system_resources(self):
        """Monitor and log system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_stats = self.gpu_manager.get_gpu_stats()
        
        logger.info(f"📊 System Status:")
        logger.info(f"   CPU: {cpu_percent:.1f}% ({psutil.cpu_count()} cores)")
        logger.info(f"   RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        for gpu_id, stats in gpu_stats.items():
            logger.info(f"   GPU {gpu_id}: {stats['utilization']:.1f}% util, {stats['memory_percent']:.1f}% mem ({stats['memory_used']}MB/{stats['memory_total']}MB)")
    
    def load_matcher50_results(self, results_path: str) -> Dict[str, Any]:
        """Load matcher50.py results with quality filtering"""
        logger.info(f"📖 Loading matcher50 results: {results_path}")
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        if 'results' in data:
            results = data['results']
        else:
            results = data
        
        # Apply quality filtering
        filtered_results = self.filter_high_quality_matches(results)
        
        logger.info(f"🔍 Filtered to {len(filtered_results)} high-quality video matches")
        return filtered_results
    
    def filter_high_quality_matches(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter for high-quality matches to maximize processing efficiency"""
        min_score = self.config.get('min_score', 0.6)  # Higher default for performance
        quality_threshold = self.config.get('min_quality', 'good')
        
        quality_map = {'excellent': 5, 'very_good': 4, 'good': 3, 'fair': 2, 'poor': 1}
        min_quality_level = quality_map.get(quality_threshold, 3)
        
        filtered = {}
        
        for video_path, video_data in results.items():
            if 'matches' not in video_data or not video_data['matches']:
                continue
            
            # Get best match
            best_match = video_data['matches'][0]
            score = best_match.get('combined_score', 0)
            quality = best_match.get('quality', 'poor')
            quality_level = quality_map.get(quality, 0)
            
            if score >= min_score and quality_level >= min_quality_level:
                filtered[video_path] = {'matches': [best_match]}
        
        reduction = ((len(results) - len(filtered)) / len(results) * 100) if results else 0
        logger.info(f"🎯 Quality filter: {len(results)} → {len(filtered)} videos ({reduction:.1f}% reduction)")
        
        return filtered
    
    def extract_frames_batch(self, video_path: str, max_frames: int = None) -> Tuple[List[np.ndarray], Dict]:
        """Extract frames from video in optimized batches"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {}
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'fps': fps, 'frame_count': frame_count,
            'width': width, 'height': height,
            'duration': frame_count / fps if fps > 0 else 0,
            'is_360': 1.8 <= (width/height) <= 2.2
        }
        
        # Memory management - limit frames based on available memory
        if max_frames is None:
            # Estimate memory usage per frame
            frame_size_mb = (width * height * 3) / (1024 * 1024)  # RGB
            max_memory_frames = int((self.memory_limit_gb * 1024) / frame_size_mb)
            max_frames = min(frame_count, max_memory_frames, 10000)  # Cap at 10k frames
        
        frames = []
        frame_indices = []
        
        # Smart frame sampling for very long videos
        if frame_count > max_frames:
            step = frame_count // max_frames
            frame_numbers = list(range(0, frame_count, step))[:max_frames]
        else:
            frame_numbers = list(range(frame_count))
        
        for frame_idx in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_indices.append(frame_idx)
            
            # Memory check
            if len(frames) % 1000 == 0:
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 80:  # Stop if memory usage too high
                    logger.warning(f"⚠️ High memory usage ({memory_usage:.1f}%), stopping frame extraction")
                    break
        
        cap.release()
        
        video_info['extracted_frames'] = len(frames)
        video_info['frame_indices'] = frame_indices
        video_info['sampling_rate'] = len(frame_numbers) / frame_count if frame_count > 0 else 0
        
        logger.info(f"📹 Extracted {len(frames)} frames from {Path(video_path).name}")
        return frames, video_info
    
    def process_frames_batch_gpu(self, frames: List[np.ndarray], gpu_id: int) -> List[Dict]:
        """Process batch of frames on specific GPU"""
        if not frames:
            return []
        
        try:
            # Get model for this GPU
            model = self.gpu_manager.models[gpu_id]
            
            # Track GPU usage
            self.gpu_manager.increment_gpu_load(gpu_id)
            
            # Batch processing
            torch.cuda.set_device(gpu_id)
            
            results = []
            batch_size = min(self.batch_size, len(frames))
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Process batch
                with torch.cuda.device(gpu_id):
                    detections_batch = model(batch_frames, verbose=False)
                
                # Extract results
                for j, detections in enumerate(detections_batch):
                    frame_idx = i + j
                    frame_results = self.extract_detections_from_results(detections, frame_idx)
                    results.append(frame_results)
                
                self.stats['total_frames_processed'] += len(batch_frames)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} processing error: {e}")
            return []
        finally:
            self.gpu_manager.decrement_gpu_load(gpu_id)
    
    def extract_detections_from_results(self, detections, frame_idx: int) -> Dict:
        """Extract detection data from YOLO results"""
        results = {
            'frame_idx': frame_idx,
            'detections': [],
            'object_counts': defaultdict(int)
        }
        
        if detections.boxes is None or len(detections.boxes) == 0:
            return results
        
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy().astype(int)
        confidences = detections.boxes.conf.cpu().numpy()
        
        class_names = self.gpu_manager.models[list(self.gpu_manager.models.keys())[0]].names
        
        for box, cls, conf in zip(boxes, classes, confidences):
            if conf < self.config.get('confidence_threshold', 0.3):
                continue
            
            obj_class = class_names.get(cls, 'unknown')
            
            detection = {
                'bbox': box.tolist(),
                'class': obj_class,
                'confidence': float(conf),
                'class_id': int(cls)
            }
            
            results['detections'].append(detection)
            results['object_counts'][obj_class] += 1
            self.stats['total_detections'] += 1
        
        return results
    
    def process_video_parallel(self, video_path: str, match_info: Dict, gps_df: pd.DataFrame) -> Dict:
        """Process single video with parallel GPU processing"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 Processing {video_name} with parallel GPUs")
        
        start_time = time.time()
        
        try:
            # Extract frames
            frames, video_info = self.extract_frames_batch(video_path)
            if not frames:
                return {'status': 'failed', 'error': 'No frames extracted'}
            
            logger.info(f"🔄 Processing {len(frames)} frames across {len(self.gpu_manager.available_gpus)} GPUs")
            
            # Split frames across GPUs
            frames_per_gpu = len(frames) // len(self.gpu_manager.available_gpus)
            gpu_tasks = []
            
            for i, gpu_id in enumerate(self.gpu_manager.available_gpus):
                start_idx = i * frames_per_gpu
                end_idx = start_idx + frames_per_gpu if i < len(self.gpu_manager.available_gpus) - 1 else len(frames)
                gpu_frames = frames[start_idx:end_idx]
                
                if gpu_frames:
                    task = self.executor.submit(self.process_frames_batch_gpu, gpu_frames, gpu_id)
                    gpu_tasks.append((task, start_idx, gpu_id))
            
            # Collect results
            all_detections = []
            for task, start_idx, gpu_id in gpu_tasks:
                try:
                    gpu_results = task.result(timeout=300)  # 5 minute timeout
                    for result in gpu_results:
                        result['frame_idx'] += start_idx  # Adjust frame index
                        all_detections.append(result)
                except Exception as e:
                    logger.error(f"❌ GPU {gpu_id} task failed: {e}")
            
            # Process detections with GPS data
            final_results = self.merge_detections_with_gps(all_detections, gps_df, video_info)
            
            # Audio analysis (if enabled)
            if AUDIO_AVAILABLE and self.config.get('analyze_audio', True):
                final_results['audio_analysis'] = self.analyze_audio_fast(video_path, gps_df)
            
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['processed_videos'] += 1
            
            # Performance metrics
            fps_processed = len(frames) / processing_time if processing_time > 0 else 0
            
            logger.info(f"✅ Completed {video_name}:")
            logger.info(f"   Time: {processing_time:.2f}s")
            logger.info(f"   FPS: {fps_processed:.1f}")
            logger.info(f"   Detections: {sum(len(d['detections']) for d in all_detections)}")
            
            return {
                'status': 'success',
                'video_info': video_info,
                'processing_time': processing_time,
                'fps_processed': fps_processed,
                'results': final_results
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {video_path}: {e}")
            self.stats['failed_videos'] += 1
            return {'status': 'failed', 'error': str(e)}
    
    def merge_detections_with_gps(self, all_detections: List[Dict], gps_df: pd.DataFrame, 
                                 video_info: Dict) -> Dict:
        """Merge detection results with GPS data"""
        fps = video_info['fps']
        frame_indices = video_info.get('frame_indices', [])
        
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int),
            'scene_complexity': []
        }
        
        for detection_result in all_detections:
            frame_idx = detection_result['frame_idx']
            
            # Calculate timestamp
            if frame_idx < len(frame_indices):
                actual_frame_number = frame_indices[frame_idx]
                second = int(actual_frame_number / fps) if fps > 0 else frame_idx
            else:
                second = frame_idx
            
            # Get GPS data
            gps_row = gps_df[gps_df['second'] == second] if not gps_df.empty else pd.DataFrame()
            gps_data = gps_row.iloc[0] if not gps_row.empty else pd.Series()
            
            # Process each detection
            for detection in detection_result['detections']:
                detection_record = {
                    'frame_second': second,
                    'object_class': detection['class'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'lat': float(gps_data.get('lat', 0)),
                    'lon': float(gps_data.get('lon', 0)),
                    'gps_speed': float(gps_data.get('speed_mph', 0)),
                    'gps_time': str(gps_data.get('gpx_time', '')),
                    'video_type': '360°' if video_info.get('is_360', False) else 'flat'
                }
                
                results['object_tracking'].append(detection_record)
                
                # Traffic light processing
                if detection['class'] == 'traffic light':
                    stoplight_record = detection_record.copy()
                    stoplight_record['stoplight_color'] = 'detected'  # Could add color detection here
                    results['stoplight_detection'].append(stoplight_record)
            
            # Count objects
            for obj_class, count in detection_result['object_counts'].items():
                results['traffic_counting'][obj_class] += count
        
        return results
    
    def analyze_audio_fast(self, video_path: str, gps_df: pd.DataFrame) -> List[Dict]:
        """Fast audio analysis using downsampling"""
        try:
            # Load with lower sample rate for speed
            y, sr = librosa.load(video_path, sr=22050)  # Lower sample rate
            
            # Downsample further for very long audio
            if len(y) > sr * 300:  # If longer than 5 minutes
                y = y[::2]  # Downsample by factor of 2
                sr = sr // 2
            
            hop_length = sr * 2  # 2-second windows for speed
            
            # Basic audio features
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            audio_results = []
            for i, rms_val in enumerate(rms):
                second = i * 2  # 2-second intervals
                
                gps_row = gps_df[gps_df['second'] == second] if not gps_df.empty else pd.DataFrame()
                gps_data = gps_row.iloc[0] if not gps_row.empty else pd.Series()
                
                audio_results.append({
                    'second': second,
                    'noise_level': float(rms_val * 100),
                    'lat': float(gps_data.get('lat', 0)),
                    'lon': float(gps_data.get('lon', 0))
                })
            
            return audio_results
            
        except Exception as e:
            logger.warning(f"⚠️ Fast audio analysis failed: {e}")
            return []
    
    def save_results_optimized(self, results: Dict, video_name: str):
        """Save results with optimized I/O"""
        save_tasks = []
        
        # Object tracking
        if results['object_tracking']:
            df = pd.DataFrame(results['object_tracking'])
            output_file = self.output_dir / 'object_tracking' / f"{video_name}_tracking.csv"
            save_task = self.executor.submit(df.to_csv, output_file, index=False)
            save_tasks.append(('tracking', save_task))
        
        # Stoplight detection
        if results['stoplight_detection']:
            df = pd.DataFrame(results['stoplight_detection'])
            output_file = self.output_dir / 'stoplight_detection' / f"{video_name}_stoplights.csv"
            save_task = self.executor.submit(df.to_csv, output_file, index=False)
            save_tasks.append(('stoplights', save_task))
        
        # Traffic counting
        if results['traffic_counting']:
            counting_data = [
                {'video_name': video_name, 'object_type': obj_type, 'total_count': count}
                for obj_type, count in results['traffic_counting'].items()
            ]
            df = pd.DataFrame(counting_data)
            output_file = self.output_dir / 'traffic_counting' / f"{video_name}_counts.csv"
            save_task = self.executor.submit(df.to_csv, output_file, index=False)
            save_tasks.append(('counting', save_task))
        
        # Wait for all saves to complete
        for save_type, task in save_tasks:
            try:
                task.result(timeout=30)
            except Exception as e:
                logger.warning(f"⚠️ Failed to save {save_type} for {video_name}: {e}")
    
    def process_videos_high_performance(self, video_matches: Dict[str, Any]):
        """Process all videos with maximum performance"""
        total_videos = len(video_matches)
        logger.info(f"🚀 Starting high-performance processing of {total_videos} videos")
        
        # Start resource monitoring
        monitor_thread = threading.Thread(target=self.periodic_monitoring, daemon=True)
        monitor_thread.start()
        
        # Process videos with parallel execution
        max_concurrent = min(self.config.get('max_concurrent_videos', 4), len(self.gpu_manager.available_gpus))
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as video_executor:
            future_to_video = {}
            
            for video_path, match_info in video_matches.items():
                # Load GPS data
                if match_info and 'matches' in match_info and match_info['matches']:
                    gps_path = match_info['matches'][0]['path']
                    gps_df = self.load_gps_data(gps_path)
                else:
                    gps_df = pd.DataFrame()
                
                # Submit video processing task
                future = video_executor.submit(self.process_video_parallel, video_path, match_info, gps_df)
                future_to_video[future] = (video_path, match_info)
            
            # Process completed videos
            completed = 0
            for future in as_completed(future_to_video):
                video_path, match_info = future_to_video[future]
                video_name = Path(video_path).stem
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        # Save results asynchronously
                        self.save_results_optimized(result['results'], video_name)
                        completed += 1
                        
                        progress = (completed / total_videos) * 100
                        logger.info(f"📊 Progress: {progress:.1f}% ({completed}/{total_videos}) - {video_name}")
                    else:
                        logger.error(f"❌ Failed: {video_name} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception processing {video_name}: {e}")
        
        # Final performance report
        self.generate_performance_report()
    
    def periodic_monitoring(self):
        """Periodically monitor system resources"""
        while True:
            time.sleep(30)  # Monitor every 30 seconds
            self.monitor_system_resources()
    
    def load_gps_data(self, gps_path: str) -> pd.DataFrame:
        """Load GPS data efficiently"""
        try:
            if gps_path.endswith('.csv'):
                df = pd.read_csv(gps_path)
                if 'long' in df.columns:
                    df['lon'] = df['long']
                return df
            elif gps_path.endswith('.gpx') and GPS_AVAILABLE:
                return self.parse_gpx_fast(gps_path)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"⚠️ GPS loading failed for {gps_path}: {e}")
            return pd.DataFrame()
    
    def parse_gpx_fast(self, gpx_path: str) -> pd.DataFrame:
        """Fast GPX parsing"""
        with open(gpx_path, 'r') as f:
            gpx = gpxpy.parse(f)
        
        records = []
        for track in gpx.tracks:
            for segment in track.segments:
                for i, point in enumerate(segment.points):
                    records.append({
                        'second': i,
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'gpx_time': point.time
                    })
        
        return pd.DataFrame(records)
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        total_time = sum(self.stats['processing_times'])
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        report = {
            'performance_summary': {
                'total_videos_processed': self.stats['processed_videos'],
                'total_videos_failed': self.stats['failed_videos'],
                'total_processing_time': total_time,
                'average_processing_time': avg_time,
                'total_frames_processed': self.stats['total_frames_processed'],
                'total_detections': self.stats['total_detections'],
                'videos_per_hour': (self.stats['processed_videos'] / (total_time / 3600)) if total_time > 0 else 0,
                'frames_per_second': (self.stats['total_frames_processed'] / total_time) if total_time > 0 else 0,
                'detections_per_video': (self.stats['total_detections'] / self.stats['processed_videos']) if self.stats['processed_videos'] > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'hardware_utilization': {
                'gpus_used': len(self.gpu_manager.available_gpus),
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'memory_limit_gb': self.memory_limit_gb
            },
            'gpu_stats': self.gpu_manager.get_gpu_stats()
        }
        
        report_path = self.output_dir / 'processing_reports' / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"🏁 High-Performance Processing Complete!")
        logger.info(f"   📊 Videos processed: {self.stats['processed_videos']}")
        logger.info(f"   🎯 Success rate: {report['performance_summary'].get('videos_per_hour', 0):.1f} videos/hour")
        logger.info(f"   🔥 Frames/sec: {report['performance_summary'].get('frames_per_second', 0):.1f}")
        logger.info(f"   📄 Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance Multi-GPU Video Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path to matcher50.py results JSON')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--yolo-model', default='yolo11x.pt',
                       help='YOLO model path')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU processing (default: 32)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum worker threads (default: 8)')
    parser.add_argument('--max-concurrent-videos', type=int, default=4,
                       help='Maximum concurrent videos (default: 4)')
    parser.add_argument('--memory-limit-gb', type=int, default=64,
                       help='Memory limit in GB (default: 64)')
    parser.add_argument('--min-score', type=float, default=0.6,
                       help='Minimum match score (default: 0.6)')
    parser.add_argument('--min-quality', default='good',
                       help='Minimum match quality (default: good)')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='YOLO confidence threshold (default: 0.3)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Skip audio analysis for maximum speed')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Configuration
    config = {
        'output_dir': args.output,
        'yolo_model': args.yolo_model,
        'batch_size': args.batch_size,
        'max_workers': args.max_workers,
        'max_concurrent_videos': args.max_concurrent_videos,
        'memory_limit_gb': args.memory_limit_gb,
        'min_score': args.min_score,
        'min_quality': args.min_quality,
        'confidence_threshold': args.confidence_threshold,
        'analyze_audio': not args.no_audio
    }
    
    logger.info("🚀 Initializing High-Performance Video Processor...")
    logger.info(f"   Target hardware: 2x RTX 5060 Ti, 128GB RAM, 16-core CPU")
    
    # Initialize processor
    try:
        processor = HighPerformanceVideoProcessor(config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Load matcher50 results
    try:
        video_matches = processor.load_matcher50_results(args.input)
    except Exception as e:
        logger.error(f"❌ Failed to load results: {e}")
        sys.exit(1)
    
    if not video_matches:
        logger.error("❌ No videos to process after filtering")
        sys.exit(1)
    
    # Start processing
    logger.info(f"🎬 Starting processing of {len(video_matches)} videos...")
    start_time = time.time()
    
    processor.process_videos_high_performance(video_matches)
    
    total_time = time.time() - start_time
    logger.info(f"🏁 Total processing time: {total_time:.2f} seconds")
    logger.info(f"🎯 Average time per video: {total_time/len(video_matches):.2f} seconds")


if __name__ == "__main__":
    main()

