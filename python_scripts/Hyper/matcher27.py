#!/usr/bin/env python3
"""
Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script

Fixes for previous version:
- Fixed GPU load balancing for true multi-GPU utilization
- Fixed broadcast errors in GPX processing 
- Increased frame processing for better accuracy
- Improved parallel processing architecture
- Enhanced similarity computation methods
- Better temporal alignment and feature extraction
- Complete PowerSafe implementation with incremental saves and resume capability

Features:
- True multi-GPU parallel video processing
- Robust GPX feature extraction with proper array handling
- Advanced temporal correlation with DTW alignment
- Comprehensive similarity metrics
- Production-grade error handling and monitoring
- Scalable architecture for large datasets
- PowerSafe mode for long-running operations with resume capability

Usage:
    python enhanced_correlator.py -d /path/to/data --gpu_ids 0 1 --max_frames 1000
    python enhanced_correlator.py -d /path/to/data --parallel_videos 4 --debug
    python enhanced_correlator.py -d /path/to/data --powersafe --save_interval 5
"""

import cv2
import numpy as np
import gpxpy
import pandas as pd
import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from datetime import timedelta, datetime
import argparse
import os
import glob
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict, deque
import time
import warnings
import logging
from tqdm import tqdm
import gc
import asyncio
import aiofiles
from threading import Lock, Thread
import queue
import tempfile
import shutil
import sys
from typing import Dict, List, Tuple, Optional, Any
import psutil
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager

# Optional imports with fallbacks
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    max_frames: int = 1000
    target_size: Tuple[int, int] = (640, 360)
    sample_rate: float = 3.0
    parallel_videos: int = 4
    gpu_memory_fraction: float = 0.8
    motion_threshold: float = 0.01
    temporal_window: int = 10
    powersafe: bool = False
    save_interval: int = 5  # Save every N correlations in powersafe mode

class PowerSafeManager:
    """Power-safe processing manager with incremental saves and resume capability"""
    
    def __init__(self, cache_dir: Path, config: ProcessingConfig):
        self.cache_dir = cache_dir
        self.config = config
        self.db_path = cache_dir / "powersafe_progress.db"
        self.results_path = cache_dir / "incremental_results.json"
        self.last_scan_path = cache_dir / "last_file_scan.json"
        self.correlation_counter = 0
        self.pending_results = {}
        
        if config.powersafe:
            self._init_progress_db()
            logger.info("PowerSafe mode enabled - incremental saving activated")
    
    def _init_progress_db(self):
        """Initialize progress tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_progress (
                    video_path TEXT PRIMARY KEY,
                    status TEXT,  -- 'pending', 'processing', 'completed', 'failed'
                    processed_at TIMESTAMP,
                    feature_extraction_done BOOLEAN DEFAULT FALSE,
                    correlation_done BOOLEAN DEFAULT FALSE,
                    best_match_score REAL DEFAULT 0.0,
                    best_match_path TEXT,
                    file_mtime REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpx_progress (
                    gpx_path TEXT PRIMARY KEY,
                    status TEXT,
                    processed_at TIMESTAMP,
                    feature_extraction_done BOOLEAN DEFAULT FALSE,
                    file_mtime REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correlation_progress (
                    video_path TEXT,
                    gpx_path TEXT,
                    correlation_score REAL,
                    correlation_details TEXT,  -- JSON string
                    processed_at TIMESTAMP,
                    PRIMARY KEY (video_path, gpx_path)
                )
            """)
            
            conn.commit()
    
    def scan_for_new_files(self, directory: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Scan for files and categorize as existing, new, or resumed"""
        
        # Load last scan info
        last_scan_info = {}
        if self.last_scan_path.exists():
            try:
                with open(self.last_scan_path, 'r') as f:
                    last_scan_info = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load last scan info: {e}")
        
        # Current file scan
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
        current_videos = []
        for ext in video_extensions:
            current_videos.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
        current_videos = sorted(list(set(current_videos)))
        
        current_gpx = glob.glob(os.path.join(directory, '*.gpx'))
        current_gpx.extend(glob.glob(os.path.join(directory, '*.GPX')))
        current_gpx = sorted(list(set(current_gpx)))
        
        # Get file modification times
        current_video_info = {}
        for video in current_videos:
            try:
                mtime = os.path.getmtime(video)
                current_video_info[video] = mtime
            except:
                current_video_info[video] = 0
        
        current_gpx_info = {}
        for gpx in current_gpx:
            try:
                mtime = os.path.getmtime(gpx)
                current_gpx_info[gpx] = mtime
            except:
                current_gpx_info[gpx] = 0
        
        # Categorize files
        if not self.config.powersafe:
            # Non-powersafe mode: treat all as existing
            return current_videos, current_gpx, [], []
        
        # Get interrupted/paused files from database
        interrupted_videos, interrupted_gpx = self._get_interrupted_files()
        
        # Determine new files (not in last scan or modified after last scan)
        last_videos = last_scan_info.get('videos', {})
        last_gpx = last_scan_info.get('gpx', {})
        last_scan_time = last_scan_info.get('scan_time', 0)
        
        new_videos = []
        existing_videos = []
        
        for video, mtime in current_video_info.items():
            if (video not in last_videos or 
                mtime > last_videos.get(video, 0) or
                mtime > last_scan_time):
                if video not in interrupted_videos:
                    new_videos.append(video)
                else:
                    existing_videos.append(video)
            else:
                existing_videos.append(video)
        
        new_gpx = []
        existing_gpx = []
        
        for gpx, mtime in current_gpx_info.items():
            if (gpx not in last_gpx or 
                mtime > last_gpx.get(gpx, 0) or
                mtime > last_scan_time):
                if gpx not in interrupted_gpx:
                    new_gpx.append(gpx)
                else:
                    existing_gpx.append(gpx)
            else:
                existing_gpx.append(gpx)
        
        # Update scan info
        scan_info = {
            'scan_time': time.time(),
            'videos': current_video_info,
            'gpx': current_gpx_info
        }
        
        with open(self.last_scan_path, 'w') as f:
            json.dump(scan_info, f, indent=2)
        
        # Priority order: interrupted first, then existing, then new
        prioritized_videos = interrupted_videos + existing_videos + new_videos
        prioritized_gpx = interrupted_gpx + existing_gpx + new_gpx
        
        logger.info(f"File scan results:")
        logger.info(f"  Interrupted videos: {len(interrupted_videos)}")
        logger.info(f"  Existing videos: {len(existing_videos)}")
        logger.info(f"  New videos: {len(new_videos)}")
        logger.info(f"  Interrupted GPX: {len(interrupted_gpx)}")
        logger.info(f"  Existing GPX: {len(existing_gpx)}")
        logger.info(f"  New GPX: {len(new_gpx)}")
        
        return prioritized_videos, prioritized_gpx, new_videos, new_gpx
    
    def _get_interrupted_files(self) -> Tuple[List[str], List[str]]:
        """Get files that were interrupted during processing"""
        interrupted_videos = []
        interrupted_gpx = []
        
        if not self.db_path.exists():
            return interrupted_videos, interrupted_gpx
        
        with sqlite3.connect(self.db_path) as conn:
            # Videos that were being processed but not completed
            cursor = conn.execute("""
                SELECT video_path FROM video_progress 
                WHERE status IN ('processing', 'pending') OR 
                      (feature_extraction_done = TRUE AND correlation_done = FALSE)
                ORDER BY processed_at DESC
            """)
            interrupted_videos = [row[0] for row in cursor.fetchall()]
            
            # GPX files that were being processed but not completed
            cursor = conn.execute("""
                SELECT gpx_path FROM gpx_progress 
                WHERE status IN ('processing', 'pending') OR feature_extraction_done = FALSE
                ORDER BY processed_at DESC
            """)
            interrupted_gpx = [row[0] for row in cursor.fetchall()]
        
        return interrupted_videos, interrupted_gpx
    
    def mark_video_processing(self, video_path: str):
        """Mark video as currently being processed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else 0
            conn.execute("""
                INSERT OR REPLACE INTO video_progress 
                (video_path, status, processed_at, file_mtime)
                VALUES (?, 'processing', datetime('now'), ?)
            """, (video_path, mtime))
            conn.commit()
    
    def mark_video_features_done(self, video_path: str):
        """Mark video feature extraction as completed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE video_progress 
                SET feature_extraction_done = TRUE, processed_at = datetime('now')
                WHERE video_path = ?
            """, (video_path,))
            conn.commit()
    
    def mark_gpx_processing(self, gpx_path: str):
        """Mark GPX as currently being processed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            mtime = os.path.getmtime(gpx_path) if os.path.exists(gpx_path) else 0
            conn.execute("""
                INSERT OR REPLACE INTO gpx_progress 
                (gpx_path, status, processed_at, file_mtime)
                VALUES (?, 'processing', datetime('now'), ?)
            """, (gpx_path, mtime))
            conn.commit()
    
    def mark_gpx_features_done(self, gpx_path: str):
        """Mark GPX feature extraction as completed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE gpx_progress 
                SET feature_extraction_done = TRUE, status = 'completed', processed_at = datetime('now')
                WHERE gpx_path = ?
            """, (gpx_path,))
            conn.commit()
    
    def save_correlation_result(self, video_path: str, gpx_path: str, 
                              correlation_score: float, correlation_details: Dict):
        """Save individual correlation result"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO correlation_progress 
                (video_path, gpx_path, correlation_score, correlation_details, processed_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (video_path, gpx_path, correlation_score, json.dumps(correlation_details)))
            conn.commit()
    
    def mark_video_correlation_done(self, video_path: str, best_match_score: float, best_match_path: str):
        """Mark video correlation as completed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE video_progress 
                SET correlation_done = TRUE, status = 'completed', 
                    best_match_score = ?, best_match_path = ?, processed_at = datetime('now')
                WHERE video_path = ?
            """, (best_match_score, best_match_path, video_path))
            conn.commit()
    
    def add_pending_correlation(self, video_path: str, gpx_path: str, match_info: Dict):
        """Add correlation result to pending batch"""
        if not self.config.powersafe:
            return
        
        if video_path not in self.pending_results:
            self.pending_results[video_path] = {'matches': []}
        
        self.pending_results[video_path]['matches'].append(match_info)
        self.correlation_counter += 1
        
        # Save correlation to database
        self.save_correlation_result(video_path, gpx_path, 
                                   match_info['combined_score'], match_info)
        
        # Check if we should save incrementally
        if self.correlation_counter % self.config.save_interval == 0:
            self.save_incremental_results(self.pending_results)
            logger.info(f"Incremental save: {self.correlation_counter} correlations processed")
    
    def save_incremental_results(self, results: Dict):
        """Save current correlation results incrementally"""
        if not self.config.powersafe:
            return
        
        try:
            # Load existing results
            existing_results = {}
            if self.results_path.exists():
                with open(self.results_path, 'r') as f:
                    existing_results = json.load(f)
            
            # Update with new results
            existing_results.update(results)
            
            # Save updated results
            temp_path = self.results_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(existing_results, f, indent=2, default=str)
            
            # Atomic replace
            temp_path.replace(self.results_path)
            
            logger.debug(f"Incremental results saved: {len(existing_results)} total correlations")
            
        except Exception as e:
            logger.error(f"Failed to save incremental results: {e}")
    
    def load_existing_results(self) -> Dict:
        """Load existing correlation results"""
        if not self.config.powersafe or not self.results_path.exists():
            return {}
        
        try:
            with open(self.results_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing correlation results")
            return results
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            return {}
    
    def get_processing_status(self) -> Dict:
        """Get overall processing status"""
        if not self.config.powersafe or not self.db_path.exists():
            return {}
        
        status = {}
        with sqlite3.connect(self.db_path) as conn:
            # Video status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM video_progress GROUP BY status
            """)
            video_status = dict(cursor.fetchall())
            
            # GPX status  
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM gpx_progress GROUP BY status
            """)
            gpx_status = dict(cursor.fetchall())
            
            # Correlation progress
            cursor = conn.execute("""
                SELECT COUNT(*) FROM correlation_progress
            """)
            total_correlations = cursor.fetchone()[0]
            
            status = {
                'video_status': video_status,
                'gpx_status': gpx_status,
                'total_correlations': total_correlations
            }
        
        return status
    
    def cleanup_completed_entries(self):
        """Clean up old completed entries to keep database size manageable"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Keep only last 1000 completed entries
            conn.execute("""
                DELETE FROM video_progress WHERE video_path IN (
                    SELECT video_path FROM video_progress 
                    WHERE status = 'completed' 
                    ORDER BY processed_at DESC 
                    LIMIT -1 OFFSET 1000
                )
            """)
            
            conn.execute("""
                DELETE FROM gpx_progress WHERE gpx_path IN (
                    SELECT gpx_path FROM gpx_progress 
                    WHERE status = 'completed' 
                    ORDER BY processed_at DESC 
                    LIMIT -1 OFFSET 1000
                )
            """)
            
            conn.commit()

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup comprehensive logging"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class EnhancedGPUManager:
    """Enhanced GPU management with proper load balancing"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.gpu_locks = {gpu_id: Lock() for gpu_id in gpu_ids}
        self.gpu_usage = {gpu_id: 0 for gpu_id in gpu_ids}
        self.validate_gpus()
        
    def validate_gpus(self):
        """Validate GPU availability and memory"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_ids:
            if gpu_id >= available_gpus:
                raise RuntimeError(f"GPU {gpu_id} not available (only {available_gpus} GPUs)")
        
        # Check GPU memory
        for gpu_id in self.gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            if memory_gb < 4:
                logger.warning(f"GPU {gpu_id} has only {memory_gb:.1f}GB memory")
            logger.info(f"GPU {gpu_id}: {props.name} ({memory_gb:.1f}GB)")
    
    def get_best_gpu(self) -> int:
        """Get the GPU with lowest current usage"""
        return min(self.gpu_usage.keys(), key=lambda x: self.gpu_usage[x])
    
    def acquire_gpu(self, gpu_id: int) -> bool:
        """Acquire GPU for processing"""
        if self.gpu_locks[gpu_id].acquire(blocking=False):
            self.gpu_usage[gpu_id] += 1
            return True
        return False
    
    def release_gpu(self, gpu_id: int):
        """Release GPU after processing"""
        self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
        self.gpu_locks[gpu_id].release()

class EnhancedFFmpegDecoder:
    """Enhanced FFmpeg decoder with proper multi-GPU support"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.temp_dirs = {}
        
        # Create temp directories per GPU
        for gpu_id in gpu_manager.gpu_ids:
            self.temp_dirs[gpu_id] = tempfile.mkdtemp(prefix=f'enhanced_gpu_{gpu_id}_')
        
        logger.info(f"Enhanced decoder initialized for GPUs: {gpu_manager.gpu_ids}")
    
    def decode_video_enhanced(self, video_path: str, gpu_id: int) -> Tuple[Optional[torch.Tensor], float, float]:
        """Enhanced video decoding with better frame sampling"""
        try:
            # Get video info
            video_info = self._get_video_info(video_path)
            if not video_info:
                raise RuntimeError("Could not get video info")
            
            # Calculate optimal sampling
            total_frames = int(video_info['duration'] * video_info['fps'])
            if total_frames > self.config.max_frames:
                # Smart sampling - more frames from motion-heavy sections
                frames_tensor = self._smart_sample_frames(video_path, video_info, gpu_id)
            else:
                # Decode all frames
                frames_tensor = self._decode_all_frames(video_path, video_info, gpu_id)
            
            if frames_tensor is None:
                raise RuntimeError("Frame decoding failed")
            
            return frames_tensor, video_info['fps'], video_info['duration']
            
        except Exception as e:
            logger.error(f"Enhanced video decoding failed for {video_path}: {e}")
            return None, 0, 0
    
    def _get_video_info(self, video_path: str) -> Optional[Dict]:
        """Get comprehensive video information"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_format', '-show_streams', video_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            info = json.loads(result.stdout)
            
            video_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            fps = eval(video_stream.get('r_frame_rate', '30/1'))
            duration = float(info.get('format', {}).get('duration', 0))
            width = int(video_stream.get('width', 1920))
            height = int(video_stream.get('height', 1080))
            
            return {
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height,
                'codec': video_stream.get('codec_name'),
                'pixel_format': video_stream.get('pix_fmt')
            }
            
        except Exception as e:
            logger.error(f"Could not get video info for {video_path}: {e}")
            return None
    
    def _smart_sample_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """Smart frame sampling based on motion content"""
        try:
            # First pass: quick motion detection
            motion_scores = self._analyze_motion_content(video_path, video_info, gpu_id)
            
            # Select frames based on motion scores
            frame_indices = self._select_optimal_frames(motion_scores, self.config.max_frames)
            
            # Second pass: extract selected frames
            frames_tensor = self._extract_selected_frames(video_path, frame_indices, gpu_id)
            
            return frames_tensor
            
        except Exception as e:
            logger.error(f"Smart sampling failed: {e}")
            return self._decode_uniform_frames(video_path, video_info, gpu_id)
    
    def _analyze_motion_content(self, video_path: str, video_info: Dict, gpu_id: int) -> np.ndarray:
        """Analyze motion content for smart sampling"""
        temp_dir = self.temp_dirs[gpu_id]
        
        # Extract frames at low resolution for motion analysis
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-vf', 'scale=160:90',
            '-r', '2',  # 2 FPS for motion analysis
            '-f', 'image2pipe',
            '-pix_fmt', 'gray',
            '-vcodec', 'rawvideo', '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            frames_data = result.stdout
            
            # Parse frames
            frame_size = 160 * 90
            num_frames = len(frames_data) // frame_size
            
            motion_scores = []
            prev_frame = None
            
            for i in range(num_frames):
                start_idx = i * frame_size
                frame_data = frames_data[start_idx:start_idx + frame_size]
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(90, 160)
                
                if prev_frame is not None:
                    # Calculate motion
                    diff = np.abs(frame.astype(float) - prev_frame.astype(float))
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)
                else:
                    motion_scores.append(0.0)
                
                prev_frame = frame
            
            return np.array(motion_scores)
            
        except Exception as e:
            logger.debug(f"Motion analysis failed: {e}")
            # Fallback to uniform sampling
            total_frames = int(video_info['duration'] * 2)  # 2 FPS
            return np.ones(total_frames)
    
    def _select_optimal_frames(self, motion_scores: np.ndarray, max_frames: int) -> List[int]:
        """Select optimal frames based on motion scores"""
        if len(motion_scores) <= max_frames:
            return list(range(len(motion_scores)))
        
        # Combine motion-based and uniform sampling
        motion_frames = int(max_frames * 0.7)  # 70% motion-based
        uniform_frames = max_frames - motion_frames
        
        # Select high-motion frames
        motion_indices = np.argsort(motion_scores)[-motion_frames:]
        
        # Add uniform sampling
        uniform_indices = np.linspace(0, len(motion_scores)-1, uniform_frames, dtype=int)
        
        # Combine and sort
        all_indices = np.concatenate([motion_indices, uniform_indices])
        all_indices = np.unique(all_indices)
        
        # Convert to original video frame indices (assuming 2 FPS analysis)
        original_indices = all_indices * 15  # Approximate scaling factor
        
        return sorted(original_indices.tolist())
    
    def _extract_selected_frames(self, video_path: str, frame_indices: List[int], gpu_id: int) -> Optional[torch.Tensor]:
        """Extract specific frames from video"""
        temp_dir = self.temp_dirs[gpu_id]
        frames = []
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Extract frames using ffmpeg
        for i, frame_idx in enumerate(frame_indices):
            output_path = os.path.join(temp_dir, f'frame_{i:06d}.jpg')
            
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                '-i', video_path,
                '-vf', f'select=eq(n\\,{frame_idx}),scale={self.config.target_size[0]}:{self.config.target_size[1]}',
                '-frames:v', '1',
                '-q:v', '2',
                output_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                
                if os.path.exists(output_path):
                    # Load frame
                    img = cv2.imread(output_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_tensor = torch.from_numpy(img).float() / 255.0
                        img_tensor = img_tensor.permute(2, 0, 1).to(device)
                        frames.append(img_tensor)
                    
                    os.remove(output_path)
                    
            except Exception as e:
                logger.debug(f"Failed to extract frame {frame_idx}: {e}")
                continue
        
        if not frames:
            return None
        
        return torch.stack(frames).unsqueeze(0)  # Add batch dimension
    
    def _decode_uniform_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """Fallback uniform frame sampling"""
        temp_dir = self.temp_dirs[gpu_id]
        output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
        
        # Calculate sampling rate
        total_frames = int(video_info['duration'] * video_info['fps'])
        if total_frames > self.config.max_frames:
            sample_rate = total_frames / self.config.max_frames
            vf_filter = f'scale={self.config.target_size[0]}:{self.config.target_size[1]},select=not(mod(n\\,{int(sample_rate)}))'
        else:
            vf_filter = f'scale={self.config.target_size[0]}:{self.config.target_size[1]}'
        
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-vf', vf_filter,
            '-frames:v', str(self.config.max_frames),
            '-q:v', '2',
            output_pattern
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
            return self._load_frames_to_tensor(temp_dir, gpu_id)
        except Exception as e:
            logger.error(f"Uniform sampling failed: {e}")
            return None
    
    def _decode_all_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """Decode all frames when video is short"""
        return self._decode_uniform_frames(video_path, video_info, gpu_id)
    
    def _load_frames_to_tensor(self, temp_dir: str, gpu_id: int) -> Optional[torch.Tensor]:
        """Load frames to GPU tensor"""
        frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.jpg')))
        
        if not frame_files:
            return None
        
        device = torch.device(f'cuda:{gpu_id}')
        frames = []
        
        for frame_file in frame_files:
            try:
                img = cv2.imread(frame_file)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).to(device)
                frames.append(img_tensor)
                
                os.remove(frame_file)
                
            except Exception as e:
                logger.debug(f"Failed to load frame {frame_file}: {e}")
                continue
        
        if not frames:
            return None
        
        frames_tensor = torch.stack(frames).unsqueeze(0)
        logger.debug(f"Loaded {len(frames)} frames to GPU {gpu_id}: {frames_tensor.shape}")
        
        return frames_tensor
    
    def cleanup(self):
        """Cleanup temporary directories"""
        for temp_dir in self.temp_dirs.values():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

class EnhancedFeatureExtractor:
    """Enhanced feature extraction with improved temporal analysis"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.feature_models = {}
        
        # Initialize models for each GPU
        for gpu_id in gpu_manager.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            self.feature_models[gpu_id] = self._create_enhanced_model().to(device)
        
        logger.info("Enhanced feature extractor initialized")
    
    def _create_enhanced_model(self) -> nn.Module:
        """Create enhanced feature extraction model"""
        class EnhancedFeatureNet(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Efficient backbone
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                
                # Multiple feature heads
                self.scene_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64)
                )
                
                self.motion_head = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32)
                )
                
                self.texture_head = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                return {
                    'scene_features': self.scene_head(features),
                    'motion_features': self.motion_head(features),
                    'texture_features': self.texture_head(features)
                }
        
        model = EnhancedFeatureNet()
        model.eval()
        return model
    
    def extract_enhanced_features(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract enhanced features with improved temporal analysis"""
        try:
            device = torch.device(f'cuda:{gpu_id}')
            model = self.feature_models[gpu_id]
            
            if frames_tensor.device != device:
                frames_tensor = frames_tensor.to(device)
            
            features = {}
            
            with torch.no_grad():
                # CNN features
                batch_size, num_frames = frames_tensor.shape[:2]
                frames_flat = frames_tensor.view(-1, *frames_tensor.shape[2:])
                
                cnn_features = model(frames_flat)
                
                # Reshape to sequence format
                for key, value in cnn_features.items():
                    value = value.view(batch_size, num_frames, -1)[0]  # Remove batch dim
                    features[key] = value.cpu().numpy()
                
                # Enhanced motion features
                motion_features = self._compute_enhanced_motion(frames_tensor[0], device)
                features.update(motion_features)
                
                # Temporal coherence features
                temporal_features = self._compute_temporal_coherence(frames_tensor[0], device)
                features.update(temporal_features)
                
                # Improved color features
                color_features = self._compute_enhanced_color(frames_tensor[0], device)
                features.update(color_features)
                
                # Edge and texture features
                edge_features = self._compute_edge_features(frames_tensor[0], device)
                features.update(edge_features)
            
            logger.debug(f"Enhanced feature extraction successful: {len(features)} feature types")
            return features
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction failed: {e}")
            raise
    
    def _compute_enhanced_motion(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Compute enhanced motion features"""
        num_frames = frames.shape[0]
        
        features = {
            'motion_magnitude': np.zeros(num_frames),
            'motion_direction': np.zeros(num_frames),
            'acceleration': np.zeros(num_frames),
            'jerk': np.zeros(num_frames),
            'motion_consistency': np.zeros(num_frames)
        }
        
        if num_frames < 2:
            return {k: v for k, v in features.items()}
        
        # Convert to grayscale
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        # Compute optical flow approximation
        motion_vectors = []
        for i in range(num_frames - 1):
            frame1 = gray_frames[i]
            frame2 = gray_frames[i + 1]
            
            # Frame difference
            diff = torch.abs(frame2 - frame1)
            
            # Motion magnitude
            magnitude = torch.mean(diff).item()
            features['motion_magnitude'][i + 1] = magnitude
            
            # Motion direction (gradient-based)
            if diff.sum() > 0:
                grad_x = torch.mean(torch.abs(diff[:, 1:] - diff[:, :-1])).item()
                grad_y = torch.mean(torch.abs(diff[1:, :] - diff[:-1, :])).item()
                direction = math.atan2(grad_y, grad_x + 1e-8)
                features['motion_direction'][i + 1] = direction
                motion_vectors.append([grad_x, grad_y])
            else:
                motion_vectors.append([0, 0])
        
        # Compute acceleration
        motion_mag = features['motion_magnitude']
        for i in range(1, num_frames - 1):
            features['acceleration'][i] = motion_mag[i + 1] - motion_mag[i]
        
        # Compute jerk
        accel = features['acceleration']
        for i in range(1, num_frames - 2):
            features['jerk'][i] = accel[i + 1] - accel[i]
        
        # Motion consistency (how consistent motion direction is)
        if len(motion_vectors) > self.config.temporal_window:
            for i in range(self.config.temporal_window, len(motion_vectors)):
                window = motion_vectors[i-self.config.temporal_window:i]
                if window:
                    angles = [math.atan2(v[1], v[0] + 1e-8) for v in window]
                    # Circular variance
                    circular_mean = math.atan2(
                        sum(math.sin(a) for a in angles) / len(angles),
                        sum(math.cos(a) for a in angles) / len(angles)
                    )
                    variance = sum((a - circular_mean)**2 for a in angles) / len(angles)
                    consistency = 1.0 / (1.0 + variance)
                    features['motion_consistency'][i + 1] = consistency
        
        return {k: v for k, v in features.items()}
    
    def _compute_temporal_coherence(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Compute temporal coherence features"""
        num_frames = frames.shape[0]
        
        if num_frames < self.config.temporal_window:
            return {
                'temporal_stability': np.zeros(num_frames),
                'temporal_patterns': np.zeros(num_frames),
                'scene_changes': np.zeros(num_frames)
            }
        
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        stability = np.zeros(num_frames)
        patterns = np.zeros(num_frames)
        scene_changes = np.zeros(num_frames)
        
        window_size = self.config.temporal_window
        
        for i in range(window_size, num_frames):
            # Temporal stability
            window_frames = gray_frames[i-window_size:i+1]
            mean_frame = torch.mean(window_frames, dim=0)
            stability_score = 1.0 / (1.0 + torch.var(window_frames).item())
            stability[i] = stability_score
            
            # Temporal patterns (autocorrelation)
            if i >= window_size * 2:
                recent_window = gray_frames[i-window_size:i]
                past_window = gray_frames[i-window_size*2:i-window_size]
                
                # Simplified correlation
                correlation = torch.corrcoef(torch.stack([
                    recent_window.flatten(),
                    past_window.flatten()
                ]))[0, 1].item()
                
                if not math.isnan(correlation):
                    patterns[i] = abs(correlation)
            
            # Scene change detection
            if i > 0:
                curr_frame = gray_frames[i]
                prev_frame = gray_frames[i-1]
                change_score = torch.mean(torch.abs(curr_frame - prev_frame)).item()
                scene_changes[i] = change_score
        
        return {
            'temporal_stability': stability,
            'temporal_patterns': patterns,
            'scene_changes': scene_changes
        }
    
    def _compute_enhanced_color(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Compute enhanced color features"""
        num_frames = frames.shape[0]
        
        # Color variance over time
        color_variance = torch.var(frames, dim=[2, 3])  # Variance per channel per frame
        mean_color_variance = torch.mean(color_variance, dim=1).cpu().numpy()
        
        # Color histograms (enhanced)
        histograms = []
        color_moments = []
        
        for i in range(num_frames):
            frame = frames[i]
            
            # Color moments (mean, std, skewness approximation)
            moments = []
            for c in range(3):  # RGB channels
                channel = frame[c]
                mean_val = torch.mean(channel).item()
                std_val = torch.std(channel).item()
                # Simplified skewness
                skew_val = torch.mean((channel - mean_val)**3).item() / (std_val**3 + 1e-8)
                moments.extend([mean_val, std_val, skew_val])
            
            color_moments.append(moments)
            
            # Enhanced histogram
            frame_quantized = (frame * 31).long()  # 32 bins per channel
            hist_features = []
            
            for c in range(3):
                channel_hist = torch.bincount(frame_quantized[c].flatten(), minlength=32)[:32]
                channel_hist = channel_hist.float() / torch.sum(channel_hist)
                hist_features.extend(channel_hist.cpu().numpy())
            
            histograms.append(hist_features)
        
        return {
            'color_variance': mean_color_variance,
            'color_histograms': np.array(histograms),
            'color_moments': np.array(color_moments)
        }
    
    def _compute_edge_features(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Compute edge and texture features"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray_frames = gray_frames.unsqueeze(1)  # Add channel dimension
        
        # Edge detection
        edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
        edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3]).cpu().numpy()
        
        # Edge orientation
        edge_angle = torch.atan2(edges_y, edges_x + 1e-8)
        edge_orientation = torch.mean(torch.abs(torch.cos(edge_angle)), dim=[1, 2, 3]).cpu().numpy()
        
        # Texture features (LBP approximation)
        texture_complexity = []
        for i in range(frames.shape[0]):
            frame_gray = gray_frames[i, 0]
            # Simplified texture measure
            grad_x = torch.diff(frame_gray, dim=1)
            grad_y = torch.diff(frame_gray, dim=0)
            texture_score = torch.mean(torch.abs(grad_x)**2 + torch.abs(grad_y[:-1])**2).item()
            texture_complexity.append(texture_score)
        
        return {
            'edge_density': edge_density,
            'edge_orientation': edge_orientation,
            'texture_complexity': np.array(texture_complexity)
        }

class RobustGPXProcessor:
    """Robust GPX processor with fixed array handling"""
    
    def __init__(self, config: ProcessingConfig, powersafe_manager: Optional['PowerSafeManager'] = None):
        self.config = config
        self.powersafe_manager = powersafe_manager
        if not cp.cuda.is_available():
            raise RuntimeError("CuPy CUDA required for GPX processing")
        logger.info("Robust GPX processor initialized")
    
    def process_gpx_files(self, gpx_paths: List[str], max_workers: int = None) -> Dict[str, Any]:
        """Process GPX files with robust error handling"""
        if max_workers is None:
            max_workers = min(16, mp.cpu_count())
        
        logger.info(f"Processing {len(gpx_paths)} GPX files with {max_workers} workers")
        
        results = {}
        failed_files = []
        
        # Parse GPX files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._parse_gpx_safe, path): path for path in gpx_paths}
            
            for future in tqdm(as_completed(futures), total=len(gpx_paths), desc="Parsing GPX"):
                path = futures[future]
                
                # Mark as processing in PowerSafe mode
                if self.powersafe_manager:
                    self.powersafe_manager.mark_gpx_processing(path)
                
                try:
                    data = future.result()
                    if data is not None:
                        results[path] = data
                        # Mark as completed in PowerSafe mode
                        if self.powersafe_manager:
                            self.powersafe_manager.mark_gpx_features_done(path)
                    else:
                        failed_files.append(path)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    failed_files.append(path)
        
        logger.info(f"Successfully processed {len(results)}/{len(gpx_paths)} GPX files")
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} GPX files")
        
        return results
    
    def _parse_gpx_safe(self, gpx_path: str) -> Optional[Dict]:
        """Safely parse and process single GPX file"""
        try:
            # Parse GPX
            with open(gpx_path, 'r', encoding='utf-8', errors='ignore') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time is not None:
                            points.append({
                                'timestamp': point.time.replace(tzinfo=None),
                                'lat': float(point.latitude),
                                'lon': float(point.longitude),
                                'elevation': float(point.elevation or 0)
                            })
            
            if len(points) < 10:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate coordinates
            if (df['lat'].isna().any() or df['lon'].isna().any() or
                not (-90 <= df['lat'].min() <= df['lat'].max() <= 90) or
                not (-180 <= df['lon'].min() <= df['lon'].max() <= 180)):
                return None
            
            # Compute enhanced features
            enhanced_data = self._compute_robust_features(df)
            
            return enhanced_data
            
        except Exception as e:
            logger.debug(f"GPX processing failed for {gpx_path}: {e}")
            return None
    
    def _compute_robust_features(self, df: pd.DataFrame) -> Dict:
        """Compute robust features with proper array handling"""
        n_points = len(df)
        
        # Transfer to GPU
        lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)
        lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
        elevs_gpu = cp.array(df['elevation'].values, dtype=cp.float64)
        
        # Compute time differences (ensure proper length)
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        time_diffs_gpu = cp.array(time_diffs, dtype=cp.float64)
        
        # Ensure all arrays are same length
        assert len(lats_gpu) == len(lons_gpu) == len(elevs_gpu) == len(time_diffs_gpu) == n_points
        
        # Compute distances
        distances_gpu = self._compute_distances_robust(lats_gpu, lons_gpu)
        
        # Ensure distances array is correct length
        assert len(distances_gpu) == n_points
        
        # Compute motion features with proper array handling
        motion_features = self._compute_motion_features_robust(
            lats_gpu, lons_gpu, elevs_gpu, time_diffs_gpu, distances_gpu
        )
        
        # Validate all feature arrays have correct length
        for key, values in motion_features.items():
            if len(values) != n_points:
                logger.warning(f"Feature {key} has wrong length: {len(values)} vs {n_points}")
                # Pad or truncate to correct length
                if len(values) < n_points:
                    padded = cp.zeros(n_points)
                    padded[:len(values)] = values
                    motion_features[key] = padded
                else:
                    motion_features[key] = values[:n_points]
        
        # Compute statistical features
        statistical_features = self._compute_statistical_features_robust(motion_features, distances_gpu, elevs_gpu)
        
        # Convert to CPU
        cpu_features = {
            key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
            for key, value in {**motion_features, **statistical_features}.items()
        }
        
        # Add metadata
        duration = self._compute_duration_safe(df['timestamp'])
        total_distance = float(cp.sum(distances_gpu))
        
        return {
            'df': df,
            'features': cpu_features,
            'start_time': df['timestamp'].iloc[0],
            'end_time': df['timestamp'].iloc[-1],
            'duration': duration,
            'distance': total_distance,
            'point_count': n_points,
            'max_speed': float(cp.max(motion_features['speed'])) if 'speed' in motion_features else 0,
            'avg_speed': float(cp.mean(motion_features['speed'])) if 'speed' in motion_features else 0
        }
    
    def _compute_time_differences_safe(self, timestamps: np.ndarray) -> List[float]:
        """Safely compute time differences"""
        n = len(timestamps)
        time_diffs = [1.0]  # First point
        
        for i in range(1, n):
            try:
                time_diff = timestamps[i] - timestamps[i-1]
                
                if hasattr(time_diff, 'total_seconds'):
                    seconds = time_diff.total_seconds()
                elif isinstance(time_diff, np.timedelta64):
                    seconds = float(time_diff / np.timedelta64(1, 's'))
                else:
                    seconds = float(time_diff)
                
                # Ensure positive and reasonable
                if 0 < seconds <= 3600:  # Between 0 and 1 hour
                    time_diffs.append(seconds)
                else:
                    time_diffs.append(1.0)  # Default fallback
                    
            except Exception:
                time_diffs.append(1.0)  # Safe fallback
        
        return time_diffs
    
    def _compute_distances_robust(self, lats: cp.ndarray, lons: cp.ndarray) -> cp.ndarray:
        """Robustly compute distances with proper array handling"""
        n = len(lats)
        distances = cp.zeros(n)  # Initialize with zeros
        
        if n < 2:
            return distances
        
        R = 3958.8  # Earth radius in miles
        
        # Convert to radians
        lat1_rad = cp.radians(lats[:-1])
        lon1_rad = cp.radians(lons[:-1])
        lat2_rad = cp.radians(lats[1:])
        lon2_rad = cp.radians(lons[1:])
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))
        
        computed_distances = R * c
        
        # Set distances for points 1 to n-1 (first point distance remains 0)
        distances[1:] = computed_distances
        
        return distances
    
    def _compute_motion_features_robust(self, lats: cp.ndarray, lons: cp.ndarray, 
                                      elevs: cp.ndarray, time_diffs: cp.ndarray, 
                                      distances: cp.ndarray) -> Dict[str, cp.ndarray]:
        """Robustly compute motion features with proper array handling"""
        n = len(lats)
        
        # Initialize all arrays with correct length
        features = {
            'speed': cp.zeros(n),
            'acceleration': cp.zeros(n),
            'jerk': cp.zeros(n),
            'bearing': cp.zeros(n),
            'bearing_change': cp.zeros(n),
            'curvature': cp.zeros(n),
            'elevation_change_rate': cp.zeros(n)
        }
        
        if n < 2:
            return features
        
        # Speed computation (for points 1 to n-1)
        valid_time_diffs = cp.maximum(time_diffs[1:], 1e-6)
        speed_values = distances[1:] * 3600 / valid_time_diffs  # mph
        features['speed'][1:] = speed_values
        
        # Acceleration (for points 2 to n-1)
        if n > 2:
            speed_diff = features['speed'][2:] - features['speed'][1:-1]
            accel_values = speed_diff / cp.maximum(time_diffs[2:], 1e-6)
            features['acceleration'][2:] = accel_values
        
        # Jerk (for points 3 to n-1)
        if n > 3:
            accel_diff = features['acceleration'][3:] - features['acceleration'][2:-1]
            jerk_values = accel_diff / cp.maximum(time_diffs[3:], 1e-6)
            features['jerk'][3:] = jerk_values
        
        # Bearings (for points 1 to n-1)
        if n > 1:
            bearings = self._compute_bearings_robust(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
        # Bearing changes (for points 2 to n-1)
        if n > 2:
            bearing_diffs = cp.diff(features['bearing'][1:])  # This gives us n-2 values
            # Handle angle wraparound
            bearing_diffs = cp.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = cp.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = cp.abs(bearing_diffs)
            features['bearing_change'][2:] = bearing_changes
            
            # Curvature (bearing change per distance)
            valid_distances = cp.maximum(distances[2:], 1e-8)
            curvature_values = bearing_changes / valid_distances
            features['curvature'][2:] = curvature_values
        
        # Elevation change rate (for points 1 to n-1)
        if n > 1:
            elev_diffs = elevs[1:] - elevs[:-1]
            elev_rates = elev_diffs / cp.maximum(time_diffs[1:], 1e-6)
            features['elevation_change_rate'][1:] = elev_rates
        
        return features
    
    def _compute_bearings_robust(self, lat1: cp.ndarray, lon1: cp.ndarray, 
                               lat2: cp.ndarray, lon2: cp.ndarray) -> cp.ndarray:
        """Robustly compute bearings"""
        lat1_rad = cp.radians(lat1)
        lon1_rad = cp.radians(lon1)
        lat2_rad = cp.radians(lat2)
        lon2_rad = cp.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = cp.sin(dlon) * cp.cos(lat2_rad)
        x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(dlon)
        
        bearings = cp.degrees(cp.arctan2(y, x))
        bearings = cp.where(bearings < 0, bearings + 360, bearings)
        
        return bearings
    
    def _compute_statistical_features_robust(self, motion_features: Dict[str, cp.ndarray],
                                           distances: cp.ndarray, elevations: cp.ndarray) -> Dict[str, cp.ndarray]:
        """Compute statistical features robustly"""
        features = {}
        
        # Speed statistics
        speed = motion_features['speed']
        valid_speed = speed[speed > 0]  # Only consider non-zero speeds
        if len(valid_speed) > 0:
            features['speed_stats'] = cp.array([
                cp.mean(valid_speed), cp.std(valid_speed), 
                cp.min(valid_speed), cp.max(valid_speed),
                cp.percentile(valid_speed, 25), cp.percentile(valid_speed, 50), 
                cp.percentile(valid_speed, 75)
            ])
        else:
            features['speed_stats'] = cp.zeros(7)
        
        # Bearing statistics
        bearing = motion_features['bearing']
        valid_bearing = bearing[bearing > 0]
        if len(valid_bearing) > 0:
            features['bearing_stats'] = cp.array([
                cp.mean(valid_bearing), cp.std(valid_bearing),
                cp.min(valid_bearing), cp.max(valid_bearing)
            ])
        else:
            features['bearing_stats'] = cp.zeros(4)
        
        # Elevation statistics
        if len(elevations) > 1:
            elev_diffs = cp.diff(elevations)
            total_climb = cp.sum(cp.maximum(elev_diffs, 0))
            total_descent = cp.sum(cp.maximum(-elev_diffs, 0))
            
            features['elevation_stats'] = cp.array([
                cp.mean(elevations), cp.std(elevations),
                cp.min(elevations), cp.max(elevations),
                total_climb, total_descent
            ])
        else:
            features['elevation_stats'] = cp.zeros(6)
        
        # Distance statistics
        valid_distances = distances[distances > 0]
        if len(valid_distances) > 0:
            features['distance_stats'] = cp.array([
                cp.sum(distances), cp.mean(valid_distances),
                cp.std(valid_distances), cp.max(valid_distances)
            ])
        else:
            features['distance_stats'] = cp.zeros(4)
        
        return features
    
    def _compute_duration_safe(self, timestamps: pd.Series) -> float:
        """Safely compute duration"""
        try:
            duration_delta = timestamps.iloc[-1] - timestamps.iloc[0]
            if hasattr(duration_delta, 'total_seconds'):
                return duration_delta.total_seconds()
            else:
                return float(duration_delta / np.timedelta64(1, 's'))
        except Exception:
            return 3600.0  # Default 1 hour

class EnhancedSimilarityEngine:
    """Enhanced similarity computation with DTW and improved metrics"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.weights = {
            'motion_dynamics': 0.30,
            'temporal_correlation': 0.25,
            'statistical_profile': 0.20,
            'spatial_features': 0.15,
            'behavioral_patterns': 0.10
        }
    
    def compute_enhanced_similarity(self, video_features: Dict, gpx_features: Dict) -> Dict[str, float]:
        """Compute enhanced similarity with multiple methods"""
        try:
            similarities = {}
            
            # Motion dynamics similarity
            similarities['motion_dynamics'] = self._compute_motion_similarity(video_features, gpx_features)
            
            # Temporal correlation with DTW
            similarities['temporal_correlation'] = self._compute_temporal_similarity(video_features, gpx_features)
            
            # Statistical profile matching
            similarities['statistical_profile'] = self._compute_statistical_similarity(video_features, gpx_features)
            
            # Spatial feature correlation
            similarities['spatial_features'] = self._compute_spatial_similarity(video_features, gpx_features)
            
            # Behavioral pattern matching
            similarities['behavioral_patterns'] = self._compute_behavioral_similarity(video_features, gpx_features)
            
            # Weighted combination
            combined_score = sum(
                similarities[key] * self.weights[key] 
                for key in similarities.keys()
            )
            
            similarities['combined'] = float(np.clip(combined_score, 0.0, 1.0))
            similarities['quality'] = self._assess_quality(similarities['combined'])
            
            return similarities
            
        except Exception as e:
            logger.error(f"Enhanced similarity computation failed: {e}")
            return self._create_zero_similarity()
    
    def _compute_motion_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute motion similarity with improved alignment"""
        try:
            # Extract motion signatures
            video_motion = self._extract_motion_signature(video_features, 'video')
            gpx_motion = self._extract_motion_signature(gpx_features, 'gpx')
            
            if video_motion is None or gpx_motion is None or len(video_motion) < 3 or len(gpx_motion) < 3:
                return 0.0
            
            # Normalize signatures
            video_motion = self._robust_normalize(video_motion)
            gpx_motion = self._robust_normalize(gpx_motion)
            
            # Use DTW for alignment if available
            if FASTDTW_AVAILABLE and len(video_motion) > 10 and len(gpx_motion) > 10:
                distance, _ = fastdtw(video_motion.reshape(-1, 1), gpx_motion.reshape(-1, 1))
                similarity = 1.0 / (1.0 + distance / max(len(video_motion), len(gpx_motion)))
            else:
                # Fallback to correlation-based similarity
                min_len = min(len(video_motion), len(gpx_motion))
                video_motion = video_motion[:min_len]
                gpx_motion = gpx_motion[:min_len]
                
                correlation = np.corrcoef(video_motion, gpx_motion)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                similarity = (correlation + 1) / 2  # Normalize to [0, 1]
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Motion similarity computation failed: {e}")
            return 0.0
    
    def _extract_motion_signature(self, features: Dict, source_type: str) -> Optional[np.ndarray]:
        """Extract comprehensive motion signature"""
        signature_components = []
        
        try:
            if source_type == 'video':
                # Video motion features
                for key in ['motion_magnitude', 'acceleration', 'motion_consistency']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                # Use temporal pattern
                                signature_components.append(values)
                
                # Add edge-based motion proxy
                if 'edge_density' in features:
                    values = features['edge_density']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        if np.isfinite(values).all():
                            signature_components.append(values)
                            
            elif source_type == 'gpx':
                # GPX motion features
                for key in ['speed', 'acceleration', 'curvature']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                signature_components.append(values)
            
            if not signature_components:
                return None
            
            # Create unified signature by concatenating normalized components
            normalized_components = []
            for component in signature_components:
                normalized = self._robust_normalize(component)
                normalized_components.append(normalized)
            
            # Take the longest component as base and interpolate others
            max_len = max(len(comp) for comp in normalized_components)
            unified_signature = np.zeros(max_len)
            
            for component in normalized_components:
                if len(component) < max_len:
                    # Simple linear interpolation
                    indices = np.linspace(0, len(component)-1, max_len)
                    interpolated = np.interp(indices, np.arange(len(component)), component)
                    unified_signature += interpolated
                else:
                    unified_signature += component[:max_len]
            
            return unified_signature / len(normalized_components)  # Average
            
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_temporal_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute temporal correlation with advanced alignment"""
        try:
            # Extract temporal signatures
            video_temporal = self._extract_temporal_signature(video_features, 'video')
            gpx_temporal = self._extract_temporal_signature(gpx_features, 'gpx')
            
            if video_temporal is None or gpx_temporal is None:
                return 0.0
            
            # Multiple correlation approaches
            correlations = []
            
            # 1. Direct correlation
            min_len = min(len(video_temporal), len(gpx_temporal))
            if min_len > 5:
                v_temp = video_temporal[:min_len]
                g_temp = gpx_temporal[:min_len]
                
                corr = np.corrcoef(v_temp, g_temp)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            # 2. Phase-shifted correlation
            if min_len > 20:
                max_shifts = min(10, min_len // 4)
                for shift in range(-max_shifts, max_shifts + 1):
                    if shift == 0:
                        continue
                    
                    if shift > 0:
                        v_shifted = video_temporal[shift:min_len]
                        g_shifted = gpx_temporal[:min_len-shift]
                    else:
                        v_shifted = video_temporal[:min_len+shift]
                        g_shifted = gpx_temporal[-shift:min_len]
                    
                    if len(v_shifted) > 5:
                        corr = np.corrcoef(v_shifted, g_shifted)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            # 3. DTW-based similarity
            if FASTDTW_AVAILABLE and len(video_temporal) > 10 and len(gpx_temporal) > 10:
                try:
                    distance, _ = fastdtw(video_temporal.reshape(-1, 1), 
                                        gpx_temporal.reshape(-1, 1))
                    dtw_similarity = 1.0 / (1.0 + distance / max(len(video_temporal), len(gpx_temporal)))
                    correlations.append(dtw_similarity)
                except:
                    pass
            
            # Return best correlation
            if correlations:
                return float(np.clip(max(correlations), 0.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Temporal similarity computation failed: {e}")
            return 0.0
    
    def _extract_temporal_signature(self, features: Dict, source_type: str) -> Optional[np.ndarray]:
        """Extract temporal signature for correlation"""
        try:
            if source_type == 'video':
                # Primary: temporal patterns
                if 'temporal_patterns' in features:
                    values = features['temporal_patterns']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return values
                
                # Fallback: motion magnitude changes
                if 'motion_magnitude' in features:
                    values = features['motion_magnitude']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return np.diff(values)  # Temporal changes
                            
            elif source_type == 'gpx':
                # Primary: speed changes
                if 'speed' in features:
                    values = features['speed']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return np.diff(values)
                
                # Fallback: acceleration
                if 'acceleration' in features:
                    values = features['acceleration']
                    if isinstance(values, np.ndarray) and values.size > 3:
                        if np.isfinite(values).all():
                            return values
            
            return None
            
        except Exception as e:
            logger.debug(f"Temporal signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_statistical_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute statistical profile similarity"""
        try:
            video_stats = self._extract_statistical_profile(video_features, 'video')
            gpx_stats = self._extract_statistical_profile(gpx_features, 'gpx')
            
            if video_stats is None or gpx_stats is None:
                return 0.0
            
            # Ensure same length
            min_len = min(len(video_stats), len(gpx_stats))
            if min_len < 2:
                return 0.0
            
            video_stats = video_stats[:min_len]
            gpx_stats = gpx_stats[:min_len]
            
            # Normalize
            video_stats = self._robust_normalize(video_stats)
            gpx_stats = self._robust_normalize(gpx_stats)
            
            # Multiple similarity measures
            similarities = []
            
            # Cosine similarity
            cosine_sim = np.dot(video_stats, gpx_stats) / (
                np.linalg.norm(video_stats) * np.linalg.norm(gpx_stats) + 1e-8
            )
            if not np.isnan(cosine_sim):
                similarities.append(abs(cosine_sim))
            
            # Euclidean similarity
            euclidean_dist = np.linalg.norm(video_stats - gpx_stats)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            similarities.append(euclidean_sim)
            
            # Correlation
            correlation = np.corrcoef(video_stats, gpx_stats)[0, 1]
            if not np.isnan(correlation):
                similarities.append(abs(correlation))
            
            if similarities:
                return float(np.clip(np.mean(similarities), 0.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Statistical similarity computation failed: {e}")
            return 0.0
    
    def _extract_statistical_profile(self, features: Dict, source_type: str) -> Optional[np.ndarray]:
        """Extract statistical profile"""
        profile_components = []
        
        try:
            if source_type == 'video':
                # Video statistical features
                for key in ['motion_magnitude', 'color_variance', 'edge_density', 'texture_complexity']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend([
                                    np.mean(values), np.std(values), np.median(values),
                                    np.percentile(values, 25), np.percentile(values, 75)
                                ])
                
                # Color moments
                if 'color_moments' in features:
                    values = features['color_moments']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        if np.isfinite(values).all():
                            profile_components.extend(np.mean(values, axis=0)[:9])  # RGB moments
                            
            elif source_type == 'gpx':
                # GPX statistical features
                stat_keys = ['speed_stats', 'bearing_stats', 'elevation_stats', 'distance_stats']
                for key in stat_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend(values.flatten())
                
                # Motion statistics
                for key in ['speed', 'acceleration', 'curvature']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            valid_values = values[np.isfinite(values)]
                            if len(valid_values) > 0:
                                profile_components.extend([
                                    np.mean(valid_values), np.std(valid_values), np.median(valid_values)
                                ])
            
            if not profile_components:
                return None
            
            return np.array(profile_components)
            
        except Exception as e:
            logger.debug(f"Statistical profile extraction failed for {source_type}: {e}")
            return None
    
    def _compute_spatial_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute spatial feature similarity"""
        try:
            # This is a placeholder for spatial correlation
            # In practice, you might compare scene features with geographic context
            
            spatial_score = 0.0
            count = 0
            
            # Scene consistency (placeholder)
            if 'scene_features' in video_features:
                scene_features = video_features['scene_features']
                if isinstance(scene_features, np.ndarray) and scene_features.size > 0:
                    # Scene variation as proxy for spatial diversity
                    scene_variation = np.std(scene_features)
                    spatial_score += min(scene_variation * 2, 1.0)  # Scale appropriately
                    count += 1
            
            # Terrain correlation (placeholder using elevation)
            if 'elevation_stats' in gpx_features:
                elev_stats = gpx_features['elevation_stats']
                if isinstance(elev_stats, np.ndarray) and elev_stats.size >= 6:
                    # Elevation variation
                    elev_range = elev_stats[3] - elev_stats[2]  # max - min
                    elev_variation = elev_stats[1]  # std
                    terrain_complexity = min(elev_variation / max(elev_range, 1.0), 1.0)
                    spatial_score += terrain_complexity
                    count += 1
            
            return spatial_score / max(count, 1)
            
        except Exception as e:
            logger.debug(f"Spatial similarity computation failed: {e}")
            return 0.0
    
    def _compute_behavioral_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute behavioral pattern similarity"""
        try:
            behavioral_score = 0.0
            count = 0
            
            # Motion consistency patterns
            if 'motion_consistency' in video_features and 'curvature' in gpx_features:
                video_consistency = video_features['motion_consistency']
                gpx_curvature = gpx_features['curvature']
                
                if (isinstance(video_consistency, np.ndarray) and video_consistency.size > 0 and
                    isinstance(gpx_curvature, np.ndarray) and gpx_curvature.size > 0):
                    
                    # Compare motion smoothness patterns
                    video_smoothness = np.mean(video_consistency)
                    gpx_smoothness = 1.0 / (1.0 + np.mean(gpx_curvature))
                    
                    smoothness_similarity = 1.0 - abs(video_smoothness - gpx_smoothness)
                    behavioral_score += smoothness_similarity
                    count += 1
            
            # Speed behavior patterns
            if 'motion_magnitude' in video_features and 'speed_stats' in gpx_features:
                video_motion = video_features['motion_magnitude']
                gpx_speed_stats = gpx_features['speed_stats']
                
                if (isinstance(video_motion, np.ndarray) and video_motion.size > 0 and
                    isinstance(gpx_speed_stats, np.ndarray) and gpx_speed_stats.size > 1):
                    
                    # Compare variability patterns
                    video_cv = np.std(video_motion) / (np.mean(video_motion) + 1e-8)
                    gpx_cv = gpx_speed_stats[1] / (gpx_speed_stats[0] + 1e-8)  # std/mean
                    
                    cv_similarity = 1.0 / (1.0 + abs(video_cv - gpx_cv))
                    behavioral_score += cv_similarity
                    count += 1
            
            return behavioral_score / max(count, 1)
            
        except Exception as e:
            logger.debug(f"Behavioral similarity computation failed: {e}")
            return 0.0
    
    def _robust_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Robust normalization with outlier handling"""
        try:
            if len(vector) == 0:
                return vector
            
            # Remove outliers using IQR
            q25, q75 = np.percentile(vector, [25, 75])
            iqr = q75 - q25
            
            if iqr > 1e-8:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                vector = np.clip(vector, lower_bound, upper_bound)
            
            # Normalize to [0, 1]
            min_val, max_val = np.min(vector), np.max(vector)
            if max_val - min_val > 1e-8:
                vector = (vector - min_val) / (max_val - min_val)
            else:
                vector = np.zeros_like(vector)
            
            return vector
            
        except Exception:
            return vector
    
    def _assess_quality(self, score: float) -> str:
        """Assess similarity quality"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        elif score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def _create_zero_similarity(self) -> Dict[str, float]:
        """Create zero similarity result"""
        return {
            'motion_dynamics': 0.0,
            'temporal_correlation': 0.0,
            'statistical_profile': 0.0,
            'spatial_features': 0.0,
            'behavioral_patterns': 0.0,
            'combined': 0.0,
            'quality': 'failed'
        }

def process_video_parallel(args) -> Tuple[str, Optional[Dict]]:
    """Process video in parallel worker with power-safe tracking"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    gpu_id = gpu_manager.get_best_gpu()
    
    if not gpu_manager.acquire_gpu(gpu_id):
        # Try other GPUs
        for attempt_gpu in gpu_manager.gpu_ids:
            if gpu_manager.acquire_gpu(attempt_gpu):
                gpu_id = attempt_gpu
                break
        else:
            logger.error(f"Could not acquire any GPU for {video_path}")
            return video_path, None
    
    try:
        decoder = EnhancedFFmpegDecoder(gpu_manager, config)
        feature_extractor = EnhancedFeatureExtractor(gpu_manager, config)
        
        # Decode video
        frames_tensor, fps, duration = decoder.decode_video_enhanced(video_path, gpu_id)
        
        if frames_tensor is None:
            return video_path, None
        
        # Extract features
        features = feature_extractor.extract_enhanced_features(frames_tensor, gpu_id)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_id
        
        # Mark feature extraction as done in power-safe mode
        if powersafe_manager:
            powersafe_manager.mark_video_features_done(video_path)
        
        logger.debug(f"Successfully processed {Path(video_path).name} on GPU {gpu_id}")
        return video_path, features
        
    except Exception as e:
        logger.error(f"Video processing failed for {Path(video_path).name}: {e}")
        return video_path, None
        
    finally:
        gpu_manager.release_gpu(gpu_id)
        # Cleanup GPU memory
        try:
            torch.cuda.empty_cache()
            if gpu_id < torch.cuda.device_count():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        except:
            pass

def main():
    """Enhanced main function with complete PowerSafe implementation"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-GPU Video-GPX Correlation with PowerSafe",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", required=True,
                       help="Directory containing videos and GPX files")
    
    # Processing configuration
    parser.add_argument("--max_frames", type=int, default=1000,
                       help="Maximum frames per video (default: 1000)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[640, 360],
                       help="Target video resolution (default: 640 360)")
    parser.add_argument("--sample_rate", type=float, default=3.0,
                       help="Video sampling rate (default: 3.0)")
    parser.add_argument("--parallel_videos", type=int, default=4,
                       help="Number of videos to process in parallel (default: 4)")
    
    # GPU configuration
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1],
                       help="GPU IDs to use (default: [0, 1])")
    
    # Output configuration
    parser.add_argument("-o", "--output", default="./enhanced_results",
                       help="Output directory")
    parser.add_argument("-c", "--cache", default="./enhanced_cache",
                       help="Cache directory")
    parser.add_argument("-k", "--top_k", type=int, default=5,
                       help="Number of top matches per video")
    
    # Processing options
    parser.add_argument("--force", action='store_true',
                       help="Force reprocessing (ignore cache)")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug logging")
    
    # Power-safe mode arguments
    parser.add_argument("--powersafe", action='store_true',
                       help="Enable power-safe mode with incremental saves and resume capability")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save results every N correlations in powersafe mode (default: 5)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level, "enhanced_correlation.log")
    
    logger.info("Starting Enhanced Video-GPX Correlation System with PowerSafe")
    
    try:
        # Create configuration
        config = ProcessingConfig(
            max_frames=args.max_frames,
            target_size=tuple(args.video_size),
            sample_rate=args.sample_rate,
            parallel_videos=args.parallel_videos,
            powersafe=args.powersafe,
            save_interval=args.save_interval
        )
        
        # Setup directories
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        cache_dir = Path(args.cache)
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize PowerSafe manager
        powersafe_manager = PowerSafeManager(cache_dir, config)
        
        # Initialize GPU manager
        gpu_manager = EnhancedGPUManager(args.gpu_ids)
        
        # Smart file scanning with PowerSafe prioritization
        logger.info("Scanning for input files with PowerSafe prioritization...")
        
        if config.powersafe:
            video_files, gpx_files, new_videos, new_gpx = powersafe_manager.scan_for_new_files(args.directory)
        else:
            # Standard file scanning
            video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
            video_files = sorted(list(set(video_files)))
            
            gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
            gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
            gpx_files = sorted(list(set(gpx_files)))
            new_videos, new_gpx = [], []
        
        logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
        
        if not video_files or not gpx_files:
            raise RuntimeError("Need both video and GPX files")
        
        # Load existing results in PowerSafe mode
        existing_results = {}
        if config.powersafe:
            existing_results = powersafe_manager.load_existing_results()
            if existing_results:
                logger.info(f"PowerSafe: Loaded {len(existing_results)} existing correlation results")
        
        # Process videos with PowerSafe support
        logger.info("Processing videos with enhanced parallel processing...")
        video_cache_path = cache_dir / "enhanced_video_features.pkl"
        
        video_features = {}
        if video_cache_path.exists() and not args.force:
            logger.info("Loading cached video features...")
            try:
                with open(video_cache_path, 'rb') as f:
                    video_features = pickle.load(f)
                logger.info(f"Loaded {len(video_features)} cached video features")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                video_features = {}
        
        # Process missing videos
        videos_to_process = [v for v in video_files if v not in video_features or video_features[v] is None]
        
        if videos_to_process:
            logger.info(f"Processing {len(videos_to_process)} videos in parallel...")
            
            # Prepare arguments for parallel processing (FIXED: include powersafe_manager)
            video_args = [(video_path, gpu_manager, config, powersafe_manager) for video_path in videos_to_process]
            
            # Use ThreadPoolExecutor for better GPU sharing
            with ThreadPoolExecutor(max_workers=config.parallel_videos) as executor:
                futures = [executor.submit(process_video_parallel, arg) for arg in video_args]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                    video_path, features = future.result()
                    video_features[video_path] = features
                    
                    # Periodic cache save
                    if len([f for f in futures if f.done()]) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            
            logger.info("Video processing complete")
        
        # Process GPX files with PowerSafe support
        logger.info("Processing GPX files...")
        gpx_cache_path = cache_dir / "enhanced_gpx_features.pkl"
        
        gpx_database = {}
        if gpx_cache_path.exists() and not args.force:
            logger.info("Loading cached GPX features...")
            try:
                with open(gpx_cache_path, 'rb') as f:
                    gpx_database = pickle.load(f)
                logger.info(f"Loaded {len(gpx_database)} cached GPX features")
            except Exception as e:
                logger.warning(f"Failed to load GPX cache: {e}")
                gpx_database = {}
        
        # Process missing GPX files
        missing_gpx = [g for g in gpx_files if g not in gpx_database]
        
        if missing_gpx or args.force:
            processor = RobustGPXProcessor(config, powersafe_manager)
            new_gpx_results = processor.process_gpx_files(gpx_files)
            gpx_database.update(new_gpx_results)
            
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_database, f)
            
            logger.info("GPX processing complete")
        
        # Perform enhanced correlation with PowerSafe
        logger.info("Starting enhanced correlation analysis...")
        
        # Filter valid features
        valid_videos = {k: v for k, v in video_features.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None and 'features' in v}
        
        logger.info(f"Valid features: {len(valid_videos)} videos, {len(valid_gpx)} GPX tracks")
        
        if not valid_videos or not valid_gpx:
            raise RuntimeError("No valid features for correlation")
        
        # Initialize similarity engine
        similarity_engine = EnhancedSimilarityEngine(config)
        
        # Compute correlations with PowerSafe incremental saves
        results = existing_results.copy()  # Start with existing results
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        # Calculate how many correlations already exist
        existing_correlations = sum(len(result.get('matches', [])) for result in existing_results.values())
        
        # Reset PowerSafe manager counter if continuing
        if config.powersafe:
            powersafe_manager.correlation_counter = existing_correlations
        
        with tqdm(total=total_comparisons, desc="Computing correlations") as pbar:
            # Update progress bar for existing correlations
            if existing_correlations > 0:
                pbar.update(existing_correlations)
            
            for video_path, video_features_data in valid_videos.items():
                # Skip if video already has complete correlation results
                if (video_path in results and 
                    len(results[video_path].get('matches', [])) >= min(args.top_k, len(valid_gpx))):
                    # Fast-forward progress bar for this video
                    pbar.update(len(valid_gpx))
                    continue
                
                matches = []
                
                for gpx_path, gpx_data in valid_gpx.items():
                    # Check if this specific correlation already exists
                    existing_match = None
                    if video_path in results:
                        existing_match = next(
                            (m for m in results[video_path].get('matches', []) if m['path'] == gpx_path),
                            None
                        )
                    
                    if existing_match:
                        matches.append(existing_match)
                        pbar.update(1)
                        continue
                    
                    gpx_features = gpx_data['features']
                    
                    try:
                        similarities = similarity_engine.compute_enhanced_similarity(
                            video_features_data, gpx_features
                        )
                        
                        match_info = {
                            'path': gpx_path,
                            'combined_score': similarities['combined'],
                            'motion_score': similarities['motion_dynamics'],
                            'temporal_score': similarities['temporal_correlation'],
                            'statistical_score': similarities['statistical_profile'],
                            'spatial_score': similarities['spatial_features'],
                            'behavioral_score': similarities['behavioral_patterns'],
                            'quality': similarities['quality'],
                            'distance': gpx_data.get('distance', 0),
                            'duration': gpx_data.get('duration', 0),
                            'avg_speed': gpx_data.get('avg_speed', 0)
                        }
                        
                        matches.append(match_info)
                        
                        # PowerSafe: Add to pending correlations
                        if config.powersafe:
                            powersafe_manager.add_pending_correlation(video_path, gpx_path, match_info)
                        
                    except Exception as e:
                        logger.debug(f"Correlation failed for {video_path} vs {gpx_path}: {e}")
                        match_info = {
                            'path': gpx_path,
                            'combined_score': 0.0,
                            'quality': 'failed'
                        }
                        matches.append(match_info)
                        
                        if config.powersafe:
                            powersafe_manager.add_pending_correlation(video_path, gpx_path, match_info)
                    
                    pbar.update(1)
                
                # Sort by score and keep top K
                matches.sort(key=lambda x: x['combined_score'], reverse=True)
                results[video_path] = {'matches': matches[:args.top_k]}
                
                # PowerSafe: Mark video correlation as done
                if config.powersafe and matches:
                    best_match = matches[0]
                    powersafe_manager.mark_video_correlation_done(
                        video_path, best_match['combined_score'], best_match['path']
                    )
                
                # Log best match
                if matches:
                    best = matches[0]
                    logger.info(f"Best match for {Path(video_path).name}: "
                              f"{Path(best['path']).name} "
                              f"(score: {best['combined_score']:.3f}, quality: {best['quality']})")
        
        # Final PowerSafe save
        if config.powersafe:
            powersafe_manager.save_incremental_results(powersafe_manager.pending_results)
            logger.info("PowerSafe: Final incremental save completed")
        
        # Save final results
        results_path = output_dir / "enhanced_correlations.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Generate comprehensive report
        report_data = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'powersafe_enabled': config.powersafe,
                'total_videos': len(video_files),
                'total_gpx': len(gpx_files),
                'valid_videos': len(valid_videos),
                'valid_gpx': len(valid_gpx),
                'gpu_ids': args.gpu_ids,
                'config': config.__dict__
            },
            'results': results
        }
        
        # Add PowerSafe status if enabled
        if config.powersafe:
            report_data['powersafe_status'] = powersafe_manager.get_processing_status()
        
        with open(output_dir / "enhanced_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate summary statistics
        successful_matches = sum(1 for r in results.values() 
                               if r['matches'] and r['matches'][0]['combined_score'] > 0.1)
        
        excellent_matches = sum(1 for r in results.values() 
                              if r['matches'] and r['matches'][0].get('quality') == 'excellent')
        
        good_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'good')
        
        # Calculate average scores
        all_scores = []
        for r in results.values():
            if r['matches']:
                all_scores.append(r['matches'][0]['combined_score'])
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        
        # PowerSafe cleanup
        if config.powersafe:
            powersafe_manager.cleanup_completed_entries()
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print(f"🎯 ENHANCED VIDEO-GPX CORRELATION SUMMARY")
        print(f"{'='*80}")
        print(f"Processing Mode: {'PowerSafe' if config.powersafe else 'Standard'}")
        print(f"Videos Processed: {len(valid_videos)}/{len(video_files)}")
        print(f"GPX Tracks Processed: {len(valid_gpx)}/{len(gpx_files)}")
        print(f"Total Correlations: {len(results)}")
        print(f"Successful Correlations: {successful_matches}/{len(results)} ({100*successful_matches/max(len(results), 1):.1f}%)")
        print(f"Quality Distribution:")
        print(f"  🟢 Excellent (≥0.8): {excellent_matches}")
        print(f"  🟡 Good (≥0.6): {good_matches}")
        print(f"  🔴 Other: {len(results) - excellent_matches - good_matches}")
        print(f"Average Correlation Score: {avg_score:.3f}")
        
        if config.powersafe:
            status = powersafe_manager.get_processing_status()
            print(f"\nPowerSafe Status:")
            print(f"  Total Saved Correlations: {status.get('total_correlations', 0)}")
            print(f"  Results saved to: {powersafe_manager.results_path}")
            print(f"  Progress database: {powersafe_manager.db_path}")
        
        print(f"\nOutput Files:")
        print(f"  📊 Results: {results_path}")
        print(f"  📋 Report: {output_dir / 'enhanced_report.json'}")
        print(f"  💾 Cache: {cache_dir}")
        print(f"{'='*80}")
        
        logger.info("Enhanced correlation system completed successfully")
        
        # Display top 5 matches for demonstration
        print(f"\n🏆 TOP 5 CORRELATIONS:")
        print(f"{'='*80}")
        
        # Get top correlations across all videos
        all_correlations = []
        for video_path, result in results.items():
            if result['matches']:
                best_match = result['matches'][0]
                all_correlations.append((
                    Path(video_path).name,
                    Path(best_match['path']).name,
                    best_match['combined_score'],
                    best_match.get('quality', 'unknown')
                ))
        
        # Sort by score and display top 5
        all_correlations.sort(key=lambda x: x[2], reverse=True)
        for i, (video, gpx, score, quality) in enumerate(all_correlations[:5], 1):
            quality_emoji = {'excellent': '🟢', 'good': '🟡', 'fair': '🟠', 'poor': '🔴'}.get(quality, '⚪')
            print(f"{i}. {video} ↔ {gpx}")
            print(f"   Score: {score:.3f} | Quality: {quality_emoji} {quality}")
        
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        if config.powersafe:
            logger.info("PowerSafe: Progress has been saved and can be resumed")
        print("\n⚠️  Process interrupted. PowerSafe progress has been saved." if config.powersafe else "\n⚠️  Process interrupted.")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Enhanced system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        if config.powersafe:
            logger.info("PowerSafe: Partial progress has been saved")
            print(f"\n❌ Error occurred: {e}")
            print("PowerSafe: Partial progress has been saved and can be resumed with --powersafe flag")
        
        sys.exit(1)
    
    finally:
        # Cleanup temporary directories
        try:
            if 'decoder' in locals():
                decoder.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
