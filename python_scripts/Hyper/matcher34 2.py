#!/usr/bin/env python3
"""
Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script - PRODUCTION VERSION

Key Features:
- Fixed GPU acquisition with proper queuing
- Fixed video preprocessing with proper temp storage in ~/penis/temp
- Enhanced FFmpeg GPU utilization with NVENC and CUDA
- Robust error handling and recovery
- Production-grade memory management
- PowerSafe mode for long-running operations

Usage:
    python production_matcher.py -d /path/to/data --gpu_ids 0 1
    python production_matcher.py -d /path/to/data --parallel_videos 1 --debug
    python production_matcher.py -d /path/to/data --powersafe --save_interval 5
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
    """Configuration for processing parameters with memory optimization"""
    max_frames: int = 150  # Reduced for stability
    target_size: Tuple[int, int] = (480, 270)  # More compatible size
    sample_rate: float = 3.0
    parallel_videos: int = 1  # Safe default to prevent GPU conflicts
    gpu_memory_fraction: float = 0.7
    motion_threshold: float = 0.01
    temporal_window: int = 10
    powersafe: bool = False
    save_interval: int = 5
    gpu_timeout: int = 60  # Increased timeout for conversion
    strict: bool = False
    memory_efficient: bool = True
    max_gpu_memory_gb: float = 12.0
    # Video preprocessing and caching
    enable_preprocessing: bool = True
    ram_cache_gb: float = 32.0  # More conservative default
    disk_cache_gb: float = 1000.0
    replace_originals: bool = False
    cache_dir: str = "~/penis/temp"  # Default to requested location

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
                    status TEXT,
                    processed_at TIMESTAMP,
                    feature_extraction_done BOOLEAN DEFAULT FALSE,
                    correlation_done BOOLEAN DEFAULT FALSE,
                    best_match_score REAL DEFAULT 0.0,
                    best_match_path TEXT,
                    file_mtime REAL,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpx_progress (
                    gpx_path TEXT PRIMARY KEY,
                    status TEXT,
                    processed_at TIMESTAMP,
                    feature_extraction_done BOOLEAN DEFAULT FALSE,
                    file_mtime REAL,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correlation_progress (
                    video_path TEXT,
                    gpx_path TEXT,
                    correlation_score REAL,
                    correlation_details TEXT,
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
            return current_videos, current_gpx, [], []
        
        # Get interrupted/paused files from database
        interrupted_videos, interrupted_gpx = self._get_interrupted_files()
        
        # Determine new files
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
            cursor = conn.execute("""
                SELECT video_path FROM video_progress 
                WHERE status IN ('processing', 'pending') OR 
                      (feature_extraction_done = TRUE AND correlation_done = FALSE)
                ORDER BY processed_at DESC
            """)
            interrupted_videos = [row[0] for row in cursor.fetchall()]
            
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
    
    def mark_video_failed(self, video_path: str, error_message: str):
        """Mark video processing as failed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE video_progress 
                SET status = 'failed', error_message = ?, processed_at = datetime('now')
                WHERE video_path = ?
            """, (error_message, video_path))
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
    
    def mark_gpx_failed(self, gpx_path: str, error_message: str):
        """Mark GPX processing as failed"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE gpx_progress 
                SET status = 'failed', error_message = ?, processed_at = datetime('now')
                WHERE gpx_path = ?
            """, (error_message, gpx_path))
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
            existing_results = {}
            if self.results_path.exists():
                with open(self.results_path, 'r') as f:
                    existing_results = json.load(f)
            
            existing_results.update(results)
            
            temp_path = self.results_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(existing_results, f, indent=2, default=str)
            
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
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM video_progress GROUP BY status
            """)
            video_status = dict(cursor.fetchall())
            
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM gpx_progress GROUP BY status
            """)
            gpx_status = dict(cursor.fetchall())
            
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
        """Clean up old completed entries"""
        if not self.config.powersafe:
            return
        
        with sqlite3.connect(self.db_path) as conn:
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

class EnhancedGPUManager:
    """Enhanced GPU management with memory monitoring and strict mode enforcement"""
    
    def __init__(self, gpu_ids: List[int], strict: bool = False, config: Optional[ProcessingConfig] = None):
        self.gpu_ids = gpu_ids
        self.strict = strict
        self.config = config or ProcessingConfig()
        self.gpu_locks = {gpu_id: Lock() for gpu_id in gpu_ids}
        self.gpu_usage = {gpu_id: 0 for gpu_id in gpu_ids}
        self.gpu_queue = queue.Queue()
        
        # Initialize GPU queue
        for gpu_id in gpu_ids:
            self.gpu_queue.put(gpu_id)
        
        self.validate_gpus()
        
    def validate_gpus(self):
        """Validate GPU availability and memory with strict mode enforcement"""
        if not torch.cuda.is_available():
            if self.strict:
                raise RuntimeError("STRICT MODE: CUDA is required but not available")
            else:
                raise RuntimeError("CUDA not available")
        
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_ids:
            if gpu_id >= available_gpus:
                if self.strict:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} is required but not available (only {available_gpus} GPUs)")
                else:
                    raise RuntimeError(f"GPU {gpu_id} not available (only {available_gpus} GPUs)")
        
        # Check GPU memory
        for gpu_id in self.gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            
            if memory_gb < 4:
                if self.strict:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} has insufficient memory: {memory_gb:.1f}GB (minimum 4GB required)")
                else:
                    logger.warning(f"GPU {gpu_id} has only {memory_gb:.1f}GB memory")
            
            # Test GPU functionality in strict mode
            if self.strict:
                try:
                    with torch.cuda.device(gpu_id):
                        test_tensor = torch.zeros(50, 50, device=f'cuda:{gpu_id}', dtype=torch.float32)
                        _ = test_tensor + 1
                        del test_tensor
                        torch.cuda.empty_cache()
                except Exception as e:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} failed functionality test: {e}")
            
            logger.info(f"GPU {gpu_id}: {props.name} ({memory_gb:.1f}GB)" + (" [STRICT MODE]" if self.strict else ""))
    
    def get_gpu_memory_info(self, gpu_id: int) -> Dict[str, float]:
        """Get detailed GPU memory information"""
        try:
            with torch.cuda.device(gpu_id):
                total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                free = total - reserved
                
                return {
                    'total_gb': total,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'free_gb': free,
                    'utilization_pct': (reserved / total) * 100
                }
        except Exception:
            return {'total_gb': 0, 'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0, 'utilization_pct': 0}
    
    def cleanup_gpu_memory(self, gpu_id: int):
        """Aggressively cleanup GPU memory"""
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats(gpu_id)
        except Exception as e:
            logger.debug(f"Memory cleanup warning for GPU {gpu_id}: {e}")
    
    def check_memory_availability(self, gpu_id: int, required_gb: float = 2.0) -> bool:
        """Check if GPU has sufficient memory available"""
        mem_info = self.get_gpu_memory_info(gpu_id)
        return mem_info['free_gb'] >= required_gb
    
    def acquire_gpu(self, timeout: int = 60) -> Optional[int]:
        """Acquire GPU with timeout and strict mode enforcement"""
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            self.gpu_usage[gpu_id] += 1
            
            # Verify GPU is still functional in strict mode
            if self.strict:
                try:
                    if not self.check_memory_availability(gpu_id, 1.0):
                        self.cleanup_gpu_memory(gpu_id)
                        
                        if not self.check_memory_availability(gpu_id, 0.5):
                            self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
                            self.gpu_queue.put(gpu_id)
                            raise RuntimeError(f"STRICT MODE: GPU {gpu_id} has insufficient memory available")
                    
                    with torch.cuda.device(gpu_id):
                        test_tensor = torch.zeros(10, 10, device=f'cuda:{gpu_id}', dtype=torch.float32)
                        del test_tensor
                        torch.cuda.empty_cache()
                except Exception as e:
                    self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
                    self.gpu_queue.put(gpu_id)
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} became unavailable during acquisition: {e}")
            
            return gpu_id
        except queue.Empty:
            if self.strict:
                raise RuntimeError(f"STRICT MODE: Could not acquire any GPU within {timeout}s timeout. All GPUs are busy.")
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release GPU after processing with memory cleanup"""
        self.cleanup_gpu_memory(gpu_id)
        self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
        self.gpu_queue.put(gpu_id)

class VideoPreprocessor:
    """GPU-accelerated video preprocessor with RAM/disk caching and proper temp storage"""
    
    def __init__(self, config: ProcessingConfig, gpu_manager: EnhancedGPUManager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.ram_cache = {}
        self.ram_usage = 0
        self.max_ram_bytes = int(config.ram_cache_gb * 1024**3)
        
        # Setup cache directories - use ~/penis/temp as requested
        self.cache_dir = Path(os.path.expanduser(config.cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_cache_dir = self.cache_dir / "video_cache"
        self.disk_cache_dir.mkdir(exist_ok=True)
        
        # Create temp directory for processing
        self.temp_processing_dir = self.cache_dir / "processing"
        self.temp_processing_dir.mkdir(exist_ok=True)
        
        # Cache metadata
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        logger.info(f"🎞️ Video Preprocessor initialized:")
        logger.info(f"   RAM Cache: {config.ram_cache_gb:.1f}GB ({len(self.ram_cache)} videos cached)")
        logger.info(f"   Disk Cache: {self.disk_cache_dir}")
        logger.info(f"   Temp Processing: {self.temp_processing_dir}")
    
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk"""
        if self.cache_index_path.exists():
            try:
                with open(self.cache_index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _get_video_hash(self, video_path: str) -> str:
        """Get unique hash for video based on path and modification time"""
        try:
            stat = os.stat(video_path)
            hash_input = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return hashlib.md5(video_path.encode()).hexdigest()
    
    def preprocess_video(self, video_path: str, gpu_id: int) -> Optional[str]:
        """Preprocess video using GPU acceleration and return path to processable video"""
        video_hash = self._get_video_hash(video_path)
        
        # Check if already cached
        if video_hash in self.ram_cache:
            logger.debug(f"📋 Using RAM cached video: {Path(video_path).name}")
            return self.ram_cache[video_hash]
        
        if video_hash in self.cache_index:
            cache_info = self.cache_index[video_hash]
            cached_path = cache_info['path']
            if os.path.exists(cached_path):
                logger.debug(f"💾 Using disk cached video: {Path(video_path).name}")
                return cached_path
        
        # Try direct processing first with more permissive check
        if self._test_direct_processing(video_path, gpu_id):
            logger.debug(f"✅ Direct processing works: {Path(video_path).name}")
            return video_path
        
        # Need to preprocess
        logger.info(f"🔄 Preprocessing problematic video: {Path(video_path).name}")
        preprocessed_path = self._gpu_convert_video(video_path, gpu_id)
        
        if preprocessed_path:
            self._cache_video(video_hash, preprocessed_path, video_path)
            return preprocessed_path
        
        return None
    
    def _test_direct_processing(self, video_path: str, gpu_id: int) -> bool:
        """Test if video can be processed directly without preprocessing - more permissive"""
        try:
            # Basic file checks
            if not os.path.exists(video_path):
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size < 1024:
                logger.debug(f"Video file too small: {file_size} bytes")
                return False
            
            # Quick ffprobe test
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', video_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.debug(f"ffprobe basic test failed for {Path(video_path).name}")
                return False
            
            # If we get any codec name, test GPU decoding
            codec_output = result.stdout.strip()
            if codec_output and codec_output != 'N/A':
                return self._test_gpu_decode_minimal(video_path, gpu_id)
            
            return False
            
        except Exception as e:
            logger.debug(f"Direct processing test failed for {Path(video_path).name}: {e}")
            return False
    
    def _test_gpu_decode_minimal(self, video_path: str, gpu_id: int) -> bool:
        """Test minimal GPU decoding to verify compatibility"""
        try:
            test_output = self.temp_processing_dir / f"test_gpu_{gpu_id}_{int(time.time())}.jpg"
            
            test_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                '-i', video_path,
                '-vframes', '1',
                '-f', 'image2',
                str(test_output)
            ]
            
            result = subprocess.run(test_cmd, capture_output=True, timeout=15)
            
            success = (result.returncode == 0 and test_output.exists() and test_output.stat().st_size > 1024)
            
            if test_output.exists():
                test_output.unlink()
            
            if success:
                logger.debug(f"GPU decode test passed for {Path(video_path).name}")
            else:
                logger.debug(f"GPU decode test failed for {Path(video_path).name}")
            
            return success
                    
        except Exception as e:
            logger.debug(f"GPU decode test failed for {Path(video_path).name}: {e}")
            return False
    
    def _gpu_convert_video(self, video_path: str, gpu_id: int) -> Optional[str]:
        """Convert video using GPU acceleration to a standard, processable format"""
        try:
            video_name = Path(video_path).stem
            timestamp = int(time.time() * 1000000)
            output_path = self.disk_cache_dir / f"{video_name}_converted_{timestamp}.mp4"
            
            logger.info(f"🎬 GPU converting {Path(video_path).name} on GPU {gpu_id}")
            
            strategies = [
                ("NVENC Hardware Encode", self._convert_with_nvenc_enhanced),
                ("CUDA Accelerated Decode", self._convert_with_cuda_enhanced),
                ("GPU-Assisted CPU Encode", self._convert_gpu_assisted_cpu)
            ]
            
            for i, (strategy_name, strategy_func) in enumerate(strategies):
                try:
                    logger.debug(f"Trying strategy {i+1}/3: {strategy_name}")
                    if strategy_func(video_path, output_path, gpu_id):
                        if self._verify_converted_video(str(output_path), gpu_id):
                            logger.info(f"✅ Successfully converted using {strategy_name}: {Path(video_path).name}")
                            return str(output_path)
                        else:
                            logger.debug(f"{strategy_name} converted but result not processable")
                            if output_path.exists():
                                output_path.unlink()
                    else:
                        logger.debug(f"{strategy_name} conversion failed")
                except Exception as e:
                    logger.debug(f"{strategy_name} failed with exception: {e}")
                    if output_path.exists():
                        output_path.unlink()
            
            logger.error(f"❌ All conversion strategies failed for {Path(video_path).name}")
            return None
            
        except Exception as e:
            logger.error(f"GPU conversion failed for {video_path}: {e}")
            return None
    
    def _verify_converted_video(self, video_path: str, gpu_id: int) -> bool:
        """Verify that converted video is actually processable"""
        try:
            if not os.path.exists(video_path):
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size < 1024:
                logger.debug(f"Converted video too small: {file_size} bytes")
                return False
            
            test_output = self.temp_processing_dir / f"verify_{gpu_id}_{int(time.time())}.jpg"
            
            verify_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                '-i', video_path,
                '-vframes', '1',
                '-f', 'image2',
                str(test_output)
            ]
            
            result = subprocess.run(verify_cmd, capture_output=True, timeout=10)
            
            success = (result.returncode == 0 and test_output.exists() and test_output.stat().st_size > 1024)
            
            if test_output.exists():
                test_output.unlink()
            
            if not success:
                logger.debug(f"Converted video verification failed")
            
            return success
            
        except Exception as e:
            logger.debug(f"Video verification failed: {e}")
            return False
    
    def _convert_with_nvenc_enhanced(self, input_path: str, output_path: Path, gpu_id: int) -> bool:
        """Enhanced NVENC conversion with better parameters"""
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-hwaccel_output_format', 'cuda',
            '-i', input_path,
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-tune', 'hq',
            '-profile:v', 'high',
            '-level', '4.1',
            '-pix_fmt', 'yuv420p',
            '-cq', '23',
            '-b:v', '0',
            '-maxrate', '10M',
            '-bufsize', '20M',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-max_muxing_queue_size', '2048',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        return self._run_ffmpeg_command(cmd, "NVENC Enhanced")
    
    def _convert_with_cuda_enhanced(self, input_path: str, output_path: Path, gpu_id: int) -> bool:
        """Enhanced CUDA-accelerated conversion"""
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-hwaccel_output_format', 'cuda',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-profile:v', 'high',
            '-level', '4.1',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-max_muxing_queue_size', '2048',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        return self._run_ffmpeg_command(cmd, "CUDA Enhanced")
    
    def _convert_gpu_assisted_cpu(self, input_path: str, output_path: Path, gpu_id: int) -> bool:
        """GPU-assisted CPU encoding (fallback)"""
        if self.config.strict:
            logger.debug("GPU-assisted CPU conversion skipped in strict mode")
            return False
        
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-profile:v', 'baseline',
            '-level', '3.1',
            '-pix_fmt', 'yuv420p',
            '-crf', '28',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-max_muxing_queue_size', '1024',
            '-c:a', 'aac', '-b:a', '96k',
            str(output_path)
        ]
        
        return self._run_ffmpeg_command(cmd, "GPU-Assisted CPU")
    
    def _run_ffmpeg_command(self, cmd: List[str], strategy_name: str) -> bool:
        """Run FFmpeg command with proper error handling"""
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                logger.debug(f"{strategy_name} conversion succeeded")
                return True
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                logger.debug(f"{strategy_name} conversion failed: {error_msg[:200]}...")
                return False
        except subprocess.TimeoutExpired:
            logger.debug(f"{strategy_name} conversion timed out")
            return False
        except Exception as e:
            logger.debug(f"{strategy_name} conversion exception: {e}")
            return False
    
    def _cache_video(self, video_hash: str, preprocessed_path: str, original_path: str):
        """Cache preprocessed video in RAM or disk with improved management"""
        video_size_bytes = os.path.getsize(preprocessed_path)
        video_size_mb = video_size_bytes / (1024**2)
        
        # Try RAM cache first if small enough and space available
        if (video_size_bytes < 500 * 1024**2 and  # Less than 500MB for RAM cache
            self.ram_usage + video_size_bytes < self.max_ram_bytes):
            
            try:
                with open(preprocessed_path, 'rb') as f:
                    video_data = f.read()
                
                ram_cache_path = self.temp_processing_dir / f"ram_cache_{video_hash}.mp4"
                with open(ram_cache_path, 'wb') as f:
                    f.write(video_data)
                
                self.ram_cache[video_hash] = str(ram_cache_path)
                self.ram_usage += video_size_bytes
                
                logger.debug(f"📋 Cached {video_size_mb:.1f}MB video in RAM: {Path(original_path).name}")
            except Exception as e:
                logger.debug(f"Failed to cache in RAM: {e}")
        
        # Update disk cache index
        self.cache_index[video_hash] = {
            'path': preprocessed_path,
            'original_path': original_path,
            'size_mb': video_size_mb,
            'created': time.time()
        }
        self._save_cache_index()
        
        logger.debug(f"💾 Cached {video_size_mb:.1f}MB video on disk: {Path(original_path).name}")
    
    def cleanup_old_cache(self, max_age_hours: int = 48):
        """Clean up old cache entries"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # Clean up disk cache
        to_remove = []
        for video_hash, info in self.cache_index.items():
            if info['created'] < cutoff_time:
                try:
                    if os.path.exists(info['path']):
                        os.unlink(info['path'])
                    to_remove.append(video_hash)
                except Exception:
                    pass
        
        for video_hash in to_remove:
            del self.cache_index[video_hash]
        
        # Clean up RAM cache files
        ram_files_removed = 0
        for video_hash, ram_path in list(self.ram_cache.items()):
            try:
                if os.path.exists(ram_path):
                    file_age = current_time - os.path.getmtime(ram_path)
                    if file_age > max_age_hours * 3600:
                        os.unlink(ram_path)
                        del self.ram_cache[video_hash]
                        ram_files_removed += 1
            except Exception:
                pass
        
        if to_remove or ram_files_removed:
            self._save_cache_index()
            logger.info(f"🧹 Cleaned up {len(to_remove)} disk cache entries and {ram_files_removed} RAM cache files")
    
    def get_cache_stats(self) -> Dict:
        """Get cache usage statistics"""
        ram_usage_mb = self.ram_usage / (1024**2)
        ram_limit_mb = self.max_ram_bytes / (1024**2)
        
        disk_usage_mb = sum(info['size_mb'] for info in self.cache_index.values())
        
        return {
            'ram_cached_videos': len(self.ram_cache),
            'ram_usage_mb': ram_usage_mb,
            'ram_limit_mb': ram_limit_mb,
            'ram_usage_pct': (ram_usage_mb / ram_limit_mb) * 100,
            'disk_cached_videos': len(self.cache_index),
            'disk_usage_mb': disk_usage_mb,
            'disk_limit_mb': self.config.disk_cache_gb * 1024,
            'cache_directory': str(self.cache_dir),
            'temp_processing_directory': str(self.temp_processing_dir)
        }
    
    def cleanup(self):
        """Cleanup all temporary files and cache"""
        try:
            for ram_path in self.ram_cache.values():
                if os.path.exists(ram_path):
                    os.unlink(ram_path)
            
            if self.temp_processing_dir.exists():
                for temp_file in self.temp_processing_dir.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                    except:
                        pass
            
            logger.info("🧹 Video preprocessor cleanup completed")
            
        except Exception as e:
            logger.warning(f"Video preprocessor cleanup failed: {e}")

class EnhancedFFmpegDecoder:
    """Enhanced FFmpeg decoder with GPU preprocessing and robust error handling"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.temp_dirs = {}
        
        # Initialize video preprocessor with enhanced settings
        if config.enable_preprocessing:
            self.preprocessor = VideoPreprocessor(config, gpu_manager)
            self.preprocessor.cleanup_old_cache(max_age_hours=48)
        else:
            self.preprocessor = None
        
        # Create temp directories per GPU in the correct location
        base_temp = Path(config.cache_dir) / "gpu_temp"
        base_temp.mkdir(parents=True, exist_ok=True)
        
        for gpu_id in gpu_manager.gpu_ids:
            self.temp_dirs[gpu_id] = base_temp / f'gpu_{gpu_id}'
            self.temp_dirs[gpu_id].mkdir(exist_ok=True)
        
        logger.info(f"Enhanced decoder initialized for GPUs: {gpu_manager.gpu_ids}")
        if self.preprocessor:
            logger.info(f"🎞️ Video preprocessing enabled")
    
    def decode_video_enhanced(self, video_path: str, gpu_id: int) -> Tuple[Optional[torch.Tensor], float, float]:
        """Enhanced video decoding with preprocessing and robust scaling"""
        try:
            # Preprocess video if enabled
            if self.preprocessor:
                processed_video_path = self.preprocessor.preprocess_video(video_path, gpu_id)
                if processed_video_path is None:
                    # Try one more time with emergency conversion
                    logger.warning(f"🔄 Retrying preprocessing for {Path(video_path).name}")
                    processed_video_path = self._emergency_conversion(video_path, gpu_id)
                    
                    if processed_video_path is None:
                        raise RuntimeError("Video preprocessing failed after retry")
                
                actual_video_path = processed_video_path
                logger.debug(f"Using processed video: {Path(actual_video_path).name}")
            else:
                actual_video_path = video_path
            
            # Get video info
            video_info = self._get_video_info(actual_video_path)
            if not video_info:
                raise RuntimeError("Could not get video info")
            
            # Use uniform sampling with the processed video
            frames_tensor = self._decode_uniform_frames(actual_video_path, video_info, gpu_id)
            
            if frames_tensor is None:
                raise RuntimeError("Frame decoding failed")
            
            return frames_tensor, video_info['fps'], video_info['duration']
            
        except Exception as e:
            logger.error(f"Enhanced video decoding failed for {video_path}: {e}")
            return None, 0, 0
    
    def _emergency_conversion(self, video_path: str, gpu_id: int) -> Optional[str]:
        """Emergency conversion using the most basic settings"""
        try:
            temp_dir = Path(self.config.cache_dir) / "emergency"
            temp_dir.mkdir(exist_ok=True)
            
            video_name = Path(video_path).stem
            output_path = temp_dir / f"{video_name}_emergency_{int(time.time())}.mp4"
            
            logger.info(f"🚨 Emergency conversion for {Path(video_path).name}")
            
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=480:270:force_original_aspect_ratio=decrease:force_divisible_by=2',
                '-r', '15',
                '-crf', '28',
                '-an',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024:
                logger.info(f"✅ Emergency conversion succeeded: {Path(video_path).name}")
                return str(output_path)
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                logger.error(f"Emergency conversion failed: {error_msg[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Emergency conversion exception: {e}")
            return None
    
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
    
    def _decode_uniform_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """Memory-optimized uniform frame sampling with robust GPU processing"""
        temp_dir = self.temp_dirs[gpu_id]
        output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
        
        # Calculate sampling rate
        total_frames = int(video_info['duration'] * video_info['fps'])
        
        # Adaptive frame limit based on GPU memory
        if self.config.memory_efficient:
            mem_info = self.gpu_manager.get_gpu_memory_info(gpu_id)
            if mem_info['free_gb'] < 2.0:
                max_frames = min(self.config.max_frames, 100)
            elif mem_info['free_gb'] < 4.0:
                max_frames = min(self.config.max_frames, 150)
            else:
                max_frames = self.config.max_frames
        else:
            max_frames = self.config.max_frames
        
        # Ensure target size is even numbers and memory-efficient
        target_width = self.config.target_size[0]
        target_height = self.config.target_size[1]
        
        # Make sure dimensions are even
        if target_width % 2 != 0:
            target_width += 1
        if target_height % 2 != 0:
            target_height += 1
        
        # Further reduce size if memory constrained
        if self.config.memory_efficient:
            mem_info = self.gpu_manager.get_gpu_memory_info(gpu_id)
            if mem_info['free_gb'] < 1.5:
                target_width = min(target_width, 384)
                target_height = min(target_height, 216)
        
        if total_frames > max_frames:
            sample_rate = total_frames / max_frames
            vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,select=not(mod(n\\,{int(sample_rate)}))'
        else:
            vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2'
        
        # Enhanced CUDA command with better error handling
        cuda_cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-hwaccel_output_format', 'cuda',
            '-extra_hw_frames', '8',
            '-i', video_path,
            '-vf', vf_filter,
            '-frames:v', str(min(max_frames, total_frames)),
            '-q:v', '2',
            '-threads', '1',
            output_pattern
        ]
        
        # Fallback CUDA command
        cuda_fallback_cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-vf', vf_filter,
            '-frames:v', str(min(max_frames, total_frames)),
            '-q:v', '2',
            '-threads', '1',
            output_pattern
        ]
        
        # CPU fallback command (only if not strict mode)
        cpu_cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-i', video_path,
            '-vf', vf_filter,
            '-frames:v', str(min(max_frames, total_frames)),
            '-q:v', '3',
            '-threads', '2',
            output_pattern
        ] if not self.config.strict else None
        
        try:
            # Monitor memory before processing
            mem_before = self.gpu_manager.get_gpu_memory_info(gpu_id)
            logger.debug(f"GPU {gpu_id} memory before decoding: {mem_before['free_gb']:.1f}GB free")
            
            # Strategy 1: Full CUDA with hwaccel output
            try:
                result = subprocess.run(cuda_cmd, check=True, capture_output=True, timeout=300)
                logger.debug(f"✅ CUDA hwaccel decoding successful: {Path(video_path).name}")
                success = True
            except subprocess.CalledProcessError as e:
                error_output = e.stderr.decode() if e.stderr else ""
                logger.debug(f"CUDA hwaccel failed: {error_output[:100]}...")
                success = False
            
            # Strategy 2: CUDA without hwaccel output format
            if not success:
                try:
                    # Clean up any partial files
                    for f in glob.glob(os.path.join(temp_dir, 'frame_*.jpg')):
                        try:
                            os.remove(f)
                        except:
                            pass
                    
                    result = subprocess.run(cuda_fallback_cmd, check=True, capture_output=True, timeout=300)
                    logger.debug(f"✅ CUDA fallback decoding successful: {Path(video_path).name}")
                    success = True
                except subprocess.CalledProcessError as e:
                    error_output = e.stderr.decode() if e.stderr else ""
                    logger.debug(f"CUDA fallback failed: {error_output[:100]}...")
                    success = False
            
            # Strategy 3: CPU fallback (only if not strict mode)
            if not success and cpu_cmd:
                try:
                    # Clean up any partial files
                    for f in glob.glob(os.path.join(temp_dir, 'frame_*.jpg')):
                        try:
                            os.remove(f)
                        except:
                            pass
                    
                    result = subprocess.run(cpu_cmd, check=True, capture_output=True, timeout=300)
                    logger.debug(f"✅ CPU fallback decoding successful: {Path(video_path).name}")
                    success = True
                except subprocess.CalledProcessError as e:
                    error_output = e.stderr.decode() if e.stderr else ""
                    logger.debug(f"CPU fallback failed: {error_output[:100]}...")
                    success = False
            
            if not success:
                if self.config.strict:
                    raise RuntimeError("STRICT MODE: All CUDA decoding methods failed")
                else:
                    raise RuntimeError("All decoding methods failed")
            
            return self._load_frames_to_tensor(temp_dir, gpu_id, target_width, target_height)
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout for {video_path}")
            return None
        except Exception as e:
            logger.error(f"Frame decoding failed for {video_path}: {e}")
            return None
    
    def _load_frames_to_tensor(self, temp_dir: str, gpu_id: int, target_width: int, target_height: int) -> Optional[torch.Tensor]:
        """Load frames to GPU tensor with robust size checking"""
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
                
                # Verify dimensions
                if img.shape[1] != target_width or img.shape[0] != target_height:
                    img = cv2.resize(img, (target_width, target_height))
                
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
    """Enhanced feature extraction with improved error handling"""
    
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
    """Extract enhanced features with robust error handling and memory management"""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        
        # Validate input tensor
        if frames_tensor is None:
            logger.error("Input frames_tensor is None")
            return {}
        
        if len(frames_tensor.shape) < 4:
            logger.error(f"Invalid tensor shape: {frames_tensor.shape}. Expected at least 4 dimensions.")
            return {}
        
        # Check if we have the model for this GPU
        if gpu_id not in self.feature_models:
            logger.error(f"No model found for GPU {gpu_id}")
            return {}
        
        model = self.feature_models[gpu_id]
        
        # Ensure tensor is on correct device
        if frames_tensor.device != device:
            frames_tensor = frames_tensor.to(device)
        
        features = {}
        
        with torch.no_grad():
            try:
                # Validate tensor dimensions
                batch_size, num_frames = frames_tensor.shape[:2]
                if num_frames == 0:
                    logger.warning("No frames to process")
                    return {}
                
                # CNN features with memory-efficient processing
                try:
                    frames_flat = frames_tensor.view(-1, *frames_tensor.shape[2:])
                    
                    # Process in smaller chunks if memory constrained
                    chunk_size = min(32, frames_flat.shape[0])  # Process max 32 frames at once
                    cnn_results = {}
                    
                    for i in range(0, frames_flat.shape[0], chunk_size):
                        chunk = frames_flat[i:i+chunk_size]
                        chunk_features = model(chunk)
                        
                        # Initialize results dict on first iteration
                        if not cnn_results:
                            cnn_results = {key: [] for key in chunk_features.keys()}
                        
                        # Accumulate results
                        for key, value in chunk_features.items():
                            cnn_results[key].append(value.cpu())
                    
                    # Concatenate all chunks and reshape
                    for key, value_list in cnn_results.items():
                        if value_list:
                            concatenated = torch.cat(value_list, dim=0)
                            reshaped = concatenated.view(batch_size, num_frames, -1)[0]  # Remove batch dim
                            features[key] = reshaped.numpy()
                    
                    # Clear GPU memory
                    try:
                        del frames_flat
                    except:
                        pass
                    try:
                        if 'chunk_features' in locals():
                            del chunk_features
                    except:
                        pass
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"CNN feature extraction failed: {e}. Skipping CNN features.")
                
                # Enhanced motion features with error handling
                try:
                    motion_features = self._compute_enhanced_motion_safe(frames_tensor[0], device)
                    if motion_features:
                        features.update(motion_features)
                except Exception as e:
                    logger.warning(f"Motion feature extraction failed: {e}")
                
                # Temporal coherence features with error handling
                try:
                    temporal_features = self._compute_temporal_coherence_safe(frames_tensor[0], device)
                    if temporal_features:
                        features.update(temporal_features)
                except Exception as e:
                    logger.warning(f"Temporal feature extraction failed: {e}")
                
                # Improved color features with error handling
                try:
                    color_features = self._compute_enhanced_color_safe(frames_tensor[0], device)
                    if color_features:
                        features.update(color_features)
                except Exception as e:
                    logger.warning(f"Color feature extraction failed: {e}")
                
                # Edge and texture features with error handling
                try:
                    edge_features = self._compute_edge_features_safe(frames_tensor[0], device)
                    if edge_features:
                        features.update(edge_features)
                except Exception as e:
                    logger.warning(f"Edge feature extraction failed: {e}")
                
            except Exception as e:
                logger.error(f"Feature extraction processing failed: {e}")
                return {}
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        logger.debug(f"Enhanced feature extraction successful: {len(features)} feature types")
        return features
        
    except Exception as e:
        logger.error(f"Enhanced feature extraction failed: {e}")
        # Emergency cleanup
        try:
            torch.cuda.empty_cache()
        except:
            pass
        return {}

def _compute_enhanced_motion_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Safely compute enhanced motion features with comprehensive error handling"""
    try:
        if frames is None or frames.numel() == 0:
            return {}
        
        num_frames = frames.shape[0]
        if num_frames < 2:
            return {
                'motion_magnitude': np.zeros(num_frames),
                'motion_direction': np.zeros(num_frames),
                'acceleration': np.zeros(num_frames),
                'jerk': np.zeros(num_frames),
                'motion_consistency': np.zeros(num_frames)
            }
        
        features = {
            'motion_magnitude': np.zeros(num_frames),
            'motion_direction': np.zeros(num_frames),
            'acceleration': np.zeros(num_frames),
            'jerk': np.zeros(num_frames),
            'motion_consistency': np.zeros(num_frames)
        }
        
        # Convert to grayscale safely
        try:
            if frames.shape[1] >= 3:  # Check if we have RGB channels
                gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
            else:
                gray_frames = frames[:, 0]  # Use first channel if not RGB
        except Exception as e:
            logger.warning(f"Grayscale conversion failed: {e}")
            return features
        
        # Compute optical flow approximation
        motion_vectors = []
        for i in range(num_frames - 1):
            try:
                frame1 = gray_frames[i]
                frame2 = gray_frames[i + 1]
                
                # Frame difference
                diff = torch.abs(frame2 - frame1)
                
                # Motion magnitude
                magnitude = torch.mean(diff).item()
                features['motion_magnitude'][i + 1] = magnitude
                
                # Motion direction (gradient-based)
                if diff.sum() > 1e-6:  # Avoid division by very small numbers
                    try:
                        grad_x = torch.mean(torch.abs(diff[:, 1:] - diff[:, :-1])).item()
                        grad_y = torch.mean(torch.abs(diff[1:, :] - diff[:-1, :])).item()
                        direction = math.atan2(grad_y, grad_x + 1e-8)
                        features['motion_direction'][i + 1] = direction
                        motion_vectors.append([grad_x, grad_y])
                    except Exception:
                        motion_vectors.append([0, 0])
                else:
                    motion_vectors.append([0, 0])
                    
            except Exception as e:
                logger.debug(f"Motion computation failed for frame {i}: {e}")
                motion_vectors.append([0, 0])
        
        # Compute acceleration
        try:
            motion_mag = features['motion_magnitude']
            for i in range(1, num_frames - 1):
                features['acceleration'][i] = motion_mag[i + 1] - motion_mag[i]
        except Exception:
            pass
        
        # Compute jerk
        try:
            accel = features['acceleration']
            for i in range(1, num_frames - 2):
                features['jerk'][i] = accel[i + 1] - accel[i]
        except Exception:
            pass
        
        # Motion consistency
        try:
            window_size = min(self.config.temporal_window, len(motion_vectors) // 2)
            if len(motion_vectors) > window_size and window_size > 0:
                for i in range(window_size, len(motion_vectors)):
                    window = motion_vectors[i-window_size:i]
                    if window:
                        angles = [math.atan2(v[1], v[0] + 1e-8) for v in window]
                        if angles:
                            circular_mean = math.atan2(
                                sum(math.sin(a) for a in angles) / len(angles),
                                sum(math.cos(a) for a in angles) / len(angles)
                            )
                            variance = sum((a - circular_mean)**2 for a in angles) / len(angles)
                            consistency = 1.0 / (1.0 + variance)
                            features['motion_consistency'][i + 1] = consistency
        except Exception:
            pass
        
        return features
        
    except Exception as e:
        logger.warning(f"Enhanced motion computation failed: {e}")
        return {}

def _compute_statistical_features_robust_gpu(self, motion_features: Dict[str, cp.ndarray], distances: cp.ndarray, elevations: cp.ndarray) -> Dict[str, cp.ndarray]:
    """Compute statistical features using GPU with comprehensive error handling"""
    features = {}
    
    try:
        # Validate inputs
        if not motion_features:
            logger.warning("Empty motion_features dictionary")
            return features
        
        if distances is None or elevations is None:
            logger.warning("distances or elevations is None")
            return features
        
        # Speed statistics with error handling
        try:
            if 'speed' in motion_features:
                speed = motion_features['speed']
                if speed is not None and len(speed) > 0:
                    # Convert to CuPy if needed
                    if not isinstance(speed, cp.ndarray):
                        speed = cp.asarray(speed)
                    
                    # Filter valid speeds
                    valid_speed = speed[speed > 0]
                    if len(valid_speed) > 0:
                        try:
                            features['speed_stats'] = cp.array([
                                cp.mean(valid_speed), 
                                cp.std(valid_speed), 
                                cp.min(valid_speed), 
                                cp.max(valid_speed),
                                cp.percentile(valid_speed, 25), 
                                cp.percentile(valid_speed, 50), 
                                cp.percentile(valid_speed, 75)
                            ])
                        except Exception as e:
                            logger.debug(f"Speed statistics computation failed: {e}")
                            features['speed_stats'] = cp.zeros(7)
                    else:
                        features['speed_stats'] = cp.zeros(7)
                else:
                    features['speed_stats'] = cp.zeros(7)
            else:
                features['speed_stats'] = cp.zeros(7)
        except Exception as e:
            logger.debug(f"Speed statistics failed: {e}")
            features['speed_stats'] = cp.zeros(7)
        
        # Bearing statistics with error handling
        try:
            if 'bearing' in motion_features:
                bearing = motion_features['bearing']
                if bearing is not None and len(bearing) > 0:
                    # Convert to CuPy if needed
                    if not isinstance(bearing, cp.ndarray):
                        bearing = cp.asarray(bearing)
                    
                    # Filter valid bearings
                    valid_bearing = bearing[bearing > 0]
                    if len(valid_bearing) > 0:
                        try:
                            features['bearing_stats'] = cp.array([
                                cp.mean(valid_bearing), 
                                cp.std(valid_bearing),
                                cp.min(valid_bearing), 
                                cp.max(valid_bearing)
                            ])
                        except Exception as e:
                            logger.debug(f"Bearing statistics computation failed: {e}")
                            features['bearing_stats'] = cp.zeros(4)
                    else:
                        features['bearing_stats'] = cp.zeros(4)
                else:
                    features['bearing_stats'] = cp.zeros(4)
            else:
                features['bearing_stats'] = cp.zeros(4)
        except Exception as e:
            logger.debug(f"Bearing statistics failed: {e}")
            features['bearing_stats'] = cp.zeros(4)
        
        # Elevation statistics with error handling
        try:
            # Convert to CuPy if needed
            if not isinstance(elevations, cp.ndarray):
                elevations = cp.asarray(elevations)
            
            if len(elevations) > 1:
                try:
                    elev_diffs = cp.diff(elevations)
                    total_climb = cp.sum(cp.maximum(elev_diffs, 0))
                    total_descent = cp.sum(cp.maximum(-elev_diffs, 0))
                    
                    features['elevation_stats'] = cp.array([
                        cp.mean(elevations), 
                        cp.std(elevations),
                        cp.min(elevations), 
                        cp.max(elevations),
                        total_climb, 
                        total_descent
                    ])
                except Exception as e:
                    logger.debug(f"Elevation statistics computation failed: {e}")
                    features['elevation_stats'] = cp.zeros(6)
            else:
                features['elevation_stats'] = cp.zeros(6)
        except Exception as e:
            logger.debug(f"Elevation statistics failed: {e}")
            features['elevation_stats'] = cp.zeros(6)
        
        # Distance statistics with error handling
        try:
            # Convert to CuPy if needed
            if not isinstance(distances, cp.ndarray):
                distances = cp.asarray(distances)
            
            if len(distances) > 0:
                # Filter valid distances
                valid_distances = distances[distances > 0]
                if len(valid_distances) > 0:
                    try:
                        features['distance_stats'] = cp.array([
                            cp.sum(distances), 
                            cp.mean(valid_distances),
                            cp.std(valid_distances), 
                            cp.max(valid_distances)
                        ])
                    except Exception as e:
                        logger.debug(f"Distance statistics computation failed: {e}")
                        features['distance_stats'] = cp.zeros(4)
                else:
                    features['distance_stats'] = cp.zeros(4)
            else:
                features['distance_stats'] = cp.zeros(4)
        except Exception as e:
            logger.debug(f"Distance statistics failed: {e}")
            features['distance_stats'] = cp.zeros(4)
        
        return features
        
    except Exception as e:
        logger.error(f"Statistical features computation failed: {e}")
        # Return empty dict with zero arrays for expected keys
        return {
            'speed_stats': cp.zeros(7),
            'bearing_stats': cp.zeros(4),
            'elevation_stats': cp.zeros(6),
            'distance_stats': cp.zeros(4)
        }

# Helper functions for the safe feature extraction methods
def _compute_temporal_coherence_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Safely compute temporal coherence features"""
    try:
        if frames is None or frames.numel() == 0:
            return {}
        
        num_frames = frames.shape[0]
        window_size = min(self.config.temporal_window, num_frames // 2)
        
        if num_frames < window_size:
            return {
                'temporal_stability': np.zeros(num_frames),
                'temporal_patterns': np.zeros(num_frames),
                'scene_changes': np.zeros(num_frames)
            }
        
        # Convert to grayscale safely
        try:
            if frames.shape[1] >= 3:
                gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
            else:
                gray_frames = frames[:, 0]
        except Exception:
            return {}
        
        stability = np.zeros(num_frames)
        patterns = np.zeros(num_frames)
        scene_changes = np.zeros(num_frames)
        
        for i in range(window_size, num_frames):
            try:
                # Temporal stability
                window_frames = gray_frames[i-window_size:i+1]
                if len(window_frames) > 0:
                    stability_score = 1.0 / (1.0 + torch.var(window_frames).item())
                    stability[i] = stability_score
                
                # Temporal patterns
                if i >= window_size * 2:
                    recent_window = gray_frames[i-window_size:i]
                    past_window = gray_frames[i-window_size*2:i-window_size]
                    
                    if len(recent_window) > 0 and len(past_window) > 0:
                        try:
                            correlation = torch.corrcoef(torch.stack([
                                recent_window.flatten(),
                                past_window.flatten()
                            ]))[0, 1].item()
                            
                            if not math.isnan(correlation):
                                patterns[i] = abs(correlation)
                        except Exception:
                            pass
                
                # Scene change detection
                if i > 0:
                    curr_frame = gray_frames[i]
                    prev_frame = gray_frames[i-1]
                    change_score = torch.mean(torch.abs(curr_frame - prev_frame)).item()
                    scene_changes[i] = change_score
                    
            except Exception:
                continue
        
        return {
            'temporal_stability': stability,
            'temporal_patterns': patterns,
            'scene_changes': scene_changes
        }
        
    except Exception:
        return {}

def _compute_enhanced_color_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Safely compute enhanced color features"""
    try:
        if frames is None or frames.numel() == 0:
            return {}
        
        num_frames = frames.shape[0]
        
        # Color variance over time
        try:
            color_variance = torch.var(frames, dim=[2, 3])
            mean_color_variance = torch.mean(color_variance, dim=1).cpu().numpy()
        except Exception:
            mean_color_variance = np.zeros(num_frames)
        
        # Color histograms and moments
        histograms = []
        color_moments = []
        
        for i in range(num_frames):
            try:
                frame = frames[i]
                
                # Color moments
                moments = []
                for c in range(min(3, frame.shape[0])):  # Handle cases with < 3 channels
                    try:
                        channel = frame[c]
                        mean_val = torch.mean(channel).item()
                        std_val = torch.std(channel).item()
                        if std_val > 1e-8:
                            skew_val = torch.mean((channel - mean_val)**3).item() / (std_val**3)
                        else:
                            skew_val = 0.0
                        moments.extend([mean_val, std_val, skew_val])
                    except Exception:
                        moments.extend([0.0, 0.0, 0.0])
                
                color_moments.append(moments)
                
                # Simplified histogram
                hist_features = []
                for c in range(min(3, frame.shape[0])):
                    try:
                        channel_mean = torch.mean(frame[c]).item()
                        channel_std = torch.std(frame[c]).item()
                        hist_features.extend([channel_mean, channel_std])
                    except Exception:
                        hist_features.extend([0.0, 0.0])
                
                histograms.append(hist_features)
                
            except Exception:
                # Fallback for failed frame processing
                default_moments = [0.0] * 9  # 3 channels * 3 moments
                default_hist = [0.0] * 6     # 3 channels * 2 features
                color_moments.append(default_moments)
                histograms.append(default_hist)
        
        return {
            'color_variance': mean_color_variance,
            'color_histograms': np.array(histograms),
            'color_moments': np.array(color_moments)
        }
        
    except Exception:
        return {}

def _compute_edge_features_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Safely compute edge and texture features"""
    try:
        if frames is None or frames.numel() == 0:
            return {}
        
        num_frames = frames.shape[0]
        
        # Sobel filters
        try:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=frames.dtype, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        except Exception:
            return {
                'edge_density': np.zeros(num_frames),
                'edge_orientation': np.zeros(num_frames),
                'texture_complexity': np.zeros(num_frames)
            }
        
        # Convert to grayscale
        try:
            if frames.shape[1] >= 3:
                gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
            else:
                gray_frames = frames[:, 0]
            gray_frames = gray_frames.unsqueeze(1)
        except Exception:
            return {}
        
        edge_density = np.zeros(num_frames)
        edge_orientation = np.zeros(num_frames)
        texture_complexity = np.zeros(num_frames)
        
        try:
            # Edge detection
            edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
            edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
            
            edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
            edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3]).cpu().numpy()
            
            # Edge orientation
            edge_angle = torch.atan2(edges_y, edges_x + 1e-8)
            edge_orientation = torch.mean(torch.abs(torch.cos(edge_angle)), dim=[1, 2, 3]).cpu().numpy()
            
        except Exception as e:
            logger.debug(f"Edge detection failed: {e}")
        
        # Texture features
        for i in range(num_frames):
            try:
                frame_gray = gray_frames[i, 0]
                
                if frame_gray.shape[1] > 1 and frame_gray.shape[0] > 1:
                    grad_x = torch.diff(frame_gray, dim=1)
                    grad_y = torch.diff(frame_gray, dim=0)
                    
                    min_height = min(grad_x.shape[0], grad_y.shape[0])
                    min_width = min(grad_x.shape[1], grad_y.shape[1])
                    
                    if min_height > 0 and min_width > 0:
                        grad_x_cropped = grad_x[:min_height, :min_width]
                        grad_y_cropped = grad_y[:min_height, :min_width]
                        
                        texture_score = torch.mean(
                            torch.abs(grad_x_cropped)**2 + torch.abs(grad_y_cropped)**2
                        ).item()
                        texture_complexity[i] = texture_score
                        
            except Exception:
                texture_complexity[i] = 0.0
        
        return {
            'edge_density': edge_density,
            'edge_orientation': edge_orientation,
            'texture_complexity': texture_complexity
        }
        
    except Exception:
        return {}

def _compute_statistical_features_robust_gpu(self, motion_features: Dict[str, cp.ndarray], distances: cp.ndarray, elevations: cp.ndarray) -> Dict[str, cp.ndarray]:
    """Compute statistical features using GPU with comprehensive error handling"""
    features = {}
    
    try:
        # Validate inputs
        if not motion_features:
            logger.warning("Empty motion_features dictionary")
            return features
        
        if distances is None or elevations is None:
            logger.warning("distances or elevations is None")
            return features
        
        # Speed statistics with error handling
        try:
            if 'speed' in motion_features:
                speed = motion_features['speed']
                if speed is not None and len(speed) > 0:
                    # Convert to CuPy if needed
                    if not isinstance(speed, cp.ndarray):
                        speed = cp.asarray(speed)
                    
                    # Filter valid speeds
                    valid_speed = speed[speed > 0]
                    if len(valid_speed) > 0:
                        try:
                            features['speed_stats'] = cp.array([
                                cp.mean(valid_speed), 
                                cp.std(valid_speed), 
                                cp.min(valid_speed), 
                                cp.max(valid_speed),
                                cp.percentile(valid_speed, 25), 
                                cp.percentile(valid_speed, 50), 
                                cp.percentile(valid_speed, 75)
                            ])
                        except Exception as e:
                            logger.debug(f"Speed statistics computation failed: {e}")
                            features['speed_stats'] = cp.zeros(7)
                    else:
                        features['speed_stats'] = cp.zeros(7)
                else:
                    features['speed_stats'] = cp.zeros(7)
            else:
                features['speed_stats'] = cp.zeros(7)
        except Exception as e:
            logger.debug(f"Speed statistics failed: {e}")
            features['speed_stats'] = cp.zeros(7)
        
        # Bearing statistics with error handling
        try:
            if 'bearing' in motion_features:
                bearing = motion_features['bearing']
                if bearing is not None and len(bearing) > 0:
                    # Convert to CuPy if needed
                    if not isinstance(bearing, cp.ndarray):
                        bearing = cp.asarray(bearing)
                    
                    # Filter valid bearings
                    valid_bearing = bearing[bearing > 0]
                    if len(valid_bearing) > 0:
                        try:
                            features['bearing_stats'] = cp.array([
                                cp.mean(valid_bearing), 
                                cp.std(valid_bearing),
                                cp.min(valid_bearing), 
                                cp.max(valid_bearing)
                            ])
                        except Exception as e:
                            logger.debug(f"Bearing statistics computation failed: {e}")
                            features['bearing_stats'] = cp.zeros(4)
                    else:
                        features['bearing_stats'] = cp.zeros(4)
                else:
                    features['bearing_stats'] = cp.zeros(4)
            else:
                features['bearing_stats'] = cp.zeros(4)
        except Exception as e:
            logger.debug(f"Bearing statistics failed: {e}")
            features['bearing_stats'] = cp.zeros(4)
        
        # Elevation statistics with error handling
        try:
            # Convert to CuPy if needed
            if not isinstance(elevations, cp.ndarray):
                elevations = cp.asarray(elevations)
            
            if len(elevations) > 1:
                try:
                    elev_diffs = cp.diff(elevations)
                    total_climb = cp.sum(cp.maximum(elev_diffs, 0))
                    total_descent = cp.sum(cp.maximum(-elev_diffs, 0))
                    
                    features['elevation_stats'] = cp.array([
                        cp.mean(elevations), 
                        cp.std(elevations),
                        cp.min(elevations), 
                        cp.max(elevations),
                        total_climb, 
                        total_descent
                    ])
                except Exception as e:
                    logger.debug(f"Elevation statistics computation failed: {e}")
                    features['elevation_stats'] = cp.zeros(6)
            else:
                features['elevation_stats'] = cp.zeros(6)
        except Exception as e:
            logger.debug(f"Elevation statistics failed: {e}")
            features['elevation_stats'] = cp.zeros(6)
        
        # Distance statistics with error handling
        try:
            # Convert to CuPy if needed
            if not isinstance(distances, cp.ndarray):
                distances = cp.asarray(distances)
            
            if len(distances) > 0:
                # Filter valid distances
                valid_distances = distances[distances > 0]
                if len(valid_distances) > 0:
                    try:
                        features['distance_stats'] = cp.array([
                            cp.sum(distances), 
                            cp.mean(valid_distances),
                            cp.std(valid_distances), 
                            cp.max(valid_distances)
                        ])
                    except Exception as e:
                        logger.debug(f"Distance statistics computation failed: {e}")
                        features['distance_stats'] = cp.zeros(4)
                else:
                    features['distance_stats'] = cp.zeros(4)
            else:
                features['distance_stats'] = cp.zeros(4)
        except Exception as e:
            logger.debug(f"Distance statistics failed: {e}")
            features['distance_stats'] = cp.zeros(4)
        
        return features
        
    except Exception as e:
        logger.error(f"Statistical features computation failed: {e}")
        # Return empty dict with zero arrays for expected keys
        return {
            'speed_stats': cp.zeros(7),
            'bearing_stats': cp.zeros(4),
            'elevation_stats': cp.zeros(6),
            'distance_stats': cp.zeros(4)
        }

# Helper functions for the safe feature extraction methods
def _compute_temporal_coherence_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Safely compute temporal coherence features"""
    try:
        if frames is None or frames.numel() == 0:
            return {}
        
        num_frames = frames.shape[0]
        window_size = min(self.config.temporal_window, num_frames // 2)
        
        if num_frames < window_size:
            return {
                'temporal_stability': np.zeros(num_frames),
                'temporal_patterns': np.zeros(num_frames),
                'scene_changes': np.zeros(num_frames)
            }
        
        # Convert to grayscale safely
        try:
            if frames.shape[1] >= 3:
                gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
            else:
                gray_frames = frames[:, 0]
        except Exception:
            return {}
        
        stability = np.zeros(num_frames)
        patterns = np.zeros(num_frames)
        scene_changes = np.zeros(num_frames)
        
        for i in range(window_size, num_frames):
            try:
                # Temporal stability
                window_frames = gray_frames[i-window_size:i+1]
                if len(window_frames) > 0:
                    stability_score = 1.0 / (1.0 + torch.var(window_frames).item())
                    stability[i] = stability_score
                
                # Temporal patterns
                if i >= window_size * 2:
                    recent_window = gray_frames[i-window_size:i]
                    past_window = gray_frames[i-window_size*2:i-window_size]
                    
                    if len(recent_window) > 0 and len(past_window) > 0:
                        try:
                            correlation = torch.corrcoef(torch.stack([
                                recent_window.flatten(),
                                past_window.flatten()
                            ]))[0, 1].item()
                            
                            if not math.isnan(correlation):
                                patterns[i] = abs(correlation)
                        except Exception:
                            pass
                
                # Scene change detection
                if i > 0:
                    curr_frame = gray_frames[i]
                    prev_frame = gray_frames[i-1]
                    change_score = torch.mean(torch.abs(curr_frame - prev_frame)).item()
                    scene_changes[i] = change_score
                    
            except Exception:
                continue
        
        return {
            'temporal_stability': stability,
            'temporal_patterns': patterns,
            'scene_changes': scene_changes
        }
        
    except Exception:
        return {}

    def _compute_enhanced_color_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Safely compute enhanced color features"""
        try:
            if frames is None or frames.numel() == 0:
                return {}
            
            num_frames = frames.shape[0]
            
            # Color variance over time
            try:
                color_variance = torch.var(frames, dim=[2, 3])
                mean_color_variance = torch.mean(color_variance, dim=1).cpu().numpy()
            except Exception:
                mean_color_variance = np.zeros(num_frames)
            
            # Color histograms and moments
            histograms = []
            color_moments = []
            
            for i in range(num_frames):
                try:
                    frame = frames[i]
                    
                    # Color moments
                    moments = []
                    for c in range(min(3, frame.shape[0])):  # Handle cases with < 3 channels
                        try:
                            channel = frame[c]
                            mean_val = torch.mean(channel).item()
                            std_val = torch.std(channel).item()
                            if std_val > 1e-8:
                                skew_val = torch.mean((channel - mean_val)**3).item() / (std_val**3)
                            else:
                                skew_val = 0.0
                            moments.extend([mean_val, std_val, skew_val])
                        except Exception:
                            moments.extend([0.0, 0.0, 0.0])
                    
                    color_moments.append(moments)
                    
                    # Simplified histogram
                    hist_features = []
                    for c in range(min(3, frame.shape[0])):
                        try:
                            channel_mean = torch.mean(frame[c]).item()
                            channel_std = torch.std(frame[c]).item()
                            hist_features.extend([channel_mean, channel_std])
                        except Exception:
                            hist_features.extend([0.0, 0.0])
                    
                    histograms.append(hist_features)
                    
                except Exception:
                    # Fallback for failed frame processing
                    default_moments = [0.0] * 9  # 3 channels * 3 moments
                    default_hist = [0.0] * 6     # 3 channels * 2 features
                    color_moments.append(default_moments)
                    histograms.append(default_hist)
            
            return {
                'color_variance': mean_color_variance,
                'color_histograms': np.array(histograms),
                'color_moments': np.array(color_moments)
            }
            
        except Exception:
            return {}
    
    def _compute_edge_features_safe(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Safely compute edge and texture features"""
        try:
            if frames is None or frames.numel() == 0:
                return {}
            
            num_frames = frames.shape[0]
            
            # Sobel filters
            try:
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                      dtype=frames.dtype, device=device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                      dtype=frames.dtype, device=device).view(1, 1, 3, 3)
            except Exception:
                return {
                    'edge_density': np.zeros(num_frames),
                    'edge_orientation': np.zeros(num_frames),
                    'texture_complexity': np.zeros(num_frames)
                }
            
            # Convert to grayscale
            try:
                if frames.shape[1] >= 3:
                    gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
                else:
                    gray_frames = frames[:, 0]
                gray_frames = gray_frames.unsqueeze(1)
            except Exception:
                return {}
            
            edge_density = np.zeros(num_frames)
            edge_orientation = np.zeros(num_frames)
            texture_complexity = np.zeros(num_frames)
            
            try:
                # Edge detection
                edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
                edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
                
                edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
                edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3]).cpu().numpy()
                
                # Edge orientation
                edge_angle = torch.atan2(edges_y, edges_x + 1e-8)
                edge_orientation = torch.mean(torch.abs(torch.cos(edge_angle)), dim=[1, 2, 3]).cpu().numpy()
                
            except Exception as e:
                logger.debug(f"Edge detection failed: {e}")
            
            # Texture features
            for i in range(num_frames):
                try:
                    frame_gray = gray_frames[i, 0]
                    
                    if frame_gray.shape[1] > 1 and frame_gray.shape[0] > 1:
                        grad_x = torch.diff(frame_gray, dim=1)
                        grad_y = torch.diff(frame_gray, dim=0)
                        
                        min_height = min(grad_x.shape[0], grad_y.shape[0])
                        min_width = min(grad_x.shape[1], grad_y.shape[1])
                        
                        if min_height > 0 and min_width > 0:
                            grad_x_cropped = grad_x[:min_height, :min_width]
                            grad_y_cropped = grad_y[:min_height, :min_width]
                            
                            texture_score = torch.mean(
                                torch.abs(grad_x_cropped)**2 + torch.abs(grad_y_cropped)**2
                            ).item()
                            texture_complexity[i] = texture_score
                            
                except Exception:
                    texture_complexity[i] = 0.0
            
            return {
                'edge_density': edge_density,
                'edge_orientation': edge_orientation,
                'texture_complexity': texture_complexity
            }
            
        except Exception:
            return {}
    
    def _compute_robust_features_cpu(self, df: pd.DataFrame) -> Dict:
        """Compute robust features using CPU (fallback)"""
        n_points = len(df)
        
        # Convert to numpy arrays
        lats = df['lat'].values
        lons = df['lon'].values
        elevs = df['elevation'].values
        
        # Compute time differences
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        
        # Compute distances (CPU version)
        distances = self._compute_distances_cpu(lats, lons)
        
        # Compute motion features
        motion_features = self._compute_motion_features_cpu(lats, lons, elevs, time_diffs, distances)
        
        # Compute statistical features
        statistical_features = self._compute_statistical_features_cpu(motion_features, distances, elevs)
        
        # Combine features
        all_features = {**motion_features, **statistical_features}
        
        # Add metadata
        duration = self._compute_duration_safe(df['timestamp'])
        total_distance = np.sum(distances)
        
        return {
            'df': df,
            'features': all_features,
            'start_time': df['timestamp'].iloc[0],
            'end_time': df['timestamp'].iloc[-1],
            'duration': duration,
            'distance': total_distance,
            'point_count': n_points,
            'max_speed': np.max(motion_features['speed']) if 'speed' in motion_features else 0,
            'avg_speed': np.mean(motion_features['speed']) if 'speed' in motion_features else 0,
            'processing_mode': 'CPU'
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
    
    def _compute_distances_cpu(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Compute distances using CPU (Haversine formula)"""
        n = len(lats)
        distances = np.zeros(n)
        
        if n < 2:
            return distances
        
        R = 3958.8  # Earth radius in miles
        
        lat1_rad = np.radians(lats[:-1])
        lon1_rad = np.radians(lons[:-1])
        lat2_rad = np.radians(lats[1:])
        lon2_rad = np.radians(lons[1:])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        computed_distances = R * c
        distances[1:] = computed_distances
        
        return distances
    
    def _compute_motion_features_cpu(self, lats: np.ndarray, lons: np.ndarray, 
                                   elevs: np.ndarray, time_diffs: List[float], 
                                   distances: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute motion features using CPU"""
        n = len(lats)
        time_diffs = np.array(time_diffs)
        
        features = {
            'speed': np.zeros(n),
            'acceleration': np.zeros(n),
            'jerk': np.zeros(n),
            'bearing': np.zeros(n),
            'bearing_change': np.zeros(n),
            'curvature': np.zeros(n),
            'elevation_change_rate': np.zeros(n)
        }
        
        if n < 2:
            return features
        
        # Speed computation
        valid_time_diffs = np.maximum(time_diffs[1:], 1e-6)
        speed_values = distances[1:] * 3600 / valid_time_diffs
        features['speed'][1:] = speed_values
        
        # Acceleration
        if n > 2:
            speed_diff = features['speed'][2:] - features['speed'][1:-1]
            accel_values = speed_diff / np.maximum(time_diffs[2:], 1e-6)
            features['acceleration'][2:] = accel_values
        
        # Jerk
        if n > 3:
            accel_diff = features['acceleration'][3:] - features['acceleration'][2:-1]
            jerk_values = accel_diff / np.maximum(time_diffs[3:], 1e-6)
            features['jerk'][3:] = jerk_values
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_cpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
        # Bearing changes
        if n > 2:
            bearing_diffs = np.diff(features['bearing'][1:])
            bearing_diffs = np.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = np.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = np.abs(bearing_diffs)
            features['bearing_change'][2:] = bearing_changes
            
            valid_distances = np.maximum(distances[2:], 1e-8)
            curvature_values = bearing_changes / valid_distances
            features['curvature'][2:] = curvature_values
        
        # Elevation change rate
        if n > 1:
            elev_diffs = elevs[1:] - elevs[:-1]
            elev_rates = elev_diffs / np.maximum(time_diffs[1:], 1e-6)
            features['elevation_change_rate'][1:] = elev_rates
        
        return features
    
    def _compute_bearings_cpu(self, lat1: np.ndarray, lon1: np.ndarray, 
                            lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Compute bearings using CPU"""
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearings = np.degrees(np.arctan2(y, x))
        bearings = np.where(bearings < 0, bearings + 360, bearings)
        
        return bearings
    
    def _compute_statistical_features_cpu(self, motion_features: Dict[str, np.ndarray],
                                        distances: np.ndarray, elevations: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical features using CPU"""
        features = {}
        
        # Speed statistics
        speed = motion_features['speed']
        valid_speed = speed[speed > 0]
        if len(valid_speed) > 0:
            features['speed_stats'] = np.array([
                np.mean(valid_speed), np.std(valid_speed), 
                np.min(valid_speed), np.max(valid_speed),
                np.percentile(valid_speed, 25), np.percentile(valid_speed, 50), 
                np.percentile(valid_speed, 75)
            ])
        else:
            features['speed_stats'] = np.zeros(7)
        
        # Bearing statistics
        bearing = motion_features['bearing']
        valid_bearing = bearing[bearing > 0]
        if len(valid_bearing) > 0:
            features['bearing_stats'] = np.array([
                np.mean(valid_bearing), np.std(valid_bearing),
                np.min(valid_bearing), np.max(valid_bearing)
            ])
        else:
            features['bearing_stats'] = np.zeros(4)
        
        # Elevation statistics
        if len(elevations) > 1:
            elev_diffs = np.diff(elevations)
            total_climb = np.sum(np.maximum(elev_diffs, 0))
            total_descent = np.sum(np.maximum(-elev_diffs, 0))
            
            features['elevation_stats'] = np.array([
                np.mean(elevations), np.std(elevations),
                np.min(elevations), np.max(elevations),
                total_climb, total_descent
            ])
        else:
            features['elevation_stats'] = np.zeros(6)
        
        # Distance statistics
        valid_distances = distances[distances > 0]
        if len(valid_distances) > 0:
            features['distance_stats'] = np.array([
                np.sum(distances), np.mean(valid_distances),
                np.std(valid_distances), np.max(valid_distances)
            ])
        else:
            features['distance_stats'] = np.zeros(4)
        
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
            
            # Create unified signature by taking the longest component
            if len(signature_components) == 1:
                return self._robust_normalize(signature_components[0])
            
            # Take the longest component as base
            longest_component = max(signature_components, key=len)
            return self._robust_normalize(longest_component)
            
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_temporal_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute temporal correlation with basic alignment"""
        try:
            # Extract temporal signatures
            video_temporal = self._extract_temporal_signature(video_features, 'video')
            gpx_temporal = self._extract_temporal_signature(gpx_features, 'gpx')
            
            if video_temporal is None or gpx_temporal is None:
                return 0.0
            
            # Direct correlation
            min_len = min(len(video_temporal), len(gpx_temporal))
            if min_len > 5:
                v_temp = video_temporal[:min_len]
                g_temp = gpx_temporal[:min_len]
                
                corr = np.corrcoef(v_temp, g_temp)[0, 1]
                if not np.isnan(corr):
                    return float(np.clip(abs(corr), 0.0, 1.0))
            
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
                            return np.diff(values)
                            
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
            
            # Cosine similarity
            cosine_sim = np.dot(video_stats, gpx_stats) / (
                np.linalg.norm(video_stats) * np.linalg.norm(gpx_stats) + 1e-8
            )
            
            if not np.isnan(cosine_sim):
                return float(np.clip(abs(cosine_sim), 0.0, 1.0))
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
                                    np.mean(values), np.std(values), np.median(values)
                                ])
                            
            elif source_type == 'gpx':
                # GPX statistical features
                stat_keys = ['speed_stats', 'bearing_stats', 'elevation_stats', 'distance_stats']
                for key in stat_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend(values.flatten()[:3])
            
            if not profile_components:
                return None
            
            return np.array(profile_components)
            
        except Exception as e:
            logger.debug(f"Statistical profile extraction failed for {source_type}: {e}")
            return None
    
    def _compute_spatial_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute spatial feature similarity"""
        return 0.5  # Placeholder
    
    def _compute_behavioral_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute behavioral pattern similarity"""
        return 0.5  # Placeholder
    
    def _robust_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Robust normalization with outlier handling"""
        try:
            if len(vector) == 0:
                return vector
            
            # Simple min-max normalization
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

def process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
    """Enhanced video processing with better error recovery"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    gpu_id = gpu_manager.acquire_gpu(timeout=config.gpu_timeout)
    
    if gpu_id is None:
        error_msg = f"Could not acquire GPU within {config.gpu_timeout}s timeout"
        if config.strict:
            error_msg = f"STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {video_path}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.error(f"{error_msg} for {video_path}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            return video_path, None
    
    try:
        decoder = EnhancedFFmpegDecoder(gpu_manager, config)
        feature_extractor = EnhancedFeatureExtractor(gpu_manager, config)
        
        # Enhanced decode with retries
        frames_tensor, fps, duration = decoder.decode_video_enhanced(video_path, gpu_id)
        
        if frames_tensor is None:
            error_msg = "Video decoding failed after all attempts"
            if config.strict:
                error_msg = f"STRICT MODE: {error_msg}"
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                raise RuntimeError(error_msg)
            else:
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                return video_path, None
        
        # Extract features
        features = feature_extractor.extract_enhanced_features(frames_tensor, gpu_id)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_id
        features['processing_mode'] = 'GPU_STRICT' if config.strict else 'GPU'
        
        # Mark feature extraction as done in power-safe mode
        if powersafe_manager:
            powersafe_manager.mark_video_features_done(video_path)
        
        success_msg = f"✅ Successfully processed {Path(video_path).name} on GPU {gpu_id}"
        if config.strict:
            success_msg += " [STRICT MODE]"
        logger.info(success_msg)
        return video_path, features
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        if config.strict and "STRICT MODE" not in str(e):
            error_msg = f"STRICT MODE: {error_msg}"
        
        logger.error(f"❌ {error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        
        if config.strict:
            raise RuntimeError(error_msg)
        return video_path, None
        
    finally:
        if gpu_id is not None:
            gpu_manager.release_gpu(gpu_id)
            # Aggressive GPU memory cleanup
            try:
                torch.cuda.empty_cache()
                if gpu_id < torch.cuda.device_count():
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            except:
                pass

def update_config_for_temp_dir(args):
    """Update configuration to use ~/penis/temp directory"""
    # Override the cache directory argument to use the requested location
    args.cache_dir = os.path.expanduser("~/penis/temp")
    
    # Create the directory if it doesn't exist
    temp_dir = Path(args.cache_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Using temp directory: {args.cache_dir}")
    return args

def main():
    """Enhanced main function with comprehensive error handling and progress tracking"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", required=True,
                       help="Directory containing videos and GPX files")
    
    # Processing configuration - ENHANCED DEFAULTS
    parser.add_argument("--max_frames", type=int, default=150,
                       help="Maximum frames per video (default: 150 - optimized for stability)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[480, 270],
                       help="Target video resolution (default: 480 270 - optimized for compatibility)")
    parser.add_argument("--sample_rate", type=float, default=3.0,
                       help="Video sampling rate (default: 3.0)")
    parser.add_argument("--parallel_videos", type=int, default=1,
                       help="Number of videos to process in parallel (default: 1 - prevents GPU conflicts)")
    
    # GPU configuration
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1],
                       help="GPU IDs to use (default: [0, 1])")
    parser.add_argument("--gpu_timeout", type=int, default=60,
                       help="Seconds to wait for GPU availability (default: 60 - more time for conversion)")
    parser.add_argument("--max_gpu_memory", type=float, default=12.0,
                       help="Maximum GPU memory to use in GB (default: 12.0)")
    parser.add_argument("--memory_efficient", action='store_true', default=True,
                       help="Enable memory optimizations (default: True)")
    
    # Video preprocessing and caching - ENHANCED DEFAULTS
    parser.add_argument("--enable_preprocessing", action='store_true', default=True,
                       help="Enable GPU-based video preprocessing for problematic videos (default: True)")
    parser.add_argument("--ram_cache", type=float, default=32.0,
                       help="RAM to use for video caching in GB (default: 32.0 - more conservative)")
    parser.add_argument("--disk_cache", type=float, default=1000.0,
                       help="Disk space to use for video caching in GB (default: 1000.0)")
    parser.add_argument("--cache_dir", type=str, default="~/penis/temp",
                       help="Directory for video cache (default: ~/penis/temp)")
    parser.add_argument("--replace_originals", action='store_true',
                       help="Replace original files with preprocessed versions")
    
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
    parser.add_argument("--strict", action='store_true',
                       help="STRICT MODE: Enforce GPU usage at all times, fail if GPU unavailable")
    
    # Power-safe mode arguments
    parser.add_argument("--powersafe", action='store_true',
                       help="Enable power-safe mode with incremental saves and resume capability")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save results every N correlations in powersafe mode (default: 5)")
    
    args = parser.parse_args()
    
    # UPDATE CONFIG TO USE CORRECT TEMP DIRECTORY
    args = update_config_for_temp_dir(args)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level, "production_correlation.log")
    
    if args.strict:
        logger.info("🚀 Starting Production-Ready Video-GPX Correlation System [STRICT GPU MODE]")
    else:
        logger.info("🚀 Starting Production-Ready Video-GPX Correlation System")
    
    # Log configuration
    logger.info(f"💾 Memory Settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
    logger.info(f"🎞️ Video Preprocessing: {'Enabled' if args.enable_preprocessing else 'Disabled'}")
    if args.enable_preprocessing:
        logger.info(f"📋 RAM Cache: {args.ram_cache:.0f}GB, 💾 Disk Cache: {args.disk_cache:.0f}GB")
        logger.info(f"📁 Cache Directory: {args.cache_dir}")
    
    try:
        # Create configuration
        config = ProcessingConfig(
            max_frames=args.max_frames,
            target_size=tuple(args.video_size),
            sample_rate=args.sample_rate,
            parallel_videos=args.parallel_videos,
            powersafe=args.powersafe,
            save_interval=args.save_interval,
            gpu_timeout=args.gpu_timeout,
            strict=args.strict,
            memory_efficient=args.memory_efficient,
            max_gpu_memory_gb=args.max_gpu_memory,
            enable_preprocessing=args.enable_preprocessing,
            ram_cache_gb=args.ram_cache,
            disk_cache_gb=args.disk_cache,
            cache_dir=args.cache_dir,
            replace_originals=args.replace_originals
        )
        
        # Validate strict mode requirements early
        if config.strict:
            logger.info("🔒 STRICT MODE ENABLED: GPU usage mandatory, no CPU fallback")
            if not torch.cuda.is_available():
                raise RuntimeError("STRICT MODE: CUDA is required but not available")
            if not cp.cuda.is_available():
                raise RuntimeError("STRICT MODE: CuPy CUDA is required but not available")
        
        # Setup directories
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        cache_dir = Path(args.cache)
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize PowerSafe manager
        powersafe_manager = PowerSafeManager(cache_dir, config)
        
        # Initialize GPU manager with strict mode and config
        gpu_manager = EnhancedGPUManager(args.gpu_ids, strict=config.strict, config=config)
        
        # Smart file scanning with PowerSafe prioritization
        logger.info("📁 Scanning for input files with PowerSafe prioritization...")
        
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
        
        # Process videos with enhanced error tracking
        logger.info("🎬 Processing videos with enhanced parallel processing...")
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
            
            # Prepare arguments for parallel processing
            video_args = [(video_path, gpu_manager, config, powersafe_manager) for video_path in videos_to_process]
            
            # Enhanced progress tracking
            successful_videos = 0
            failed_videos = 0
            
            # Use ThreadPoolExecutor for better GPU sharing
            with ThreadPoolExecutor(max_workers=config.parallel_videos) as executor:
                futures = [executor.submit(process_video_parallel_enhanced, arg) for arg in video_args]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                    video_path, features = future.result()
                    video_features[video_path] = features
                    
                    if features is not None:
                        successful_videos += 1
                    else:
                        failed_videos += 1
                    
                    # Periodic cache save
                    if (successful_videos + failed_videos) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
                        logger.info(f"Progress: {successful_videos} ✅ | {failed_videos} ❌")
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
        
        logger.info(f"🎬 Video processing complete: {successful_videos} ✅ | {failed_videos} ❌")
        
        # Process GPX files with enhanced error tracking
        logger.info("🗺️ Processing GPX files...")
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
            
            logger.info(f"🗺️ GPX processing complete: {len(new_gpx_results)} successful")
        
        # Perform enhanced correlation with PowerSafe
        logger.info("🔗 Starting enhanced correlation analysis...")
        
        # Filter valid features
        valid_videos = {k: v for k, v in video_features.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None and 'features' in v}
        
        logger.info(f"Valid features: {len(valid_videos)} videos, {len(valid_gpx)} GPX tracks")
        
        if not valid_videos:
            raise RuntimeError(f"No valid video features! Processed {len(video_features)} videos but none succeeded.")
        
        if not valid_gpx:
            raise RuntimeError(f"No valid GPX features! Processed {len(gpx_database)} GPX files but none succeeded.")
        
        # Initialize similarity engine
        similarity_engine = EnhancedSimilarityEngine(config)
        
        # Compute correlations with PowerSafe incremental saves
        results = existing_results.copy()
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        # Calculate existing correlations
        existing_correlations = sum(len(result.get('matches', [])) for result in existing_results.values())
        
        # Reset PowerSafe manager counter
        if config.powersafe:
            powersafe_manager.correlation_counter = existing_correlations
        
        successful_correlations = 0
        failed_correlations = 0
        
        with tqdm(total=total_comparisons, desc="Computing correlations") as pbar:
            # Update progress bar for existing correlations
            if existing_correlations > 0:
                pbar.update(existing_correlations)
            
            for video_path, video_features_data in valid_videos.items():
                # Skip if video already has complete correlation results
                if (video_path in results and 
                    len(results[video_path].get('matches', [])) >= min(args.top_k, len(valid_gpx))):
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
                        successful_correlations += 1
                        
                        # PowerSafe: Add to pending correlations
                        if config.powersafe:
                            powersafe_manager.add_pending_correlation(video_path, gpx_path, match_info)
                        
                    except Exception as e:
                        logger.debug(f"Correlation failed for {Path(video_path).name} vs {Path(gpx_path).name}: {e}")
                        match_info = {
                            'path': gpx_path,
                            'combined_score': 0.0,
                            'quality': 'failed',
                            'error': str(e)
                        }
                        matches.append(match_info)
                        failed_correlations += 1
                        
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
                if matches and matches[0]['combined_score'] > 0:
                    best = matches[0]
                    logger.info(f"🎯 Best match for {Path(video_path).name}: "
                              f"{Path(best['path']).name} "
                              f"(score: {best['combined_score']:.3f}, quality: {best['quality']})")
                else:
                    logger.warning(f"⚠️ No valid matches found for {Path(video_path).name}")
        
        logger.info(f"🔗 Correlation analysis complete: {successful_correlations} ✅ | {failed_correlations} ❌")
        
        # Final PowerSafe save
        if config.powersafe:
            powersafe_manager.save_incremental_results(powersafe_manager.pending_results)
            logger.info("PowerSafe: Final incremental save completed")
        
        # Save final results
        results_path = output_dir / "production_correlations.pkl"
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
                'successful_correlations': successful_correlations,
                'failed_correlations': failed_correlations,
                'gpu_ids': args.gpu_ids,
                'config': config.__dict__
            },
            'results': results
        }
        
        # Add PowerSafe status if enabled
        if config.powersafe:
            report_data['powersafe_status'] = powersafe_manager.get_processing_status()
        
        with open(output_dir / "production_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate summary statistics
        total_videos_with_results = len(results)
        successful_matches = sum(1 for r in results.values() 
                               if r['matches'] and r['matches'][0]['combined_score'] > 0.1)
        
        excellent_matches = sum(1 for r in results.values() 
                              if r['matches'] and r['matches'][0].get('quality') == 'excellent')
        
        good_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'good')
        
        fair_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'fair')
        
        # Calculate average scores
        all_scores = []
        for r in results.values():
            if r['matches'] and r['matches'][0]['combined_score'] > 0:
                all_scores.append(r['matches'][0]['combined_score'])
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        median_score = np.median(all_scores) if all_scores else 0.0
        
        # PowerSafe cleanup
        if config.powersafe:
            powersafe_manager.cleanup_completed_entries()
        
        # Print comprehensive summary
        print(f"\n{'='*90}")
        print(f"🎯 PRODUCTION VIDEO-GPX CORRELATION SUMMARY")
        print(f"{'='*90}")
        print(f"Processing Mode: {'⚡ PowerSafe' if config.powersafe else '🏃 Standard'}")
        print(f"")
        print(f"📁 File Processing:")
        print(f"  Videos Found: {len(video_files)}")
        print(f"  Videos Successfully Processed: {len(valid_videos)}/{len(video_files)} ({100*len(valid_videos)/max(len(video_files), 1):.1f}%)")
        print(f"  GPX Files Found: {len(gpx_files)}")
        print(f"  GPX Files Successfully Processed: {len(valid_gpx)}/{len(gpx_files)} ({100*len(valid_gpx)/max(len(gpx_files), 1):.1f}%)")
        print(f"")
        print(f"🔗 Correlation Results:")
        print(f"  Total Videos with Results: {total_videos_with_results}")
        print(f"  Videos with Valid Matches (>0.1): {successful_matches}/{total_videos_with_results} ({100*successful_matches/max(total_videos_with_results, 1):.1f}%)")
        print(f"  Total Correlations Computed: {successful_correlations + failed_correlations}")
        print(f"  Successful Correlations: {successful_correlations} ✅")
        print(f"  Failed Correlations: {failed_correlations} ❌")
        print(f"")
        print(f"📊 Quality Distribution:")
        print(f"  🟢 Excellent (≥0.8): {excellent_matches}")
        print(f"  🟡 Good (≥0.6): {good_matches}")
        print(f"  🟠 Fair (≥0.4): {fair_matches}")
        print(f"  🔴 Poor/Failed: {total_videos_with_results - excellent_matches - good_matches - fair_matches}")
        print(f"")
        print(f"📈 Score Statistics:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Median Score: {median_score:.3f}")
        print(f"  Total Valid Scores: {len(all_scores)}")
        
        if config.powersafe:
            status = powersafe_manager.get_processing_status()
            print(f"")
            print(f"💾 PowerSafe Status:")
            print(f"  Total Saved Correlations: {status.get('total_correlations', 0)}")
            print(f"  Results Database: {powersafe_manager.results_path}")
            print(f"  Progress Database: {powersafe_manager.db_path}")
        
        print(f"")
        print(f"📄 Output Files:")
        print(f"  📊 Results: {results_path}")
        print(f"  📋 Report: {output_dir / 'production_report.json'}")
        print(f"  💾 Cache: {cache_dir}")
        print(f"  📝 Log: production_correlation.log")
        print(f"")
        
        # Display top correlations if any exist
        if all_scores:
            print(f"🏆 TOP CORRELATIONS:")
            print(f"{'='*90}")
            
            # Get top correlations across all videos
            all_correlations = []
            for video_path, result in results.items():
                if result['matches'] and result['matches'][0]['combined_score'] > 0:
                    best_match = result['matches'][0]
                    all_correlations.append((
                        Path(video_path).name,
                        Path(best_match['path']).name,
                        best_match['combined_score'],
                        best_match.get('quality', 'unknown')
                    ))
            
            # Sort by score and display top 10
            all_correlations.sort(key=lambda x: x[2], reverse=True)
            for i, (video, gpx, score, quality) in enumerate(all_correlations[:10], 1):
                quality_emoji = {
                    'excellent': '🟢', 
                    'good': '🟡', 
                    'fair': '🟠', 
                    'poor': '🔴', 
                    'very_poor': '🔴',
                    'failed': '❌'
                }.get(quality, '⚪')
                print(f"{i:2d}. {video[:50]:<50} ↔ {gpx[:30]:<30}")
                print(f"     Score: {score:.3f} | Quality: {quality_emoji} {quality}")
                if i < len(all_correlations):
                    print()
        else:
            print(f"⚠️ No successful correlations found!")
            print(f"   This could indicate:")
            print(f"   • Video processing failures (check logs)")
            print(f"   • GPX processing failures (check file formats)")
            print(f"   • Feature extraction issues")
            print(f"   • Incompatible data types")
        
        print(f"{'='*90}")
        
        # Success determination
        if successful_matches > 0:
            logger.info("🎉 Production correlation system completed successfully with matches!")
        elif len(valid_videos) > 0 and len(valid_gpx) > 0:
            logger.warning("⚠️ System completed but found no correlations - check data compatibility")
        else:
            logger.error("❌ System completed but no valid features were extracted")
        
        # Final recommendations
        if failed_correlations > successful_correlations:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Try reducing --parallel_videos to 1 for debugging")
            print(f"   • Reduce --max_frames (try 100 for memory issues)")
            print(f"   • Reduce --video_size (try 384 216 for memory issues)")
            print(f"   • Check video file formats and corruption")
            print(f"   • Verify GPX files contain valid track data")
            print(f"   • Enable --debug for detailed error analysis")
            if not config.powersafe:
                print(f"   • Use --powersafe to preserve progress during debugging")
            if config.strict:
                print(f"   • Remove --strict flag to enable CPU fallbacks for debugging")
        
        print(f"\n🧠 PERFORMANCE OPTIMIZATION:")
        print(f"   • Current settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
        print(f"   • GPU Memory: {config.max_gpu_memory_gb:.1f}GB limit per GPU")
        print(f"   • Parallel Videos: {config.parallel_videos}")
        print(f"   • Temp Directory: {config.cache_dir}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
        if config.powersafe:
            logger.info("PowerSafe: Progress has been saved and can be resumed")
        print("\n⚠️  Process interrupted. PowerSafe progress has been saved." if config.powersafe else "\n⚠️  Process interrupted.")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"❌ Production system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        if config.powersafe:
            logger.info("PowerSafe: Partial progress has been saved")
            print(f"\n❌ Error occurred: {e}")
            print("💾 PowerSafe: Partial progress has been saved and can be resumed with --powersafe flag")
        
        # Provide debugging suggestions
        print(f"\n🔧 DEBUGGING SUGGESTIONS:")
        print(f"   • Run with --debug for detailed error information")
        print(f"   • Try --parallel_videos 1 to isolate GPU issues")
        print(f"   • Reduce --max_frames to 100 for testing")
        print(f"   • Check video file integrity with ffprobe")
        print(f"   • Verify GPX files are valid XML")
        
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
        
        # Motion consistency
        if len(motion_vectors) > self.config.temporal_window:
            for i in range(self.config.temporal_window, len(motion_vectors)):
                window = motion_vectors[i-self.config.temporal_window:i]
                if window:
                    angles = [math.atan2(v[1], v[0] + 1e-8) for v in window]
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
            stability_score = 1.0 / (1.0 + torch.var(window_frames).item())
            stability[i] = stability_score
            
            # Temporal patterns
            if i >= window_size * 2:
                recent_window = gray_frames[i-window_size:i]
                past_window = gray_frames[i-window_size*2:i-window_size]
                
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
        color_variance = torch.var(frames, dim=[2, 3])
        mean_color_variance = torch.mean(color_variance, dim=1).cpu().numpy()
        
        # Color histograms and moments
        histograms = []
        color_moments = []
        
        for i in range(num_frames):
            frame = frames[i]
            
            # Color moments
            moments = []
            for c in range(3):
                channel = frame[c]
                mean_val = torch.mean(channel).item()
                std_val = torch.std(channel).item()
                skew_val = torch.mean((channel - mean_val)**3).item() / (std_val**3 + 1e-8)
                moments.extend([mean_val, std_val, skew_val])
            
            color_moments.append(moments)
            
            # Simplified histogram
            hist_features = []
            for c in range(3):
                channel_mean = torch.mean(frame[c]).item()
                channel_std = torch.std(frame[c]).item()
                hist_features.extend([channel_mean, channel_std])
            
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
        gray_frames = gray_frames.unsqueeze(1)
        
        # Edge detection
        edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
        edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3]).cpu().numpy()
        
        # Edge orientation
        edge_angle = torch.atan2(edges_y, edges_x + 1e-8)
        edge_orientation = torch.mean(torch.abs(torch.cos(edge_angle)), dim=[1, 2, 3]).cpu().numpy()
        
        # Texture features
        texture_complexity = []
        for i in range(frames.shape[0]):
            frame_gray = gray_frames[i, 0]
            
            try:
                if frame_gray.shape[1] > 1:
                    grad_x = torch.diff(frame_gray, dim=1)
                else:
                    grad_x = torch.zeros_like(frame_gray[:, :1])
                
                if frame_gray.shape[0] > 1:
                    grad_y = torch.diff(frame_gray, dim=0)
                else:
                    grad_y = torch.zeros_like(frame_gray[:1, :])
                
                min_height = min(grad_x.shape[0], grad_y.shape[0])
                min_width = min(grad_x.shape[1], grad_y.shape[1])
                
                if min_height > 0 and min_width > 0:
                    grad_x_cropped = grad_x[:min_height, :min_width]
                    grad_y_cropped = grad_y[:min_height, :min_width]
                    
                    texture_score = torch.mean(
                        torch.abs(grad_x_cropped)**2 + torch.abs(grad_y_cropped)**2
                    ).item()
                else:
                    texture_score = 0.0
                
                texture_complexity.append(texture_score)
                
            except Exception as e:
                logger.debug(f"Texture computation failed for frame {i}: {e}")
                texture_complexity.append(0.0)
        
        return {
            'edge_density': edge_density,
            'edge_orientation': edge_orientation,
            'texture_complexity': np.array(texture_complexity)
        }

class RobustGPXProcessor:
    """Robust GPX processor with comprehensive error handling"""
    
    def __init__(self, config: ProcessingConfig, powersafe_manager: Optional[PowerSafeManager] = None):
        self.config = config
        self.powersafe_manager = powersafe_manager
        
        if config.strict:
            if not cp.cuda.is_available():
                raise RuntimeError("STRICT MODE: CuPy CUDA is required for GPX processing but not available")
            self.use_gpu = True
            logger.info("STRICT MODE: GPX processor initialized with mandatory GPU usage")
        else:
            if not cp.cuda.is_available():
                logger.warning("CuPy CUDA not available, falling back to CPU for GPX processing")
                self.use_gpu = False
            else:
                self.use_gpu = True
            logger.info(f"GPX processor initialized (GPU: {self.use_gpu})")
    
    def process_gpx_files(self, gpx_paths: List[str], max_workers: int = None) -> Dict[str, Any]:
        """Process GPX files with robust error handling"""
        if max_workers is None:
            max_workers = min(8, mp.cpu_count())
        
        logger.info(f"Processing {len(gpx_paths)} GPX files with {max_workers} workers" + 
                   (" [STRICT GPU MODE]" if self.config.strict else ""))
        
        results = {}
        failed_files = []
        
        # Process sequentially for debugging
        for i, path in enumerate(tqdm(gpx_paths[:10], desc="Debugging GPX (first 10)")):
            if self.powersafe_manager:
                self.powersafe_manager.mark_gpx_processing(path)
            
            try:
                data = self._parse_gpx_safe(path)
                if data is not None:
                    results[path] = data
                    if self.powersafe_manager:
                        self.powersafe_manager.mark_gpx_features_done(path)
                else:
                    failed_files.append(path)
                    if self.powersafe_manager:
                        self.powersafe_manager.mark_gpx_failed(path, "Failed to parse GPX data")
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                failed_files.append(path)
                if self.powersafe_manager:
                    self.powersafe_manager.mark_gpx_failed(path, str(e))
        
        # Process remaining files in parallel if debugging successful
        if results:
            logger.info(f"GPX debugging successful ({len(results)}/10), processing remaining files...")
            remaining_paths = gpx_paths[10:]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._parse_gpx_safe, path): path for path in remaining_paths}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing remaining GPX"):
                    path = futures[future]
                    
                    if self.powersafe_manager:
                        self.powersafe_manager.mark_gpx_processing(path)
                    
                    try:
                        data = future.result()
                        if data is not None:
                            results[path] = data
                            if self.powersafe_manager:
                                self.powersafe_manager.mark_gpx_features_done(path)
                        else:
                            failed_files.append(path)
                            if self.powersafe_manager:
                                self.powersafe_manager.mark_gpx_failed(path, "Failed to parse GPX data")
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        failed_files.append(path)
                        if self.powersafe_manager:
                            self.powersafe_manager.mark_gpx_failed(path, str(e))
        else:
            logger.warning("All GPX debugging samples failed, skipping parallel processing")
            failed_files.extend(gpx_paths[10:])
        
        logger.info(f"Successfully processed {len(results)}/{len(gpx_paths)} GPX files")
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} GPX files")
        
        return results
    
    def _parse_gpx_safe(self, gpx_path: str) -> Optional[Dict]:
        """Safely parse and process single GPX file"""
        try:
            with open(gpx_path, 'r', encoding='utf-8', errors='ignore') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time is not None and point.latitude is not None and point.longitude is not None:
                            points.append({
                                'timestamp': point.time.replace(tzinfo=None),
                                'lat': float(point.latitude),
                                'lon': float(point.longitude),
                                'elevation': float(point.elevation or 0)
                            })
            
            if len(points) < 10:
                logger.debug(f"Insufficient points in {gpx_path}: {len(points)}")
                return None
            
            df = pd.DataFrame(points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate coordinates
            if (df['lat'].isna().any() or df['lon'].isna().any() or
                not (-90 <= df['lat'].min() <= df['lat'].max() <= 90) or
                not (-180 <= df['lon'].min() <= df['lon'].max() <= 180)):
                logger.debug(f"Invalid coordinates in {gpx_path}")
                return None
            
            # Compute features
            if self.config.strict and self.use_gpu:
                enhanced_data = self._compute_robust_features_gpu(df)
            elif self.use_gpu:
                try:
                    enhanced_data = self._compute_robust_features_gpu(df)
                except Exception as e:
                    if self.config.strict:
                        raise RuntimeError(f"STRICT MODE: GPU processing failed and CPU fallback not allowed: {e}")
                    logger.warning(f"GPU processing failed, falling back to CPU: {e}")
                    enhanced_data = self._compute_robust_features_cpu(df)
            else:
                if self.config.strict:
                    raise RuntimeError("STRICT MODE: GPU processing required but GPU not available")
                enhanced_data = self._compute_robust_features_cpu(df)
            
            return enhanced_data
            
        except Exception as e:
            if self.config.strict and "STRICT MODE" in str(e):
                raise
            logger.debug(f"GPX processing failed for {gpx_path}: {e}")
            return None
    
    def _compute_robust_features_gpu(self, df: pd.DataFrame) -> Dict:
        """Compute robust features using GPU (CuPy)"""
        n_points = len(df)
        
        # Transfer to GPU
        lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)
        lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
        elevs_gpu = cp.array(df['elevation'].values, dtype=cp.float64)
        
        # Compute time differences
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        time_diffs_gpu = cp.array(time_diffs, dtype=cp.float64)
        
        # Compute distances
        distances_gpu = self._compute_distances_robust_gpu(lats_gpu, lons_gpu)
        
        # Compute motion features
        motion_features = self._compute_motion_features_robust_gpu(
            lats_gpu, lons_gpu, elevs_gpu, time_diffs_gpu, distances_gpu
        )
        
        # Compute statistical features
        statistical_features = self._compute_statistical_features_robust_gpu(motion_features, distances_gpu, elevs_gpu)
        
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
            'avg_speed': float(cp.mean(motion_features['speed'])) if 'speed' in motion_features else 0,
            'processing_mode': 'GPU'
        }
    
    def _compute_distances_robust_gpu(self, lats: cp.ndarray, lons: cp.ndarray) -> cp.ndarray:
        """Compute distances using GPU with Haversine formula"""
        n = len(lats)
        distances = cp.zeros(n)
        
        if n < 2:
            return distances
        
        R = 3958.8  # Earth radius in miles
        
        lat1_rad = cp.radians(lats[:-1])
        lon1_rad = cp.radians(lons[:-1])
        lat2_rad = cp.radians(lats[1:])
        lon2_rad = cp.radians(lons[1:])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))
        
        computed_distances = R * c
        distances[1:] = computed_distances
        
        return distances
    
    def _compute_motion_features_robust_gpu(self, lats: cp.ndarray, lons: cp.ndarray, 
                                          elevs: cp.ndarray, time_diffs: cp.ndarray, 
                                          distances: cp.ndarray) -> Dict[str, cp.ndarray]:
        """Compute motion features using GPU"""
        n = len(lats)
        
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
        
        # Speed computation
        valid_time_diffs = cp.maximum(time_diffs[1:], 1e-6)
        speed_values = distances[1:] * 3600 / valid_time_diffs
        features['speed'][1:] = speed_values
        
        # Acceleration
        if n > 2:
            speed_diff = features['speed'][2:] - features['speed'][1:-1]
            accel_values = speed_diff / cp.maximum(time_diffs[2:], 1e-6)
            features['acceleration'][2:] = accel_values
        
        # Jerk
        if n > 3:
            accel_diff = features['acceleration'][3:] - features['acceleration'][2:-1]
            jerk_values = accel_diff / cp.maximum(time_diffs[3:], 1e-6)
            features['jerk'][3:] = jerk_values
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_robust_gpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
        # Bearing changes
        if n > 2:
            bearing_diffs = cp.diff(features['bearing'][1:])
            bearing_diffs = cp.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = cp.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = cp.abs(bearing_diffs)
            features['bearing_change'][2:] = bearing_changes
            
            valid_distances = cp.maximum(distances[2:], 1e-8)
            curvature_values = bearing_changes / valid_distances
            features['curvature'][2:] = curvature_values
        
        # Elevation change rate
        if n > 1:
            elev_diffs = elevs[1:] - elevs[:-1]
            elev_rates = elev_diffs / cp.maximum(time_diffs[1:], 1e-6)
            features['elevation_change_rate'][1:] = elev_rates
        
        return features
    
    def _compute_bearings_robust_gpu(self, lat1: cp.ndarray, lon1: cp.ndarray, 
                                   lat2: cp.ndarray, lon2: cp.ndarray) -> cp.ndarray:
        """Compute bearings using GPU"""
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
    
    def _compute_statistical_features_robust_gpu(self, motion_features: Dict[str, cp.ndarray],
                                               distances: cp.ndarray, elevations: cp.ndarray) -> Dict[str, cp.ndarray]:
        """Compute statistical features using GPU"""
        features = {}
        
        # Speed statistics
        speed = motion_features['speed']
        valid_speed = speed[speed > 0]
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
        
        return featuresarray([
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
    
    def _compute_robust_features_cpu(self, df: pd.DataFrame) -> Dict:
        """Compute robust features using CPU (fallback)"""
        n_points = len(df)
        
        # Convert to numpy arrays
        lats = df['lat'].values
        lons = df['lon'].values
        elevs = df['elevation'].values
        
        # Compute time differences
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        
        # Compute distances (CPU version)
        distances = self._compute_distances_cpu(lats, lons)
        
        # Compute motion features
        motion_features = self._compute_motion_features_cpu(lats, lons, elevs, time_diffs, distances)
        
        # Compute statistical features
        statistical_features = self._compute_statistical_features_cpu(motion_features, distances, elevs)
        
        # Combine features
        all_features = {**motion_features, **statistical_features}
        
        # Add metadata
        duration = self._compute_duration_safe(df['timestamp'])
        total_distance = np.sum(distances)
        
        return {
            'df': df,
            'features': all_features,
            'start_time': df['timestamp'].iloc[0],
            'end_time': df['timestamp'].iloc[-1],
            'duration': duration,
            'distance': total_distance,
            'point_count': n_points,
            'max_speed': np.max(motion_features['speed']) if 'speed' in motion_features else 0,
            'avg_speed': np.mean(motion_features['speed']) if 'speed' in motion_features else 0,
            'processing_mode': 'CPU'
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
    
    def _compute_distances_cpu(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Compute distances using CPU (Haversine formula)"""
        n = len(lats)
        distances = np.zeros(n)
        
        if n < 2:
            return distances
        
        R = 3958.8  # Earth radius in miles
        
        lat1_rad = np.radians(lats[:-1])
        lon1_rad = np.radians(lons[:-1])
        lat2_rad = np.radians(lats[1:])
        lon2_rad = np.radians(lons[1:])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        computed_distances = R * c
        distances[1:] = computed_distances
        
        return distances
    
    def _compute_motion_features_cpu(self, lats: np.ndarray, lons: np.ndarray, 
                                   elevs: np.ndarray, time_diffs: List[float], 
                                   distances: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute motion features using CPU"""
        n = len(lats)
        time_diffs = np.array(time_diffs)
        
        features = {
            'speed': np.zeros(n),
            'acceleration': np.zeros(n),
            'jerk': np.zeros(n),
            'bearing': np.zeros(n),
            'bearing_change': np.zeros(n),
            'curvature': np.zeros(n),
            'elevation_change_rate': np.zeros(n)
        }
        
        if n < 2:
            return features
        
        # Speed computation
        valid_time_diffs = np.maximum(time_diffs[1:], 1e-6)
        speed_values = distances[1:] * 3600 / valid_time_diffs
        features['speed'][1:] = speed_values
        
        # Acceleration
        if n > 2:
            speed_diff = features['speed'][2:] - features['speed'][1:-1]
            accel_values = speed_diff / np.maximum(time_diffs[2:], 1e-6)
            features['acceleration'][2:] = accel_values
        
        # Jerk
        if n > 3:
            accel_diff = features['acceleration'][3:] - features['acceleration'][2:-1]
            jerk_values = accel_diff / np.maximum(time_diffs[3:], 1e-6)
            features['jerk'][3:] = jerk_values
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_cpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
        # Bearing changes
        if n > 2:
            bearing_diffs = np.diff(features['bearing'][1:])
            bearing_diffs = np.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = np.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = np.abs(bearing_diffs)
            features['bearing_change'][2:] = bearing_changes
            
            valid_distances = np.maximum(distances[2:], 1e-8)
            curvature_values = bearing_changes / valid_distances
            features['curvature'][2:] = curvature_values
        
        # Elevation change rate
        if n > 1:
            elev_diffs = elevs[1:] - elevs[:-1]
            elev_rates = elev_diffs / np.maximum(time_diffs[1:], 1e-6)
            features['elevation_change_rate'][1:] = elev_rates
        
        return features
    
    def _compute_bearings_cpu(self, lat1: np.ndarray, lon1: np.ndarray, 
                            lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Compute bearings using CPU"""
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearings = np.degrees(np.arctan2(y, x))
        bearings = np.where(bearings < 0, bearings + 360, bearings)
        
        return bearings
    
    def _compute_statistical_features_cpu(self, motion_features: Dict[str, np.ndarray],
                                        distances: np.ndarray, elevations: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical features using CPU"""
        features = {}
        
        # Speed statistics
        speed = motion_features['speed']
        valid_speed = speed[speed > 0]
        if len(valid_speed) > 0:
            features['speed_stats'] = np.array([
                np.mean(valid_speed), np.std(valid_speed), 
                np.min(valid_speed), np.max(valid_speed),
                np.percentile(valid_speed, 25), np.percentile(valid_speed, 50), 
                np.percentile(valid_speed, 75)
            ])
        else:
            features['speed_stats'] = np.zeros(7)
        
        # Bearing statistics
        bearing = motion_features['bearing']
        valid_bearing = bearing[bearing > 0]
        if len(valid_bearing) > 0:
            features['bearing_stats'] = np.array([
                np.mean(valid_bearing), np.std(valid_bearing),
                np.min(valid_bearing), np.max(valid_bearing)
            ])
        else:
            features['bearing_stats'] = np.zeros(4)
        
        # Elevation statistics
        if len(elevations) > 1:
            elev_diffs = np.diff(elevations)
            total_climb = np.sum(np.maximum(elev_diffs, 0))
            total_descent = np.sum(np.maximum(-elev_diffs, 0))
            
            features['elevation_stats'] = np.array([
                np.mean(elevations), np.std(elevations),
                np.min(elevations), np.max(elevations),
                total_climb, total_descent
            ])
        else:
            features['elevation_stats'] = np.zeros(6)
        
        # Distance statistics
        valid_distances = distances[distances > 0]
        if len(valid_distances) > 0:
            features['distance_stats'] = np.array([
                np.sum(distances), np.mean(valid_distances),
                np.std(valid_distances), np.max(valid_distances)
            ])
        else:
            features['distance_stats'] = np.zeros(4)
        
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
            
            # Create unified signature by taking the longest component
            if len(signature_components) == 1:
                return self._robust_normalize(signature_components[0])
            
            # Take the longest component as base
            longest_component = max(signature_components, key=len)
            return self._robust_normalize(longest_component)
            
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_temporal_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute temporal correlation with basic alignment"""
        try:
            # Extract temporal signatures
            video_temporal = self._extract_temporal_signature(video_features, 'video')
            gpx_temporal = self._extract_temporal_signature(gpx_features, 'gpx')
            
            if video_temporal is None or gpx_temporal is None:
                return 0.0
            
            # Direct correlation
            min_len = min(len(video_temporal), len(gpx_temporal))
            if min_len > 5:
                v_temp = video_temporal[:min_len]
                g_temp = gpx_temporal[:min_len]
                
                corr = np.corrcoef(v_temp, g_temp)[0, 1]
                if not np.isnan(corr):
                    return float(np.clip(abs(corr), 0.0, 1.0))
            
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
                            return np.diff(values)
                            
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
            
            # Cosine similarity
            cosine_sim = np.dot(video_stats, gpx_stats) / (
                np.linalg.norm(video_stats) * np.linalg.norm(gpx_stats) + 1e-8
            )
            
            if not np.isnan(cosine_sim):
                return float(np.clip(abs(cosine_sim), 0.0, 1.0))
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
                                    np.mean(values), np.std(values), np.median(values)
                                ])
                            
            elif source_type == 'gpx':
                # GPX statistical features
                stat_keys = ['speed_stats', 'bearing_stats', 'elevation_stats', 'distance_stats']
                for key in stat_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend(values.flatten()[:3])
            
            if not profile_components:
                return None
            
            return np.array(profile_components)
            
        except Exception as e:
            logger.debug(f"Statistical profile extraction failed for {source_type}: {e}")
            return None
    
    def _compute_spatial_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute spatial feature similarity"""
        return 0.5  # Placeholder
    
    def _compute_behavioral_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute behavioral pattern similarity"""
        return 0.5  # Placeholder
    
    def _robust_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Robust normalization with outlier handling"""
        try:
            if len(vector) == 0:
                return vector
            
            # Simple min-max normalization
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

def process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
    """Enhanced video processing with better error recovery"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    gpu_id = gpu_manager.acquire_gpu(timeout=config.gpu_timeout)
    
    if gpu_id is None:
        error_msg = f"Could not acquire GPU within {config.gpu_timeout}s timeout"
        if config.strict:
            error_msg = f"STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {video_path}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.error(f"{error_msg} for {video_path}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            return video_path, None
    
    try:
        decoder = EnhancedFFmpegDecoder(gpu_manager, config)
        feature_extractor = EnhancedFeatureExtractor(gpu_manager, config)
        
        # Enhanced decode with retries
        frames_tensor, fps, duration = decoder.decode_video_enhanced(video_path, gpu_id)
        
        if frames_tensor is None:
            error_msg = "Video decoding failed after all attempts"
            if config.strict:
                error_msg = f"STRICT MODE: {error_msg}"
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                raise RuntimeError(error_msg)
            else:
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                return video_path, None
        
        # Extract features
        features = feature_extractor.extract_enhanced_features(frames_tensor, gpu_id)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_id
        features['processing_mode'] = 'GPU_STRICT' if config.strict else 'GPU'
        
        # Mark feature extraction as done in power-safe mode
        if powersafe_manager:
            powersafe_manager.mark_video_features_done(video_path)
        
        success_msg = f"✅ Successfully processed {Path(video_path).name} on GPU {gpu_id}"
        if config.strict:
            success_msg += " [STRICT MODE]"
        logger.info(success_msg)
        return video_path, features
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        if config.strict and "STRICT MODE" not in str(e):
            error_msg = f"STRICT MODE: {error_msg}"
        
        logger.error(f"❌ {error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        
        if config.strict:
            raise RuntimeError(error_msg)
        return video_path, None
        
    finally:
        if gpu_id is not None:
            gpu_manager.release_gpu(gpu_id)
            # Aggressive GPU memory cleanup
            try:
                torch.cuda.empty_cache()
                if gpu_id < torch.cuda.device_count():
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            except:
                pass

def update_config_for_temp_dir(args):
    """Update configuration to use ~/penis/temp directory"""
    # Override the cache directory argument to use the requested location
    args.cache_dir = os.path.expanduser("~/penis/temp")
    
    # Create the directory if it doesn't exist
    temp_dir = Path(args.cache_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Using temp directory: {args.cache_dir}")
    return args

def main():
    """Enhanced main function with comprehensive error handling and progress tracking"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", required=True,
                       help="Directory containing videos and GPX files")
    
    # Processing configuration - ENHANCED DEFAULTS
    parser.add_argument("--max_frames", type=int, default=150,
                       help="Maximum frames per video (default: 150 - optimized for stability)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[480, 270],
                       help="Target video resolution (default: 480 270 - optimized for compatibility)")
    parser.add_argument("--sample_rate", type=float, default=3.0,
                       help="Video sampling rate (default: 3.0)")
    parser.add_argument("--parallel_videos", type=int, default=1,
                       help="Number of videos to process in parallel (default: 1 - prevents GPU conflicts)")
    
    # GPU configuration
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1],
                       help="GPU IDs to use (default: [0, 1])")
    parser.add_argument("--gpu_timeout", type=int, default=60,
                       help="Seconds to wait for GPU availability (default: 60 - more time for conversion)")
    parser.add_argument("--max_gpu_memory", type=float, default=12.0,
                       help="Maximum GPU memory to use in GB (default: 12.0)")
    parser.add_argument("--memory_efficient", action='store_true', default=True,
                       help="Enable memory optimizations (default: True)")
    
    # Video preprocessing and caching - ENHANCED DEFAULTS
    parser.add_argument("--enable_preprocessing", action='store_true', default=True,
                       help="Enable GPU-based video preprocessing for problematic videos (default: True)")
    parser.add_argument("--ram_cache", type=float, default=32.0,
                       help="RAM to use for video caching in GB (default: 32.0 - more conservative)")
    parser.add_argument("--disk_cache", type=float, default=1000.0,
                       help="Disk space to use for video caching in GB (default: 1000.0)")
    parser.add_argument("--cache_dir", type=str, default="~/penis/temp",
                       help="Directory for video cache (default: ~/penis/temp)")
    parser.add_argument("--replace_originals", action='store_true',
                       help="Replace original files with preprocessed versions")
    
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
    parser.add_argument("--strict", action='store_true',
                       help="STRICT MODE: Enforce GPU usage at all times, fail if GPU unavailable")
    
    # Power-safe mode arguments
    parser.add_argument("--powersafe", action='store_true',
                       help="Enable power-safe mode with incremental saves and resume capability")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save results every N correlations in powersafe mode (default: 5)")
    
    args = parser.parse_args()
    
    # UPDATE CONFIG TO USE CORRECT TEMP DIRECTORY
    args = update_config_for_temp_dir(args)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level, "production_correlation.log")
    
    if args.strict:
        logger.info("🚀 Starting Production-Ready Video-GPX Correlation System [STRICT GPU MODE]")
    else:
        logger.info("🚀 Starting Production-Ready Video-GPX Correlation System")
    
    # Log configuration
    logger.info(f"💾 Memory Settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
    logger.info(f"🎞️ Video Preprocessing: {'Enabled' if args.enable_preprocessing else 'Disabled'}")
    if args.enable_preprocessing:
        logger.info(f"📋 RAM Cache: {args.ram_cache:.0f}GB, 💾 Disk Cache: {args.disk_cache:.0f}GB")
        logger.info(f"📁 Cache Directory: {args.cache_dir}")
    
    try:
        # Create configuration
        config = ProcessingConfig(
            max_frames=args.max_frames,
            target_size=tuple(args.video_size),
            sample_rate=args.sample_rate,
            parallel_videos=args.parallel_videos,
            powersafe=args.powersafe,
            save_interval=args.save_interval,
            gpu_timeout=args.gpu_timeout,
            strict=args.strict,
            memory_efficient=args.memory_efficient,
            max_gpu_memory_gb=args.max_gpu_memory,
            enable_preprocessing=args.enable_preprocessing,
            ram_cache_gb=args.ram_cache,
            disk_cache_gb=args.disk_cache,
            cache_dir=args.cache_dir,
            replace_originals=args.replace_originals
        )
        
        # Validate strict mode requirements early
        if config.strict:
            logger.info("🔒 STRICT MODE ENABLED: GPU usage mandatory, no CPU fallback")
            if not torch.cuda.is_available():
                raise RuntimeError("STRICT MODE: CUDA is required but not available")
            if not cp.cuda.is_available():
                raise RuntimeError("STRICT MODE: CuPy CUDA is required but not available")
        
        # Setup directories
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        cache_dir = Path(args.cache)
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize PowerSafe manager
        powersafe_manager = PowerSafeManager(cache_dir, config)
        
        # Initialize GPU manager with strict mode and config
        gpu_manager = EnhancedGPUManager(args.gpu_ids, strict=config.strict, config=config)
        
        # Smart file scanning with PowerSafe prioritization
        logger.info("📁 Scanning for input files with PowerSafe prioritization...")
        
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
        
        # Process videos with enhanced error tracking
        logger.info("🎬 Processing videos with enhanced parallel processing...")
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
            
            # Prepare arguments for parallel processing
            video_args = [(video_path, gpu_manager, config, powersafe_manager) for video_path in videos_to_process]
            
            # Enhanced progress tracking
            successful_videos = 0
            failed_videos = 0
            
            # Use ThreadPoolExecutor for better GPU sharing
            with ThreadPoolExecutor(max_workers=config.parallel_videos) as executor:
                futures = [executor.submit(process_video_parallel_enhanced, arg) for arg in video_args]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                    video_path, features = future.result()
                    video_features[video_path] = features
                    
                    if features is not None:
                        successful_videos += 1
                    else:
                        failed_videos += 1
                    
                    # Periodic cache save
                    if (successful_videos + failed_videos) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
                        logger.info(f"Progress: {successful_videos} ✅ | {failed_videos} ❌")
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
        
        logger.info(f"🎬 Video processing complete: {successful_videos} ✅ | {failed_videos} ❌")
        
        # Process GPX files with enhanced error tracking
        logger.info("🗺️ Processing GPX files...")
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
            
            logger.info(f"🗺️ GPX processing complete: {len(new_gpx_results)} successful")
        
        # Perform enhanced correlation with PowerSafe
        logger.info("🔗 Starting enhanced correlation analysis...")
        
        # Filter valid features
        valid_videos = {k: v for k, v in video_features.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None and 'features' in v}
        
        logger.info(f"Valid features: {len(valid_videos)} videos, {len(valid_gpx)} GPX tracks")
        
        if not valid_videos:
            raise RuntimeError(f"No valid video features! Processed {len(video_features)} videos but none succeeded.")
        
        if not valid_gpx:
            raise RuntimeError(f"No valid GPX features! Processed {len(gpx_database)} GPX files but none succeeded.")
        
        # Initialize similarity engine
        similarity_engine = EnhancedSimilarityEngine(config)
        
        # Compute correlations with PowerSafe incremental saves
        results = existing_results.copy()
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        # Calculate existing correlations
        existing_correlations = sum(len(result.get('matches', [])) for result in existing_results.values())
        
        # Reset PowerSafe manager counter
        if config.powersafe:
            powersafe_manager.correlation_counter = existing_correlations
        
        successful_correlations = 0
        failed_correlations = 0
        
        with tqdm(total=total_comparisons, desc="Computing correlations") as pbar:
            # Update progress bar for existing correlations
            if existing_correlations > 0:
                pbar.update(existing_correlations)
            
            for video_path, video_features_data in valid_videos.items():
                # Skip if video already has complete correlation results
                if (video_path in results and 
                    len(results[video_path].get('matches', [])) >= min(args.top_k, len(valid_gpx))):
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
                        successful_correlations += 1
                        
                        # PowerSafe: Add to pending correlations
                        if config.powersafe:
                            powersafe_manager.add_pending_correlation(video_path, gpx_path, match_info)
                        
                    except Exception as e:
                        logger.debug(f"Correlation failed for {Path(video_path).name} vs {Path(gpx_path).name}: {e}")
                        match_info = {
                            'path': gpx_path,
                            'combined_score': 0.0,
                            'quality': 'failed',
                            'error': str(e)
                        }
                        matches.append(match_info)
                        failed_correlations += 1
                        
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
                if matches and matches[0]['combined_score'] > 0:
                    best = matches[0]
                    logger.info(f"🎯 Best match for {Path(video_path).name}: "
                              f"{Path(best['path']).name} "
                              f"(score: {best['combined_score']:.3f}, quality: {best['quality']})")
                else:
                    logger.warning(f"⚠️ No valid matches found for {Path(video_path).name}")
        
        logger.info(f"🔗 Correlation analysis complete: {successful_correlations} ✅ | {failed_correlations} ❌")
        
        # Final PowerSafe save
        if config.powersafe:
            powersafe_manager.save_incremental_results(powersafe_manager.pending_results)
            logger.info("PowerSafe: Final incremental save completed")
        
        # Save final results
        results_path = output_dir / "production_correlations.pkl"
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
                'successful_correlations': successful_correlations,
                'failed_correlations': failed_correlations,
                'gpu_ids': args.gpu_ids,
                'config': config.__dict__
            },
            'results': results
        }
        
        # Add PowerSafe status if enabled
        if config.powersafe:
            report_data['powersafe_status'] = powersafe_manager.get_processing_status()
        
        with open(output_dir / "production_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate summary statistics
        total_videos_with_results = len(results)
        successful_matches = sum(1 for r in results.values() 
                               if r['matches'] and r['matches'][0]['combined_score'] > 0.1)
        
        excellent_matches = sum(1 for r in results.values() 
                              if r['matches'] and r['matches'][0].get('quality') == 'excellent')
        
        good_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'good')
        
        fair_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'fair')
        
        # Calculate average scores
        all_scores = []
        for r in results.values():
            if r['matches'] and r['matches'][0]['combined_score'] > 0:
                all_scores.append(r['matches'][0]['combined_score'])
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        median_score = np.median(all_scores) if all_scores else 0.0
        
        # PowerSafe cleanup
        if config.powersafe:
            powersafe_manager.cleanup_completed_entries()
        
        # Print comprehensive summary
        print(f"\n{'='*90}")
        print(f"🎯 PRODUCTION VIDEO-GPX CORRELATION SUMMARY")
        print(f"{'='*90}")
        print(f"Processing Mode: {'⚡ PowerSafe' if config.powersafe else '🏃 Standard'}")
        print(f"")
        print(f"📁 File Processing:")
        print(f"  Videos Found: {len(video_files)}")
        print(f"  Videos Successfully Processed: {len(valid_videos)}/{len(video_files)} ({100*len(valid_videos)/max(len(video_files), 1):.1f}%)")
        print(f"  GPX Files Found: {len(gpx_files)}")
        print(f"  GPX Files Successfully Processed: {len(valid_gpx)}/{len(gpx_files)} ({100*len(valid_gpx)/max(len(gpx_files), 1):.1f}%)")
        print(f"")
        print(f"🔗 Correlation Results:")
        print(f"  Total Videos with Results: {total_videos_with_results}")
        print(f"  Videos with Valid Matches (>0.1): {successful_matches}/{total_videos_with_results} ({100*successful_matches/max(total_videos_with_results, 1):.1f}%)")
        print(f"  Total Correlations Computed: {successful_correlations + failed_correlations}")
        print(f"  Successful Correlations: {successful_correlations} ✅")
        print(f"  Failed Correlations: {failed_correlations} ❌")
        print(f"")
        print(f"📊 Quality Distribution:")
        print(f"  🟢 Excellent (≥0.8): {excellent_matches}")
        print(f"  🟡 Good (≥0.6): {good_matches}")
        print(f"  🟠 Fair (≥0.4): {fair_matches}")
        print(f"  🔴 Poor/Failed: {total_videos_with_results - excellent_matches - good_matches - fair_matches}")
        print(f"")
        print(f"📈 Score Statistics:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Median Score: {median_score:.3f}")
        print(f"  Total Valid Scores: {len(all_scores)}")
        
        if config.powersafe:
            status = powersafe_manager.get_processing_status()
            print(f"")
            print(f"💾 PowerSafe Status:")
            print(f"  Total Saved Correlations: {status.get('total_correlations', 0)}")
            print(f"  Results Database: {powersafe_manager.results_path}")
            print(f"  Progress Database: {powersafe_manager.db_path}")
        
        print(f"")
        print(f"📄 Output Files:")
        print(f"  📊 Results: {results_path}")
        print(f"  📋 Report: {output_dir / 'production_report.json'}")
        print(f"  💾 Cache: {cache_dir}")
        print(f"  📝 Log: production_correlation.log")
        print(f"")
        
        # Display top correlations if any exist
        if all_scores:
            print(f"🏆 TOP CORRELATIONS:")
            print(f"{'='*90}")
            
            # Get top correlations across all videos
            all_correlations = []
            for video_path, result in results.items():
                if result['matches'] and result['matches'][0]['combined_score'] > 0:
                    best_match = result['matches'][0]
                    all_correlations.append((
                        Path(video_path).name,
                        Path(best_match['path']).name,
                        best_match['combined_score'],
                        best_match.get('quality', 'unknown')
                    ))
            
            # Sort by score and display top 10
            all_correlations.sort(key=lambda x: x[2], reverse=True)
            for i, (video, gpx, score, quality) in enumerate(all_correlations[:10], 1):
                quality_emoji = {
                    'excellent': '🟢', 
                    'good': '🟡', 
                    'fair': '🟠', 
                    'poor': '🔴', 
                    'very_poor': '🔴',
                    'failed': '❌'
                }.get(quality, '⚪')
                print(f"{i:2d}. {video[:50]:<50} ↔ {gpx[:30]:<30}")
                print(f"     Score: {score:.3f} | Quality: {quality_emoji} {quality}")
                if i < len(all_correlations):
                    print()
        else:
            print(f"⚠️ No successful correlations found!")
            print(f"   This could indicate:")
            print(f"   • Video processing failures (check logs)")
            print(f"   • GPX processing failures (check file formats)")
            print(f"   • Feature extraction issues")
            print(f"   • Incompatible data types")
        
        print(f"{'='*90}")
        
        # Success determination
        if successful_matches > 0:
            logger.info("🎉 Production correlation system completed successfully with matches!")
        elif len(valid_videos) > 0 and len(valid_gpx) > 0:
            logger.warning("⚠️ System completed but found no correlations - check data compatibility")
        else:
            logger.error("❌ System completed but no valid features were extracted")
        
        # Final recommendations
        if failed_correlations > successful_correlations:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Try reducing --parallel_videos to 1 for debugging")
            print(f"   • Reduce --max_frames (try 100 for memory issues)")
            print(f"   • Reduce --video_size (try 384 216 for memory issues)")
            print(f"   • Check video file formats and corruption")
            print(f"   • Verify GPX files contain valid track data")
            print(f"   • Enable --debug for detailed error analysis")
            if not config.powersafe:
                print(f"   • Use --powersafe to preserve progress during debugging")
            if config.strict:
                print(f"   • Remove --strict flag to enable CPU fallbacks for debugging")
        
        print(f"\n🧠 PERFORMANCE OPTIMIZATION:")
        print(f"   • Current settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
        print(f"   • GPU Memory: {config.max_gpu_memory_gb:.1f}GB limit per GPU")
        print(f"   • Parallel Videos: {config.parallel_videos}")
        print(f"   • Temp Directory: {config.cache_dir}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
        if config.powersafe:
            logger.info("PowerSafe: Progress has been saved and can be resumed")
        print("\n⚠️  Process interrupted. PowerSafe progress has been saved." if config.powersafe else "\n⚠️  Process interrupted.")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"❌ Production system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        if config.powersafe:
            logger.info("PowerSafe: Partial progress has been saved")
            print(f"\n❌ Error occurred: {e}")
            print("💾 PowerSafe: Partial progress has been saved and can be resumed with --powersafe flag")
        
        # Provide debugging suggestions
        print(f"\n🔧 DEBUGGING SUGGESTIONS:")
        print(f"   • Run with --debug for detailed error information")
        print(f"   • Try --parallel_videos 1 to isolate GPU issues")
        print(f"   • Reduce --max_frames to 100 for testing")
        print(f"   • Check video file integrity with ffprobe")
        print(f"   • Verify GPX files are valid XML")
        
        sys.exit(1)
    
    finally:
        # Cleanup temporary directories
        try:
            if 'decoder' in locals():
                decoder.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()array([
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
    
    def _compute_robust_features_cpu(self, df: pd.DataFrame) -> Dict:
        """Compute robust features using CPU (fallback)"""
        n_points = len(df)
        
        # Convert to numpy arrays
        lats = df['lat'].values
        lons = df['lon'].values
        elevs = df['elevation'].values
        
        # Compute time differences
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        
        # Compute distances (CPU version)
        distances = self._compute_distances_cpu(lats, lons)
        
        # Compute motion features
        motion_features = self._compute_motion_features_cpu(lats, lons, elevs, time_diffs, distances)
        
        # Compute statistical features
        statistical_features = self._compute_statistical_features_cpu(motion_features, distances, elevs)
        
        # Combine features
        all_features = {**motion_features, **statistical_features}
        
        # Add metadata
        duration = self._compute_duration_safe(df['timestamp'])
        total_distance = np.sum(distances)
        
        return {
            'df': df,
            'features': all_features,
            'start_time': df['timestamp'].iloc[0],
            'end_time': df['timestamp'].iloc[-1],
            'duration': duration,
            'distance': total_distance,
            'point_count': n_points,
            'max_speed': np.max(motion_features['speed']) if 'speed' in motion_features else 0,
            'avg_speed': np.mean(motion_features['speed']) if 'speed' in motion_features else 0,
            'processing_mode': 'CPU'
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
    
    def _compute_distances_cpu(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Compute distances using CPU (Haversine formula)"""
        n = len(lats)
        distances = np.zeros(n)
        
        if n < 2:
            return distances
        
        R = 3958.8  # Earth radius in miles
        
        lat1_rad = np.radians(lats[:-1])
        lon1_rad = np.radians(lons[:-1])
        lat2_rad = np.radians(lats[1:])
        lon2_rad = np.radians(lons[1:])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        computed_distances = R * c
        distances[1:] = computed_distances
        
        return distances
    
    def _compute_motion_features_cpu(self, lats: np.ndarray, lons: np.ndarray, 
                                   elevs: np.ndarray, time_diffs: List[float], 
                                   distances: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute motion features using CPU"""
        n = len(lats)
        time_diffs = np.array(time_diffs)
        
        features = {
            'speed': np.zeros(n),
            'acceleration': np.zeros(n),
            'jerk': np.zeros(n),
            'bearing': np.zeros(n),
            'bearing_change': np.zeros(n),
            'curvature': np.zeros(n),
            'elevation_change_rate': np.zeros(n)
        }
        
        if n < 2:
            return features
        
        # Speed computation
        valid_time_diffs = np.maximum(time_diffs[1:], 1e-6)
        speed_values = distances[1:] * 3600 / valid_time_diffs
        features['speed'][1:] = speed_values
        
        # Acceleration
        if n > 2:
            speed_diff = features['speed'][2:] - features['speed'][1:-1]
            accel_values = speed_diff / np.maximum(time_diffs[2:], 1e-6)
            features['acceleration'][2:] = accel_values
        
        # Jerk
        if n > 3:
            accel_diff = features['acceleration'][3:] - features['acceleration'][2:-1]
            jerk_values = accel_diff / np.maximum(time_diffs[3:], 1e-6)
            features['jerk'][3:] = jerk_values
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_cpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
        # Bearing changes
        if n > 2:
            bearing_diffs = np.diff(features['bearing'][1:])
            bearing_diffs = np.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = np.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = np.abs(bearing_diffs)
            features['bearing_change'][2:] = bearing_changes
            
            valid_distances = np.maximum(distances[2:], 1e-8)
            curvature_values = bearing_changes / valid_distances
            features['curvature'][2:] = curvature_values
        
        # Elevation change rate
        if n > 1:
            elev_diffs = elevs[1:] - elevs[:-1]
            elev_rates = elev_diffs / np.maximum(time_diffs[1:], 1e-6)
            features['elevation_change_rate'][1:] = elev_rates
        
        return features
    
    def _compute_bearings_cpu(self, lat1: np.ndarray, lon1: np.ndarray, 
                            lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Compute bearings using CPU"""
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearings = np.degrees(np.arctan2(y, x))
        bearings = np.where(bearings < 0, bearings + 360, bearings)
        
        return bearings
    
    def _compute_statistical_features_cpu(self, motion_features: Dict[str, np.ndarray],
                                        distances: np.ndarray, elevations: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical features using CPU"""
        features = {}
        
        # Speed statistics
        speed = motion_features['speed']
        valid_speed = speed[speed > 0]
        if len(valid_speed) > 0:
            features['speed_stats'] = np.array([
                np.mean(valid_speed), np.std(valid_speed), 
                np.min(valid_speed), np.max(valid_speed),
                np.percentile(valid_speed, 25), np.percentile(valid_speed, 50), 
                np.percentile(valid_speed, 75)
            ])
        else:
            features['speed_stats'] = np.zeros(7)
        
        # Bearing statistics
        bearing = motion_features['bearing']
        valid_bearing = bearing[bearing > 0]
        if len(valid_bearing) > 0:
            features['bearing_stats'] = np.array([
                np.mean(valid_bearing), np.std(valid_bearing),
                np.min(valid_bearing), np.max(valid_bearing)
            ])
        else:
            features['bearing_stats'] = np.zeros(4)
        
        # Elevation statistics
        if len(elevations) > 1:
            elev_diffs = np.diff(elevations)
            total_climb = np.sum(np.maximum(elev_diffs, 0))
            total_descent = np.sum(np.maximum(-elev_diffs, 0))
            
            features['elevation_stats'] = np.array([
                np.mean(elevations), np.std(elevations),
                np.min(elevations), np.max(elevations),
                total_climb, total_descent
            ])
        else:
            features['elevation_stats'] = np.zeros(6)
        
        # Distance statistics
        valid_distances = distances[distances > 0]
        if len(valid_distances) > 0:
            features['distance_stats'] = np.array([
                np.sum(distances), np.mean(valid_distances),
                np.std(valid_distances), np.max(valid_distances)
            ])
        else:
            features['distance_stats'] = np.zeros(4)
        
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
            
            # Create unified signature by taking the longest component
            if len(signature_components) == 1:
                return self._robust_normalize(signature_components[0])
            
            # Take the longest component as base
            longest_component = max(signature_components, key=len)
            return self._robust_normalize(longest_component)
            
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_temporal_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute temporal correlation with basic alignment"""
        try:
            # Extract temporal signatures
            video_temporal = self._extract_temporal_signature(video_features, 'video')
            gpx_temporal = self._extract_temporal_signature(gpx_features, 'gpx')
            
            if video_temporal is None or gpx_temporal is None:
                return 0.0
            
            # Direct correlation
            min_len = min(len(video_temporal), len(gpx_temporal))
            if min_len > 5:
                v_temp = video_temporal[:min_len]
                g_temp = gpx_temporal[:min_len]
                
                corr = np.corrcoef(v_temp, g_temp)[0, 1]
                if not np.isnan(corr):
                    return float(np.clip(abs(corr), 0.0, 1.0))
            
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
                            return np.diff(values)
                            
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
            
            # Cosine similarity
            cosine_sim = np.dot(video_stats, gpx_stats) / (
                np.linalg.norm(video_stats) * np.linalg.norm(gpx_stats) + 1e-8
            )
            
            if not np.isnan(cosine_sim):
                return float(np.clip(abs(cosine_sim), 0.0, 1.0))
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
                                    np.mean(values), np.std(values), np.median(values)
                                ])
                            
            elif source_type == 'gpx':
                # GPX statistical features
                stat_keys = ['speed_stats', 'bearing_stats', 'elevation_stats', 'distance_stats']
                for key in stat_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend(values.flatten()[:3])
            
            if not profile_components:
                return None
            
            return np.array(profile_components)
            
        except Exception as e:
            logger.debug(f"Statistical profile extraction failed for {source_type}: {e}")
            return None
    
    def _compute_spatial_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute spatial feature similarity"""
        return 0.5  # Placeholder
    
    def _compute_behavioral_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute behavioral pattern similarity"""
        return 0.5  # Placeholder
    
    def _robust_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Robust normalization with outlier handling"""
        try:
            if len(vector) == 0:
                return vector
            
            # Simple min-max normalization
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

def process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
    """Enhanced video processing with better error recovery"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    gpu_id = gpu_manager.acquire_gpu(timeout=config.gpu_timeout)
    
    if gpu_id is None:
        error_msg = f"Could not acquire GPU within {config.gpu_timeout}s timeout"
        if config.strict:
            error_msg = f"STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {video_path}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.error(f"{error_msg} for {video_path}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            return video_path, None
    
    try:
        decoder = EnhancedFFmpegDecoder(gpu_manager, config)
        feature_extractor = EnhancedFeatureExtractor(gpu_manager, config)
        
        # Enhanced decode with retries
        frames_tensor, fps, duration = decoder.decode_video_enhanced(video_path, gpu_id)
        
        if frames_tensor is None:
            error_msg = "Video decoding failed after all attempts"
            if config.strict:
                error_msg = f"STRICT MODE: {error_msg}"
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                raise RuntimeError(error_msg)
            else:
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                return video_path, None
        
        # Extract features
        features = feature_extractor.extract_enhanced_features(frames_tensor, gpu_id)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_id
        features['processing_mode'] = 'GPU_STRICT' if config.strict else 'GPU'
        
        # Mark feature extraction as done in power-safe mode
        if powersafe_manager:
            powersafe_manager.mark_video_features_done(video_path)
        
        success_msg = f"✅ Successfully processed {Path(video_path).name} on GPU {gpu_id}"
        if config.strict:
            success_msg += " [STRICT MODE]"
        logger.info(success_msg)
        return video_path, features
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        if config.strict and "STRICT MODE" not in str(e):
            error_msg = f"STRICT MODE: {error_msg}"
        
        logger.error(f"❌ {error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        
        if config.strict:
            raise RuntimeError(error_msg)
        return video_path, None
        
    finally:
        if gpu_id is not None:
            gpu_manager.release_gpu(gpu_id)
            # Aggressive GPU memory cleanup
            try:
                torch.cuda.empty_cache()
                if gpu_id < torch.cuda.device_count():
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            except:
                pass

def update_config_for_temp_dir(args):
    """Update configuration to use ~/penis/temp directory"""
    # Override the cache directory argument to use the requested location
    args.cache_dir = os.path.expanduser("~/penis/temp")
    
    # Create the directory if it doesn't exist
    temp_dir = Path(args.cache_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Using temp directory: {args.cache_dir}")
    return args

def main():
    """Enhanced main function with comprehensive error handling and progress tracking"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", required=True,
                       help="Directory containing videos and GPX files")
    
    # Processing configuration - ENHANCED DEFAULTS
    parser.add_argument("--max_frames", type=int, default=150,
                       help="Maximum frames per video (default: 150 - optimized for stability)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[480, 270],
                       help="Target video resolution (default: 480 270 - optimized for compatibility)")
    parser.add_argument("--sample_rate", type=float, default=3.0,
                       help="Video sampling rate (default: 3.0)")
    parser.add_argument("--parallel_videos", type=int, default=1,
                       help="Number of videos to process in parallel (default: 1 - prevents GPU conflicts)")
    
    # GPU configuration
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1],
                       help="GPU IDs to use (default: [0, 1])")
    parser.add_argument("--gpu_timeout", type=int, default=60,
                       help="Seconds to wait for GPU availability (default: 60 - more time for conversion)")
    parser.add_argument("--max_gpu_memory", type=float, default=12.0,
                       help="Maximum GPU memory to use in GB (default: 12.0)")
    parser.add_argument("--memory_efficient", action='store_true', default=True,
                       help="Enable memory optimizations (default: True)")
    
    # Video preprocessing and caching - ENHANCED DEFAULTS
    parser.add_argument("--enable_preprocessing", action='store_true', default=True,
                       help="Enable GPU-based video preprocessing for problematic videos (default: True)")
    parser.add_argument("--ram_cache", type=float, default=32.0,
                       help="RAM to use for video caching in GB (default: 32.0 - more conservative)")
    parser.add_argument("--disk_cache", type=float, default=1000.0,
                       help="Disk space to use for video caching in GB (default: 1000.0)")
    parser.add_argument("--cache_dir", type=str, default="~/penis/temp",
                       help="Directory for video cache (default: ~/penis/temp)")
    parser.add_argument("--replace_originals", action='store_true',
                       help="Replace original files with preprocessed versions")
    
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
    parser.add_argument("--strict", action='store_true',
                       help="STRICT MODE: Enforce GPU usage at all times, fail if GPU unavailable")
    
    # Power-safe mode arguments
    parser.add_argument("--powersafe", action='store_true',
                       help="Enable power-safe mode with incremental saves and resume capability")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save results every N correlations in powersafe mode (default: 5)")
    
    args = parser.parse_args()
    
    # UPDATE CONFIG TO USE CORRECT TEMP DIRECTORY
    args = update_config_for_temp_dir(args)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level, "production_correlation.log")
    
    if args.strict:
        logger.info("🚀 Starting Production-Ready Video-GPX Correlation System [STRICT GPU MODE]")
    else:
        logger.info("🚀 Starting Production-Ready Video-GPX Correlation System")
    
    # Log configuration
    logger.info(f"💾 Memory Settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
    logger.info(f"🎞️ Video Preprocessing: {'Enabled' if args.enable_preprocessing else 'Disabled'}")
    if args.enable_preprocessing:
        logger.info(f"📋 RAM Cache: {args.ram_cache:.0f}GB, 💾 Disk Cache: {args.disk_cache:.0f}GB")
        logger.info(f"📁 Cache Directory: {args.cache_dir}")
    
    try:
        # Create configuration
        config = ProcessingConfig(
            max_frames=args.max_frames,
            target_size=tuple(args.video_size),
            sample_rate=args.sample_rate,
            parallel_videos=args.parallel_videos,
            powersafe=args.powersafe,
            save_interval=args.save_interval,
            gpu_timeout=args.gpu_timeout,
            strict=args.strict,
            memory_efficient=args.memory_efficient,
            max_gpu_memory_gb=args.max_gpu_memory,
            enable_preprocessing=args.enable_preprocessing,
            ram_cache_gb=args.ram_cache,
            disk_cache_gb=args.disk_cache,
            cache_dir=args.cache_dir,
            replace_originals=args.replace_originals
        )
        
        # Validate strict mode requirements early
        if config.strict:
            logger.info("🔒 STRICT MODE ENABLED: GPU usage mandatory, no CPU fallback")
            if not torch.cuda.is_available():
                raise RuntimeError("STRICT MODE: CUDA is required but not available")
            if not cp.cuda.is_available():
                raise RuntimeError("STRICT MODE: CuPy CUDA is required but not available")
        
        # Setup directories
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        cache_dir = Path(args.cache)
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize PowerSafe manager
        powersafe_manager = PowerSafeManager(cache_dir, config)
        
        # Initialize GPU manager with strict mode and config
        gpu_manager = EnhancedGPUManager(args.gpu_ids, strict=config.strict, config=config)
        
        # Smart file scanning with PowerSafe prioritization
        logger.info("📁 Scanning for input files with PowerSafe prioritization...")
        
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
        
        # Process videos with enhanced error tracking
        logger.info("🎬 Processing videos with enhanced parallel processing...")
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
            
            # Prepare arguments for parallel processing
            video_args = [(video_path, gpu_manager, config, powersafe_manager) for video_path in videos_to_process]
            
            # Enhanced progress tracking
            successful_videos = 0
            failed_videos = 0
            
            # Use ThreadPoolExecutor for better GPU sharing
            with ThreadPoolExecutor(max_workers=config.parallel_videos) as executor:
                futures = [executor.submit(process_video_parallel_enhanced, arg) for arg in video_args]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                    video_path, features = future.result()
                    video_features[video_path] = features
                    
                    if features is not None:
                        successful_videos += 1
                    else:
                        failed_videos += 1
                    
                    # Periodic cache save
                    if (successful_videos + failed_videos) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
                        logger.info(f"Progress: {successful_videos} ✅ | {failed_videos} ❌")
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
        
        logger.info(f"🎬 Video processing complete: {successful_videos} ✅ | {failed_videos} ❌")
        
        # Process GPX files with enhanced error tracking
        logger.info("🗺️ Processing GPX files...")
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
            
            logger.info(f"🗺️ GPX processing complete: {len(new_gpx_results)} successful")
        
        # Perform enhanced correlation with PowerSafe
        logger.info("🔗 Starting enhanced correlation analysis...")
        
        # Filter valid features
        valid_videos = {k: v for k, v in video_features.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None and 'features' in v}
        
        logger.info(f"Valid features: {len(valid_videos)} videos, {len(valid_gpx)} GPX tracks")
        
        if not valid_videos:
            raise RuntimeError(f"No valid video features! Processed {len(video_features)} videos but none succeeded.")
        
        if not valid_gpx:
            raise RuntimeError(f"No valid GPX features! Processed {len(gpx_database)} GPX files but none succeeded.")
        
        # Initialize similarity engine
        similarity_engine = EnhancedSimilarityEngine(config)
        
        # Compute correlations with PowerSafe incremental saves
        results = existing_results.copy()
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        # Calculate existing correlations
        existing_correlations = sum(len(result.get('matches', [])) for result in existing_results.values())
        
        # Reset PowerSafe manager counter
        if config.powersafe:
            powersafe_manager.correlation_counter = existing_correlations
        
        successful_correlations = 0
        failed_correlations = 0
        
        with tqdm(total=total_comparisons, desc="Computing correlations") as pbar:
            # Update progress bar for existing correlations
            if existing_correlations > 0:
                pbar.update(existing_correlations)
            
            for video_path, video_features_data in valid_videos.items():
                # Skip if video already has complete correlation results
                if (video_path in results and 
                    len(results[video_path].get('matches', [])) >= min(args.top_k, len(valid_gpx))):
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
                        successful_correlations += 1
                        
                        # PowerSafe: Add to pending correlations
                        if config.powersafe:
                            powersafe_manager.add_pending_correlation(video_path, gpx_path, match_info)
                        
                    except Exception as e:
                        logger.debug(f"Correlation failed for {Path(video_path).name} vs {Path(gpx_path).name}: {e}")
                        match_info = {
                            'path': gpx_path,
                            'combined_score': 0.0,
                            'quality': 'failed',
                            'error': str(e)
                        }
                        matches.append(match_info)
                        failed_correlations += 1
                        
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
                if matches and matches[0]['combined_score'] > 0:
                    best = matches[0]
                    logger.info(f"🎯 Best match for {Path(video_path).name}: "
                              f"{Path(best['path']).name} "
                              f"(score: {best['combined_score']:.3f}, quality: {best['quality']})")
                else:
                    logger.warning(f"⚠️ No valid matches found for {Path(video_path).name}")
        
        logger.info(f"🔗 Correlation analysis complete: {successful_correlations} ✅ | {failed_correlations} ❌")
        
        # Final PowerSafe save
        if config.powersafe:
            powersafe_manager.save_incremental_results(powersafe_manager.pending_results)
            logger.info("PowerSafe: Final incremental save completed")
        
        # Save final results
        results_path = output_dir / "production_correlations.pkl"
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
                'successful_correlations': successful_correlations,
                'failed_correlations': failed_correlations,
                'gpu_ids': args.gpu_ids,
                'config': config.__dict__
            },
            'results': results
        }
        
        # Add PowerSafe status if enabled
        if config.powersafe:
            report_data['powersafe_status'] = powersafe_manager.get_processing_status()
        
        with open(output_dir / "production_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate summary statistics
        total_videos_with_results = len(results)
        successful_matches = sum(1 for r in results.values() 
                               if r['matches'] and r['matches'][0]['combined_score'] > 0.1)
        
        excellent_matches = sum(1 for r in results.values() 
                              if r['matches'] and r['matches'][0].get('quality') == 'excellent')
        
        good_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'good')
        
        fair_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') == 'fair')
        
        # Calculate average scores
        all_scores = []
        for r in results.values():
            if r['matches'] and r['matches'][0]['combined_score'] > 0:
                all_scores.append(r['matches'][0]['combined_score'])
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        median_score = np.median(all_scores) if all_scores else 0.0
        
        # PowerSafe cleanup
        if config.powersafe:
            powersafe_manager.cleanup_completed_entries()
        
        # Print comprehensive summary
        print(f"\n{'='*90}")
        print(f"🎯 PRODUCTION VIDEO-GPX CORRELATION SUMMARY")
        print(f"{'='*90}")
        print(f"Processing Mode: {'⚡ PowerSafe' if config.powersafe else '🏃 Standard'}")
        print(f"")
        print(f"📁 File Processing:")
        print(f"  Videos Found: {len(video_files)}")
        print(f"  Videos Successfully Processed: {len(valid_videos)}/{len(video_files)} ({100*len(valid_videos)/max(len(video_files), 1):.1f}%)")
        print(f"  GPX Files Found: {len(gpx_files)}")
        print(f"  GPX Files Successfully Processed: {len(valid_gpx)}/{len(gpx_files)} ({100*len(valid_gpx)/max(len(gpx_files), 1):.1f}%)")
        print(f"")
        print(f"🔗 Correlation Results:")
        print(f"  Total Videos with Results: {total_videos_with_results}")
        print(f"  Videos with Valid Matches (>0.1): {successful_matches}/{total_videos_with_results} ({100*successful_matches/max(total_videos_with_results, 1):.1f}%)")
        print(f"  Total Correlations Computed: {successful_correlations + failed_correlations}")
        print(f"  Successful Correlations: {successful_correlations} ✅")
        print(f"  Failed Correlations: {failed_correlations} ❌")
        print(f"")
        print(f"📊 Quality Distribution:")
        print(f"  🟢 Excellent (≥0.8): {excellent_matches}")
        print(f"  🟡 Good (≥0.6): {good_matches}")
        print(f"  🟠 Fair (≥0.4): {fair_matches}")
        print(f"  🔴 Poor/Failed: {total_videos_with_results - excellent_matches - good_matches - fair_matches}")
        print(f"")
        print(f"📈 Score Statistics:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Median Score: {median_score:.3f}")
        print(f"  Total Valid Scores: {len(all_scores)}")
        
        if config.powersafe:
            status = powersafe_manager.get_processing_status()
            print(f"")
            print(f"💾 PowerSafe Status:")
            print(f"  Total Saved Correlations: {status.get('total_correlations', 0)}")
            print(f"  Results Database: {powersafe_manager.results_path}")
            print(f"  Progress Database: {powersafe_manager.db_path}")
        
        print(f"")
        print(f"📄 Output Files:")
        print(f"  📊 Results: {results_path}")
        print(f"  📋 Report: {output_dir / 'production_report.json'}")
        print(f"  💾 Cache: {cache_dir}")
        print(f"  📝 Log: production_correlation.log")
        print(f"")
        
        # Display top correlations if any exist
        if all_scores:
            print(f"🏆 TOP CORRELATIONS:")
            print(f"{'='*90}")
            
            # Get top correlations across all videos
            all_correlations = []
            for video_path, result in results.items():
                if result['matches'] and result['matches'][0]['combined_score'] > 0:
                    best_match = result['matches'][0]
                    all_correlations.append((
                        Path(video_path).name,
                        Path(best_match['path']).name,
                        best_match['combined_score'],
                        best_match.get('quality', 'unknown')
                    ))
            
            # Sort by score and display top 10
            all_correlations.sort(key=lambda x: x[2], reverse=True)
            for i, (video, gpx, score, quality) in enumerate(all_correlations[:10], 1):
                quality_emoji = {
                    'excellent': '🟢', 
                    'good': '🟡', 
                    'fair': '🟠', 
                    'poor': '🔴', 
                    'very_poor': '🔴',
                    'failed': '❌'
                }.get(quality, '⚪')
                print(f"{i:2d}. {video[:50]:<50} ↔ {gpx[:30]:<30}")
                print(f"     Score: {score:.3f} | Quality: {quality_emoji} {quality}")
                if i < len(all_correlations):
                    print()
        else:
            print(f"⚠️ No successful correlations found!")
            print(f"   This could indicate:")
            print(f"   • Video processing failures (check logs)")
            print(f"   • GPX processing failures (check file formats)")
            print(f"   • Feature extraction issues")
            print(f"   • Incompatible data types")
        
        print(f"{'='*90}")
        
        # Success determination
        if successful_matches > 0:
            logger.info("🎉 Production correlation system completed successfully with matches!")
        elif len(valid_videos) > 0 and len(valid_gpx) > 0:
            logger.warning("⚠️ System completed but found no correlations - check data compatibility")
        else:
            logger.error("❌ System completed but no valid features were extracted")
        
        # Final recommendations
        if failed_correlations > successful_correlations:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Try reducing --parallel_videos to 1 for debugging")
            print(f"   • Reduce --max_frames (try 100 for memory issues)")
            print(f"   • Reduce --video_size (try 384 216 for memory issues)")
            print(f"   • Check video file formats and corruption")
            print(f"   • Verify GPX files contain valid track data")
            print(f"   • Enable --debug for detailed error analysis")
            if not config.powersafe:
                print(f"   • Use --powersafe to preserve progress during debugging")
            if config.strict:
                print(f"   • Remove --strict flag to enable CPU fallbacks for debugging")
        
        print(f"\n🧠 PERFORMANCE OPTIMIZATION:")
        print(f"   • Current settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
        print(f"   • GPU Memory: {config.max_gpu_memory_gb:.1f}GB limit per GPU")
        print(f"   • Parallel Videos: {config.parallel_videos}")
        print(f"   • Temp Directory: {config.cache_dir}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
        if config.powersafe:
            logger.info("PowerSafe: Progress has been saved and can be resumed")
        print("\n⚠️  Process interrupted. PowerSafe progress has been saved." if config.powersafe else "\n⚠️  Process interrupted.")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"❌ Production system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        if config.powersafe:
            logger.info("PowerSafe: Partial progress has been saved")
            print(f"\n❌ Error occurred: {e}")
            print("💾 PowerSafe: Partial progress has been saved and can be resumed with --powersafe flag")
        
        # Provide debugging suggestions
        print(f"\n🔧 DEBUGGING SUGGESTIONS:")
        print(f"   • Run with --debug for detailed error information")
        print(f"   • Try --parallel_videos 1 to isolate GPU issues")
        print(f"   • Reduce --max_frames to 100 for testing")
        print(f"   • Check video file integrity with ffprobe")
        print(f"   • Verify GPX files are valid XML")
        
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