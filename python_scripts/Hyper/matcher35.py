#!/usr/bin/env python3
"""
Production-Ready Multi-GPU Video-GPX Correlation Script

Key Features:
- Multi-GPU processing with proper queuing
- Enhanced video preprocessing with GPU acceleration
- PowerSafe mode for long-running operations
- Robust error handling and recovery
- Memory optimization and cleanup
- Uses ~/penis/temp for temporary storage

Usage:
    python production_matcher.py -d /path/to/data --gpu_ids 0 1
    python production_matcher.py -d /path/to/data --powersafe --debug
"""

import cv2
import numpy as np
import gpxpy
import pandas as pd
import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
import queue
import shutil
import sys
from typing import Dict, List, Tuple, Optional, Any
import psutil
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager
from threading import Lock

# Optional imports with fallbacks
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
    max_frames: int = 150
    target_size: Tuple[int, int] = (480, 270)
    sample_rate: float = 3.0
    parallel_videos: int = 1
    gpu_memory_fraction: float = 0.7
    motion_threshold: float = 0.01
    temporal_window: int = 10
    powersafe: bool = False
    save_interval: int = 5
    gpu_timeout: int = 60
    strict: bool = False
    memory_efficient: bool = True
    max_gpu_memory_gb: float = 12.0
    enable_preprocessing: bool = True
    ram_cache_gb: float = 32.0
    disk_cache_gb: float = 1000.0
    cache_dir: str = "~/penis/temp"
    replace_originals: bool = False

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
    """Power-safe processing manager with incremental saves"""
    
    def __init__(self, cache_dir: Path, config: ProcessingConfig):
        self.cache_dir = cache_dir
        self.config = config
        self.db_path = cache_dir / "powersafe_progress.db"
        self.results_path = cache_dir / "incremental_results.json"
        self.correlation_counter = 0
        self.pending_results = {}
        
        if config.powersafe:
            self._init_progress_db()
            logger.info("PowerSafe mode enabled")
    
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
    
    def add_pending_correlation(self, video_path: str, gpx_path: str, match_info: Dict):
        """Add correlation result to pending batch"""
        if not self.config.powersafe:
            return
        
        if video_path not in self.pending_results:
            self.pending_results[video_path] = {'matches': []}
        
        self.pending_results[video_path]['matches'].append(match_info)
        self.correlation_counter += 1
        
        # Save correlation to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO correlation_progress 
                (video_path, gpx_path, correlation_score, correlation_details, processed_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (video_path, gpx_path, match_info['combined_score'], json.dumps(match_info)))
            conn.commit()
        
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

class EnhancedGPUManager:
    """Enhanced GPU management with memory monitoring"""
    
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
        """Validate GPU availability and memory"""
        if not torch.cuda.is_available():
            if self.strict:
                raise RuntimeError("STRICT MODE: CUDA is required but not available")
            else:
                raise RuntimeError("CUDA not available")
        
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_ids:
            if gpu_id >= available_gpus:
                if self.strict:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} is required but not available")
                else:
                    raise RuntimeError(f"GPU {gpu_id} not available")
        
        # Check GPU memory
        for gpu_id in self.gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            
            if memory_gb < 4:
                if self.strict:
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} has insufficient memory: {memory_gb:.1f}GB")
                else:
                    logger.warning(f"GPU {gpu_id} has only {memory_gb:.1f}GB memory")
            
            logger.info(f"GPU {gpu_id}: {props.name} ({memory_gb:.1f}GB)" + 
                       (" [STRICT MODE]" if self.strict else ""))
    
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
        except Exception as e:
            logger.debug(f"Memory cleanup warning for GPU {gpu_id}: {e}")
    
    def acquire_gpu(self, timeout: int = 60) -> Optional[int]:
        """Acquire GPU with timeout"""
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            self.gpu_usage[gpu_id] += 1
            
            # Verify GPU is still functional in strict mode
            if self.strict:
                try:
                    with torch.cuda.device(gpu_id):
                        test_tensor = torch.zeros(10, 10, device=f'cuda:{gpu_id}', dtype=torch.float32)
                        del test_tensor
                        torch.cuda.empty_cache()
                except Exception as e:
                    self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
                    self.gpu_queue.put(gpu_id)
                    raise RuntimeError(f"STRICT MODE: GPU {gpu_id} became unavailable: {e}")
            
            return gpu_id
        except queue.Empty:
            if self.strict:
                raise RuntimeError(f"STRICT MODE: Could not acquire any GPU within {timeout}s timeout")
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release GPU after processing with memory cleanup"""
        self.cleanup_gpu_memory(gpu_id)
        self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
        self.gpu_queue.put(gpu_id)

class VideoPreprocessor:
    """GPU-accelerated video preprocessor with caching"""
    
    def __init__(self, config: ProcessingConfig, gpu_manager: EnhancedGPUManager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.ram_cache = {}
        self.ram_usage = 0
        self.max_ram_bytes = int(config.ram_cache_gb * 1024**3)
        
        # Setup cache directories
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
        
        logger.info(f"Video Preprocessor initialized:")
        logger.info(f"  RAM Cache: {config.ram_cache_gb:.1f}GB")
        logger.info(f"  Disk Cache: {self.disk_cache_dir}")
        logger.info(f"  Temp Processing: {self.temp_processing_dir}")
    
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
        """Preprocess video using GPU acceleration"""
        video_hash = self._get_video_hash(video_path)
        
        # Check if already cached
        if video_hash in self.ram_cache:
            logger.debug(f"Using RAM cached video: {Path(video_path).name}")
            return self.ram_cache[video_hash]
        
        if video_hash in self.cache_index:
            cache_info = self.cache_index[video_hash]
            cached_path = cache_info['path']
            if os.path.exists(cached_path):
                logger.debug(f"Using disk cached video: {Path(video_path).name}")
                return cached_path
        
        # Try direct processing first
        if self._test_direct_processing(video_path, gpu_id):
            logger.debug(f"Direct processing works: {Path(video_path).name}")
            return video_path
        
        # Need to preprocess
        logger.info(f"Preprocessing problematic video: {Path(video_path).name}")
        preprocessed_path = self._gpu_convert_video(video_path, gpu_id)
        
        if preprocessed_path:
            self._cache_video(video_hash, preprocessed_path, video_path)
            return preprocessed_path
        
        return None
    
    def _test_direct_processing(self, video_path: str, gpu_id: int) -> bool:
        """Test if video can be processed directly without preprocessing"""
        try:
            if not os.path.exists(video_path):
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size < 1024:
                return False
            
            # Quick ffprobe test
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', video_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False
            
            # Test GPU decoding
            return self._test_gpu_decode_minimal(video_path, gpu_id)
            
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
            
            success = (result.returncode == 0 and test_output.exists() and 
                      test_output.stat().st_size > 1024)
            
            if test_output.exists():
                test_output.unlink()
            
            return success
                    
        except Exception as e:
            logger.debug(f"GPU decode test failed for {Path(video_path).name}: {e}")
            return False
    
    def _gpu_convert_video(self, video_path: str, gpu_id: int) -> Optional[str]:
        """Convert video using GPU acceleration"""
        try:
            video_name = Path(video_path).stem
            timestamp = int(time.time() * 1000000)
            output_path = self.disk_cache_dir / f"{video_name}_converted_{timestamp}.mp4"
            
            logger.info(f"GPU converting {Path(video_path).name} on GPU {gpu_id}")
            
            strategies = [
                ("NVENC Hardware Encode", self._convert_with_nvenc),
                ("CUDA Accelerated Decode", self._convert_with_cuda),
                ("CPU Fallback", self._convert_cpu_fallback)
            ]
            
            for i, (strategy_name, strategy_func) in enumerate(strategies):
                try:
                    logger.debug(f"Trying strategy {i+1}/3: {strategy_name}")
                    if strategy_func(video_path, output_path, gpu_id):
                        if self._verify_converted_video(str(output_path), gpu_id):
                            logger.info(f"Successfully converted using {strategy_name}: {Path(video_path).name}")
                            return str(output_path)
                        else:
                            if output_path.exists():
                                output_path.unlink()
                    else:
                        logger.debug(f"{strategy_name} conversion failed")
                except Exception as e:
                    logger.debug(f"{strategy_name} failed with exception: {e}")
                    if output_path.exists():
                        output_path.unlink()
            
            logger.error(f"All conversion strategies failed for {Path(video_path).name}")
            return None
            
        except Exception as e:
            logger.error(f"GPU conversion failed for {video_path}: {e}")
            return None
    
    def _convert_with_nvenc(self, input_path: str, output_path: Path, gpu_id: int) -> bool:
        """NVENC conversion"""
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', input_path,
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
            '-profile:v', 'high',
            '-pix_fmt', 'yuv420p',
            '-cq', '23',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        return self._run_ffmpeg_command(cmd, "NVENC")
    
    def _convert_with_cuda(self, input_path: str, output_path: Path, gpu_id: int) -> bool:
        """CUDA-accelerated conversion"""
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-profile:v', 'high',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        return self._run_ffmpeg_command(cmd, "CUDA")
    
    def _convert_cpu_fallback(self, input_path: str, output_path: Path, gpu_id: int) -> bool:
        """CPU fallback conversion"""
        if self.config.strict:
            return False
        
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p',
            '-crf', '28',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '96k',
            str(output_path)
        ]
        
        return self._run_ffmpeg_command(cmd, "CPU Fallback")
    
    def _run_ffmpeg_command(self, cmd: List[str], strategy_name: str) -> bool:
        """Run FFmpeg command with proper error handling"""
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
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
    
    def _verify_converted_video(self, video_path: str, gpu_id: int) -> bool:
        """Verify that converted video is actually processable"""
        try:
            if not os.path.exists(video_path):
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size < 1024:
                return False
            
            return self._test_gpu_decode_minimal(video_path, gpu_id)
            
        except Exception as e:
            logger.debug(f"Video verification failed: {e}")
            return False
    
    def _cache_video(self, video_hash: str, preprocessed_path: str, original_path: str):
        """Cache preprocessed video"""
        video_size_bytes = os.path.getsize(preprocessed_path)
        video_size_mb = video_size_bytes / (1024**2)
        
        # Try RAM cache first if small enough
        if (video_size_bytes < 500 * 1024**2 and  
            self.ram_usage + video_size_bytes < self.max_ram_bytes):
            
            try:
                ram_cache_path = self.temp_processing_dir / f"ram_cache_{video_hash}.mp4"
                shutil.copy2(preprocessed_path, ram_cache_path)
                
                self.ram_cache[video_hash] = str(ram_cache_path)
                self.ram_usage += video_size_bytes
                
                logger.debug(f"Cached {video_size_mb:.1f}MB video in RAM: {Path(original_path).name}")
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
        
        logger.debug(f"Cached {video_size_mb:.1f}MB video on disk: {Path(original_path).name}")
    
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
            
            logger.info("Video preprocessor cleanup completed")
            
        except Exception as e:
            logger.warning(f"Video preprocessor cleanup failed: {e}")

class EnhancedFFmpegDecoder:
    """Enhanced FFmpeg decoder with GPU preprocessing"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.temp_dirs = {}
        
        # Initialize video preprocessor
        if config.enable_preprocessing:
            self.preprocessor = VideoPreprocessor(config, gpu_manager)
        else:
            self.preprocessor = None
        
        # Create temp directories per GPU
        base_temp = Path(config.cache_dir) / "gpu_temp"
        base_temp.mkdir(parents=True, exist_ok=True)
        
        for gpu_id in gpu_manager.gpu_ids:
            self.temp_dirs[gpu_id] = base_temp / f'gpu_{gpu_id}'
            self.temp_dirs[gpu_id].mkdir(exist_ok=True)
        
        logger.info(f"Enhanced decoder initialized for GPUs: {gpu_manager.gpu_ids}")
    
    def decode_video_enhanced(self, video_path: str, gpu_id: int) -> Tuple[Optional[torch.Tensor], float, float]:
        """Enhanced video decoding with preprocessing"""
        try:
            # Preprocess video if enabled
            if self.preprocessor:
                processed_video_path = self.preprocessor.preprocess_video(video_path, gpu_id)
                if processed_video_path is None:
                    raise RuntimeError("Video preprocessing failed")
                actual_video_path = processed_video_path
            else:
                actual_video_path = video_path
            
            # Get video info
            video_info = self._get_video_info(actual_video_path)
            if not video_info:
                raise RuntimeError("Could not get video info")
            
            # Decode frames
            frames_tensor = self._decode_uniform_frames(actual_video_path, video_info, gpu_id)
            
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
    
    def _decode_uniform_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """Memory-optimized uniform frame sampling"""
        temp_dir = self.temp_dirs[gpu_id]
        output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
        
        # Calculate sampling rate
        total_frames = int(video_info['duration'] * video_info['fps'])
        max_frames = self.config.max_frames
        
        # Ensure target size is even numbers
        target_width = self.config.target_size[0]
        target_height = self.config.target_size[1]
        
        if target_width % 2 != 0:
            target_width += 1
        if target_height % 2 != 0:
            target_height += 1
        
        if total_frames > max_frames:
            sample_rate = total_frames / max_frames
            vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,select=not(mod(n\\,{int(sample_rate)}))'
        else:
            vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2'
        
        # Enhanced CUDA command
        cuda_cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-vf', vf_filter,
            '-frames:v', str(min(max_frames, total_frames)),
            '-q:v', '2',
            '-threads', '1',
            output_pattern
        ]
        
        # CPU fallback command
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
            # Try CUDA first
            try:
                result = subprocess.run(cuda_cmd, check=True, capture_output=True, timeout=300)
                logger.debug(f"CUDA decoding successful: {Path(video_path).name}")
                success = True
            except subprocess.CalledProcessError:
                success = False
            
            # CPU fallback if allowed
            if not success and cpu_cmd:
                try:
                    # Clean up any partial files
                    for f in glob.glob(os.path.join(temp_dir, 'frame_*.jpg')):
                        try:
                            os.remove(f)
                        except:
                            pass
                    
                    result = subprocess.run(cpu_cmd, check=True, capture_output=True, timeout=300)
                    logger.debug(f"CPU fallback decoding successful: {Path(video_path).name}")
                    success = True
                except subprocess.CalledProcessError:
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
            
            def forward(self, x):
                features = self.backbone(x)
                return {
                    'scene_features': self.scene_head(features),
                    'motion_features': self.motion_head(features)
                }
        
        model = EnhancedFeatureNet()
        model.eval()
        return model
    
    def extract_enhanced_features(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract enhanced features with robust error handling"""
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
                    value = value.view(batch_size, num_frames, -1)[0]
                    features[key] = value.cpu().numpy()
                
                # Enhanced motion features
                motion_features = self._compute_enhanced_motion(frames_tensor[0], device)
                features.update(motion_features)
                
                # Color features
                color_features = self._compute_enhanced_color(frames_tensor[0], device)
                features.update(color_features)
                
                # Edge features
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
            'acceleration': np.zeros(num_frames)
        }
        
        if num_frames < 2:
            return features
        
        # Convert to grayscale
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        # Compute optical flow approximation
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
        
        # Compute acceleration
        motion_mag = features['motion_magnitude']
        for i in range(1, num_frames - 1):
            features['acceleration'][i] = motion_mag[i + 1] - motion_mag[i]
        
        return features
    
    def _compute_enhanced_color(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Compute enhanced color features"""
        num_frames = frames.shape[0]
        
        # Color variance over time
        color_variance = torch.var(frames, dim=[2, 3])
        mean_color_variance = torch.mean(color_variance, dim=1).cpu().numpy()
        
        # Color histograms
        histograms = []
        for i in range(num_frames):
            frame = frames[i]
            hist_features = []
            for c in range(3):
                channel_mean = torch.mean(frame[c]).item()
                channel_std = torch.std(frame[c]).item()
                hist_features.extend([channel_mean, channel_std])
            histograms.append(hist_features)
        
        return {
            'color_variance': mean_color_variance,
            'color_histograms': np.array(histograms)
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
        
        return {
            'edge_density': edge_density
        }

class RobustGPXProcessor:
    """Robust GPX processor with comprehensive error handling"""
    
    def __init__(self, config: ProcessingConfig, powersafe_manager: Optional[PowerSafeManager] = None):
        self.config = config
        self.powersafe_manager = powersafe_manager
        
        if config.strict:
            if not cp.cuda.is_available():
                raise RuntimeError("STRICT MODE: CuPy CUDA is required for GPX processing")
            self.use_gpu = True
        else:
            self.use_gpu = cp.cuda.is_available()
        
        logger.info(f"GPX processor initialized (GPU: {self.use_gpu})" + 
                   (" [STRICT MODE]" if config.strict else ""))
    
    def process_gpx_files(self, gpx_paths: List[str], max_workers: int = None) -> Dict[str, Any]:
        """Process GPX files with robust error handling"""
        if max_workers is None:
            max_workers = min(8, mp.cpu_count())
        
        logger.info(f"Processing {len(gpx_paths)} GPX files with {max_workers} workers")
        
        results = {}
        failed_files = []
        
        # Process sequentially for debugging first 10
        for i, path in enumerate(tqdm(gpx_paths[:10], desc="Debugging GPX (first 10)")):
            if self.powersafe_manager:
                self.powersafe_manager.mark_video_processing(path)
            
            try:
                data = self._parse_gpx_safe(path)
                if data is not None:
                    results[path] = data
                else:
                    failed_files.append(path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                failed_files.append(path)
        
        # Process remaining files in parallel if debugging successful
        if results:
            logger.info(f"GPX debugging successful, processing remaining files...")
            remaining_paths = gpx_paths[10:]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._parse_gpx_safe, path): path for path in remaining_paths}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing remaining GPX"):
                    path = futures[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[path] = data
                        else:
                            failed_files.append(path)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        failed_files.append(path)
        else:
            logger.warning("All GPX debugging samples failed")
            failed_files.extend(gpx_paths[10:])
        
        logger.info(f"Successfully processed {len(results)}/{len(gpx_paths)} GPX files")
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
                return None
            
            df = pd.DataFrame(points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate coordinates
            if (df['lat'].isna().any() or df['lon'].isna().any() or
                not (-90 <= df['lat'].min() <= df['lat'].max() <= 90) or
                not (-180 <= df['lon'].min() <= df['lon'].max() <= 180)):
                return None
            
            # Compute features
            if self.use_gpu:
                try:
                    enhanced_data = self._compute_robust_features_gpu(df)
                except Exception as e:
                    if self.config.strict:
                        raise RuntimeError(f"STRICT MODE: GPU processing failed: {e}")
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
        distances_gpu = self._compute_distances_gpu(lats_gpu, lons_gpu)
        
        # Compute motion features
        motion_features = self._compute_motion_features_gpu(
            lats_gpu, lons_gpu, elevs_gpu, time_diffs_gpu, distances_gpu
        )
        
        # Convert to CPU
        cpu_features = {
            key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
            for key, value in motion_features.items()
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
    
    def _compute_distances_gpu(self, lats: cp.ndarray, lons: cp.ndarray) -> cp.ndarray:
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
    
    def _compute_motion_features_gpu(self, lats: cp.ndarray, lons: cp.ndarray, 
                                   elevs: cp.ndarray, time_diffs: cp.ndarray, 
                                   distances: cp.ndarray) -> Dict[str, cp.ndarray]:
        """Compute motion features using GPU"""
        n = len(lats)
        
        features = {
            'speed': cp.zeros(n),
            'acceleration': cp.zeros(n),
            'bearing': cp.zeros(n)
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
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_gpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
        return features
    
    def _compute_bearings_gpu(self, lat1: cp.ndarray, lon1: cp.ndarray, 
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
    
    def _compute_robust_features_cpu(self, df: pd.DataFrame) -> Dict:
        """Compute robust features using CPU (fallback)"""
        n_points = len(df)
        
        # Convert to numpy arrays
        lats = df['lat'].values
        lons = df['lon'].values
        elevs = df['elevation'].values
        
        # Compute time differences
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        
        # Compute distances
        distances = self._compute_distances_cpu(lats, lons)
        
        # Compute motion features
        motion_features = self._compute_motion_features_cpu(lats, lons, elevs, time_diffs, distances)
        
        # Add metadata
        duration = self._compute_duration_safe(df['timestamp'])
        total_distance = np.sum(distances)
        
        return {
            'df': df,
            'features': motion_features,
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
                if 0 < seconds <= 3600:
                    time_diffs.append(seconds)
                else:
                    time_diffs.append(1.0)
                    
            except Exception:
                time_diffs.append(1.0)
        
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
            'bearing': np.zeros(n)
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
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_cpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'][1:] = bearings
        
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
    
    def _compute_duration_safe(self, timestamps: pd.Series) -> float:
        """Safely compute duration"""
        try:
            duration_delta = timestamps.iloc[-1] - timestamps.iloc[0]
            if hasattr(duration_delta, 'total_seconds'):
                return duration_delta.total_seconds()
            else:
                return float(duration_delta / np.timedelta64(1, 's'))
        except Exception:
            return 3600.0

class EnhancedSimilarityEngine:
    """Enhanced similarity computation with multiple methods"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.weights = {
            'motion_dynamics': 0.40,
            'temporal_correlation': 0.30,
            'statistical_profile': 0.30
        }
    
    def compute_enhanced_similarity(self, video_features: Dict, gpx_features: Dict) -> Dict[str, float]:
        """Compute enhanced similarity with multiple methods"""
        try:
            similarities = {}
            
            # Motion dynamics similarity
            similarities['motion_dynamics'] = self._compute_motion_similarity(video_features, gpx_features)
            
            # Temporal correlation
            similarities['temporal_correlation'] = self._compute_temporal_similarity(video_features, gpx_features)
            
            # Statistical profile matching
            similarities['statistical_profile'] = self._compute_statistical_similarity(video_features, gpx_features)
            
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
        try:
            if source_type == 'video':
                # Video motion features
                if 'motion_magnitude' in features:
                    values = features['motion_magnitude']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        if np.isfinite(values).all():
                            return self._robust_normalize(values)
                            
            elif source_type == 'gpx':
                # GPX motion features
                if 'speed' in features:
                    values = features['speed']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        if np.isfinite(values).all():
                            return self._robust_normalize(values)
            
            return None
            
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_temporal_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute temporal correlation"""
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
                if 'motion_magnitude' in features:
                    values = features['motion_magnitude']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return np.diff(values)
                            
            elif source_type == 'gpx':
                if 'speed' in features:
                    values = features['speed']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return np.diff(values)
            
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
                for key in ['motion_magnitude', 'color_variance', 'edge_density']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend([
                                    np.mean(values), np.std(values), np.median(values)
                                ])
                            
            elif source_type == 'gpx':
                # GPX statistical features
                for key in ['speed', 'acceleration', 'bearing']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend([
                                    np.mean(values), np.std(values), np.median(values)
                                ])
            
            if not profile_components:
                return None
            
            return np.array(profile_components)
            
        except Exception as e:
            logger.debug(f"Statistical profile extraction failed for {source_type}: {e}")
            return None
    
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
        
        success_msg = f"Successfully processed {Path(video_path).name} on GPU {gpu_id}"
        if config.strict:
            success_msg += " [STRICT MODE]"
        logger.info(success_msg)
        return video_path, features
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        if config.strict and "STRICT MODE" not in str(e):
            error_msg = f"STRICT MODE: {error_msg}"
        
        logger.error(f"{error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        
        if config.strict:
            raise RuntimeError(error_msg)
        return video_path, None
        
    finally:
        if gpu_id is not None:
            gpu_manager.release_gpu(gpu_id)
            # Cleanup GPU memory
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
    args.cache_dir = os.path.expanduser("~/penis/temp")
    
    # Create the directory if it doesn't exist
    temp_dir = Path(args.cache_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using temp directory: {args.cache_dir}")
    return args

def main():
    """Enhanced main function with comprehensive error handling"""
    
    parser = argparse.ArgumentParser(
        description="Production-Ready Multi-GPU Video-GPX Correlation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", required=True,
                       help="Directory containing videos and GPX files")
    
    # Processing configuration
    parser.add_argument("--max_frames", type=int, default=150,
                       help="Maximum frames per video (default: 150)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[480, 270],
                       help="Target video resolution (default: 480 270)")
    parser.add_argument("--sample_rate", type=float, default=3.0,
                       help="Video sampling rate (default: 3.0)")
    parser.add_argument("--parallel_videos", type=int, default=1,
                       help="Number of videos to process in parallel (default: 1)")
    
    # GPU configuration
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1],
                       help="GPU IDs to use (default: [0, 1])")
    parser.add_argument("--gpu_timeout", type=int, default=60,
                       help="Seconds to wait for GPU availability (default: 60)")
    parser.add_argument("--max_gpu_memory", type=float, default=12.0,
                       help="Maximum GPU memory to use in GB (default: 12.0)")
    parser.add_argument("--memory_efficient", action='store_true', default=True,
                       help="Enable memory optimizations (default: True)")
    
    # Video preprocessing and caching
    parser.add_argument("--enable_preprocessing", action='store_true', default=True,
                       help="Enable GPU-based video preprocessing (default: True)")
    parser.add_argument("--ram_cache", type=float, default=32.0,
                       help="RAM to use for video caching in GB (default: 32.0)")
    parser.add_argument("--disk_cache", type=float, default=1000.0,
                       help="Disk space to use for video caching in GB (default: 1000.0)")
    parser.add_argument("--cache_dir", type=str, default="~/penis/temp",
                       help="Directory for video cache (default: ~/penis/temp)")
    
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
                       help="STRICT MODE: Enforce GPU usage at all times")
    
    # Power-safe mode arguments
    parser.add_argument("--powersafe", action='store_true',
                       help="Enable power-safe mode with incremental saves")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save results every N correlations in powersafe mode (default: 5)")
    
    args = parser.parse_args()
    
    # Update config to use correct temp directory
    args = update_config_for_temp_dir(args)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level, "production_correlation.log")
    
    if args.strict:
        logger.info("Starting Production-Ready Video-GPX Correlation System [STRICT GPU MODE]")
    else:
        logger.info("Starting Production-Ready Video-GPX Correlation System")
    
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
            cache_dir=args.cache_dir
        )
        
        # Validate strict mode requirements early
        if config.strict:
            logger.info("STRICT MODE ENABLED: GPU usage mandatory, no CPU fallback")
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
        
        # Initialize GPU manager
        gpu_manager = EnhancedGPUManager(args.gpu_ids, strict=config.strict, config=config)
        
        # Scan for files
        logger.info("Scanning for input files...")
        
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
        video_files = sorted(list(set(video_files)))
        
        gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
        gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
        gpx_files = sorted(list(set(gpx_files)))
        
        logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
        
        if not video_files or not gpx_files:
            raise RuntimeError("Need both video and GPX files")
        
        # Load existing results in PowerSafe mode
        existing_results = {}
        if config.powersafe:
            existing_results = powersafe_manager.load_existing_results()
        
        # Process videos
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
            
            # Prepare arguments for parallel processing
            video_args = [(video_path, gpu_manager, config, powersafe_manager) for video_path in videos_to_process]
            
            # Progress tracking
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
                        logger.info(f"Progress: {successful_videos} success | {failed_videos} failed")
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
        
        logger.info(f"Video processing complete: {successful_videos} success | {failed_videos} failed")
        
        # Process GPX files
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
            
            logger.info(f"GPX processing complete: {len(new_gpx_results)} successful")
        
        # Perform enhanced correlation
        logger.info("Starting enhanced correlation analysis...")
        
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
        
        # Compute correlations
        results = existing_results.copy()
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        successful_correlations = 0
        failed_correlations = 0
        
        with tqdm(total=total_comparisons, desc="Computing correlations") as pbar:
            for video_path, video_features_data in valid_videos.items():
                matches = []
                
                for gpx_path, gpx_data in valid_gpx.items():
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
                
                # Log best match
                if matches and matches[0]['combined_score'] > 0:
                    best = matches[0]
                    logger.info(f"Best match for {Path(video_path).name}: "
                              f"{Path(best['path']).name} "
                              f"(score: {best['combined_score']:.3f}, quality: {best['quality']})")
                else:
                    logger.warning(f"No valid matches found for {Path(video_path).name}")
        
        logger.info(f"Correlation analysis complete: {successful_correlations} success | {failed_correlations} failed")
        
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
        
        # Print comprehensive summary
        print(f"\n{'='*90}")
        print(f"PRODUCTION VIDEO-GPX CORRELATION SUMMARY")
        print(f"{'='*90}")
        print(f"Processing Mode: {'PowerSafe' if config.powersafe else 'Standard'}")
        print(f"")
        print(f"File Processing:")
        print(f"  Videos Found: {len(video_files)}")
        print(f"  Videos Successfully Processed: {len(valid_videos)}/{len(video_files)} ({100*len(valid_videos)/max(len(video_files), 1):.1f}%)")
        print(f"  GPX Files Found: {len(gpx_files)}")
        print(f"  GPX Files Successfully Processed: {len(valid_gpx)}/{len(gpx_files)} ({100*len(valid_gpx)/max(len(gpx_files), 1):.1f}%)")
        print(f"")
        print(f"Correlation Results:")
        print(f"  Total Videos with Results: {total_videos_with_results}")
        print(f"  Videos with Valid Matches (>0.1): {successful_matches}/{total_videos_with_results} ({100*successful_matches/max(total_videos_with_results, 1):.1f}%)")
        print(f"  Total Correlations Computed: {successful_correlations + failed_correlations}")
        print(f"  Successful Correlations: {successful_correlations}")
        print(f"  Failed Correlations: {failed_correlations}")
        print(f"")
        print(f"Quality Distribution:")
        print(f"  Excellent (≥0.8): {excellent_matches}")
        print(f"  Good (≥0.6): {good_matches}")
        print(f"  Fair (≥0.4): {fair_matches}")
        print(f"  Poor/Failed: {total_videos_with_results - excellent_matches - good_matches - fair_matches}")
        print(f"")
        print(f"Score Statistics:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Median Score: {median_score:.3f}")
        print(f"  Total Valid Scores: {len(all_scores)}")
        print(f"")
        print(f"Output Files:")
        print(f"  Results: {results_path}")
        print(f"  Report: {output_dir / 'production_report.json'}")
        print(f"  Cache: {cache_dir}")
        print(f"  Log: production_correlation.log")
        print(f"")
        
        # Display top correlations if any exist
        if all_scores:
            print(f"TOP CORRELATIONS:")
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
            print(f"No successful correlations found!")
            print(f"   This could indicate:")
            print(f"   • Video processing failures (check logs)")
            print(f"   • GPX processing failures (check file formats)")
            print(f"   • Feature extraction issues")
            print(f"   • Incompatible data types")
        
        print(f"{'='*90}")
        
        # Success determination
        if successful_matches > 0:
            logger.info("Production correlation system completed successfully with matches!")
        elif len(valid_videos) > 0 and len(valid_gpx) > 0:
            logger.warning("System completed but found no correlations - check data compatibility")
        else:
            logger.error("System completed but no valid features were extracted")
        
        # Final recommendations
        if failed_correlations > successful_correlations:
            print(f"\nRECOMMENDATIONS:")
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
        
        print(f"\nPERFORMANCE OPTIMIZATION:")
        print(f"   • Current settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
        print(f"   • GPU Memory: {config.max_gpu_memory_gb:.1f}GB limit per GPU")
        print(f"   • Parallel Videos: {config.parallel_videos}")
        print(f"   • Temp Directory: {config.cache_dir}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        if config.powersafe:
            logger.info("PowerSafe: Progress has been saved and can be resumed")
        print("\nProcess interrupted. PowerSafe progress has been saved." if config.powersafe else "\nProcess interrupted.")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Production system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        if config.powersafe:
            logger.info("PowerSafe: Partial progress has been saved")
            print(f"\nError occurred: {e}")
            print("PowerSafe: Partial progress has been saved and can be resumed with --powersafe flag")
        
        # Provide debugging suggestions
        print(f"\nDEBUGGING SUGGESTIONS:")
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
                        
 