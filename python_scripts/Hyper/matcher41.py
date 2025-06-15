#!/usr/bin/env python3
"""
Production-Ready Multi-GPU Video-GPX Correlation Script

Key Features:
- Multi-GPU processing with proper queuing
- Pre-flight video validation to detect corrupted files
- Enhanced video preprocessing with GPU acceleration
- PowerSafe mode for long-running operations
- Robust error handling and recovery
- Memory optimization and cleanup
- Uses ~/penis/temp for temporary storage

Usage:
    # Basic usage with validation
    python production_matcher.py -d /path/to/data --gpu_ids 0 1
    
    # Just validate videos (recommended for new datasets)
    python production_matcher.py -d /path/to/data --validation_only
    
    # PowerSafe mode for long runs
    python production_matcher.py -d /path/to/data --powersafe --debug
    
    # Skip validation (not recommended for untrusted sources)
    python production_matcher.py -d /path/to/data --skip_validation
    
    # Keep corrupted videos in place instead of quarantining
    python production_matcher.py -d /path/to/data --no_quarantine
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

# GPU-Optimized Chunked Processing Imports
import threading
import queue
from collections import deque
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
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Enhanced configuration for chunked GPU processing"""
    max_frames: int = 999999  # Unlimited frames with chunking
    target_size: Tuple[int, int] = (3840, 2160)  # 4K default
    sample_rate: float = 3.0
    parallel_videos: int = 2  # Optimized for dual GPU
    gpu_memory_fraction: float = 0.95
    motion_threshold: float = 0.01
    temporal_window: int = 10
    powersafe: bool = False
    save_interval: int = 5
    gpu_timeout: int = 60
    strict: bool = False
    strict_fail: bool = False
    memory_efficient: bool = True
    max_gpu_memory_gb: float = 15.0  # Per GPU
    enable_preprocessing: bool = True
    ram_cache_gb: float = 100.0  # Use up to 100GB RAM
    disk_cache_gb: float = 1000.0
    cache_dir: str = "~/penis/temp"
    replace_originals: bool = False
    skip_validation: bool = False
    no_quarantine: bool = False
    validation_only: bool = False
    max_ram_usage_gb: float = 100.0
    gpu_memory_safety_margin: float = 0.95
    enable_ram_fallback: bool = True
    dynamic_resolution_scaling: bool = False  # Disabled - use chunking instead
    
    # Chunked processing parameters
    enable_chunked_processing: bool = True
    chunk_frames: int = 60
    chunk_overlap: int = 5
    max_chunk_memory_gb: float = 4.0
    
class MemoryAwareProcessor:
    """Memory-aware processing that can fall back to RAM when GPU memory is insufficient"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ram_threshold_gb = 100.0  # Use up to 100GB of RAM if needed
        self.gpu_memory_threshold = 0.8  # Use 80% of GPU memory as threshold
        
    def get_available_memory(self, gpu_id: int) -> Dict[str, float]:
        """Get available memory info for both GPU and RAM"""
        memory_info = {
            'gpu_total_gb': 0,
            'gpu_available_gb': 0,
            'ram_total_gb': 0,
            'ram_available_gb': 0,
            'can_use_gpu': False,
            'can_use_ram': False
        }
        
        try:
            # GPU memory
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                total_gpu_memory = gpu_props.total_memory / (1024**3)
                
                with torch.cuda.device(gpu_id):
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    available_gpu = total_gpu_memory - reserved
                    
                memory_info.update({
                    'gpu_total_gb': total_gpu_memory,
                    'gpu_available_gb': available_gpu,
                    'can_use_gpu': available_gpu > 2.0  # Need at least 2GB free
                })
            
            # RAM memory
            ram_info = psutil.virtual_memory()
            total_ram = ram_info.total / (1024**3)
            available_ram = ram_info.available / (1024**3)
            
            memory_info.update({
                'ram_total_gb': total_ram,
                'ram_available_gb': available_ram,
                'can_use_ram': available_ram > 10.0  # Need at least 10GB free
            })
            
        except Exception as e:
            logger.warning(f"Memory assessment failed: {e}")
            
        return memory_info
    
    def estimate_memory_requirement(self, width: int, height: int, num_frames: int) -> float:
        """Estimate memory requirement in GB for video processing"""
        # Rough estimation: width * height * 3 channels * num_frames * 4 bytes (float32)
        bytes_required = width * height * 3 * num_frames * 4
        gb_required = bytes_required / (1024**3)
        
        # Add overhead (feature extraction, gradients, etc.)
        return gb_required * 3.0
    
    def get_optimal_processing_params(self, width: int, height: int, num_frames: int, gpu_id: int) -> Dict:
        """Get optimal processing parameters based on available memory"""
        memory_info = self.get_available_memory(gpu_id)
        estimated_requirement = self.estimate_memory_requirement(width, height, num_frames)
        
        params = {
            'use_gpu': False,
            'use_ram': False,
            'batch_size': 1,
            'target_width': width,
            'target_height': height,
            'target_frames': num_frames,
            'processing_mode': 'failed'
        }
        
        # Try GPU processing first
        if memory_info['can_use_gpu'] and estimated_requirement < memory_info['gpu_available_gb']:
            params.update({
                'use_gpu': True,
                'processing_mode': 'gpu_full',
                'batch_size': max(1, int(memory_info['gpu_available_gb'] / estimated_requirement))
            })
            return params
        
        # Try reduced resolution on GPU
        scale_factors = [0.75, 0.5, 0.25]
        for scale in scale_factors:
            scaled_width = int(width * scale)
            scaled_height = int(height * scale)
            scaled_requirement = self.estimate_memory_requirement(scaled_width, scaled_height, num_frames)
            
            if memory_info['can_use_gpu'] and scaled_requirement < memory_info['gpu_available_gb']:
                params.update({
                    'use_gpu': True,
                    'processing_mode': f'gpu_scaled_{scale}',
                    'target_width': scaled_width,
                    'target_height': scaled_height,
                    'batch_size': 1
                })
                logger.warning(f"Scaling down to {scaled_width}x{scaled_height} due to GPU memory constraints")
                return params
        
        # Fall back to RAM processing
        if memory_info['can_use_ram'] and estimated_requirement < memory_info['ram_available_gb']:
            params.update({
                'use_ram': True,
                'processing_mode': 'ram_full',
                'batch_size': 1
            })
            logger.info(f"Using RAM processing due to insufficient GPU memory")
            return params
        
        # Try reduced frames
        for frame_reduction in [0.75, 0.5, 0.25]:
            reduced_frames = int(num_frames * frame_reduction)
            reduced_requirement = self.estimate_memory_requirement(width, height, reduced_frames)
            
            if memory_info['can_use_ram'] and reduced_requirement < memory_info['ram_available_gb']:
                params.update({
                    'use_ram': True,
                    'processing_mode': f'ram_reduced_frames_{frame_reduction}',
                    'target_frames': reduced_frames,
                    'batch_size': 1
                })
                logger.warning(f"Reducing frames to {reduced_frames} due to memory constraints")
                return params
        
        # Last resort: very small processing
        params.update({
            'use_ram': True,
            'processing_mode': 'ram_minimal',
            'target_width': min(width, 480),
            'target_height': min(height, 270),
            'target_frames': min(num_frames, 50),
            'batch_size': 1
        })
        logger.warning(f"Using minimal processing parameters due to severe memory constraints")
        
        return params

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

class MemoryAwareProcessor:
    """Memory-aware processing that can fall back to RAM when GPU memory is insufficient"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ram_threshold_gb = 100.0  # Use up to 100GB of RAM if needed
        self.gpu_memory_threshold = 0.8  # Use 80% of GPU memory as threshold
        
    def get_available_memory(self, gpu_id: int) -> Dict[str, float]:
        """Get available memory info for both GPU and RAM"""
        memory_info = {
            'gpu_total_gb': 0,
            'gpu_available_gb': 0,
            'ram_total_gb': 0,
            'ram_available_gb': 0,
            'can_use_gpu': False,
            'can_use_ram': False
        }
        
        try:
            # GPU memory
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                total_gpu_memory = gpu_props.total_memory / (1024**3)
                
                with torch.cuda.device(gpu_id):
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    available_gpu = total_gpu_memory - reserved
                    
                memory_info.update({
                    'gpu_total_gb': total_gpu_memory,
                    'gpu_available_gb': available_gpu,
                    'can_use_gpu': available_gpu > 2.0  # Need at least 2GB free
                })
            
            # RAM memory
            ram_info = psutil.virtual_memory()
            total_ram = ram_info.total / (1024**3)
            available_ram = ram_info.available / (1024**3)
            
            memory_info.update({
                'ram_total_gb': total_ram,
                'ram_available_gb': available_ram,
                'can_use_ram': available_ram > 10.0  # Need at least 10GB free
            })
            
        except Exception as e:
            logger.warning(f"Memory assessment failed: {e}")
            
        return memory_info
    
    def estimate_memory_requirement(self, width: int, height: int, num_frames: int) -> float:
        """Estimate memory requirement in GB for video processing"""
        # Rough estimation: width * height * 3 channels * num_frames * 4 bytes (float32)
        bytes_required = width * height * 3 * num_frames * 4
        gb_required = bytes_required / (1024**3)
        
        # Add overhead (feature extraction, gradients, etc.)
        return gb_required * 3.0
    
    def get_optimal_processing_params(self, width: int, height: int, num_frames: int, gpu_id: int) -> Dict:
        """Get optimal processing parameters based on available memory"""
        memory_info = self.get_available_memory(gpu_id)
        estimated_requirement = self.estimate_memory_requirement(width, height, num_frames)
        
        params = {
            'use_gpu': False,
            'use_ram': False,
            'batch_size': 1,
            'target_width': width,
            'target_height': height,
            'target_frames': num_frames,
            'processing_mode': 'failed'
        }
        
        # Try GPU processing first
        if memory_info['can_use_gpu'] and estimated_requirement < memory_info['gpu_available_gb']:
            params.update({
                'use_gpu': True,
                'processing_mode': 'gpu_full',
                'batch_size': max(1, int(memory_info['gpu_available_gb'] / estimated_requirement))
            })
            return params
        
        # Try reduced resolution on GPU
        scale_factors = [0.75, 0.5, 0.25]
        for scale in scale_factors:
            scaled_width = int(width * scale)
            scaled_height = int(height * scale)
            scaled_requirement = self.estimate_memory_requirement(scaled_width, scaled_height, num_frames)
            
            if memory_info['can_use_gpu'] and scaled_requirement < memory_info['gpu_available_gb']:
                params.update({
                    'use_gpu': True,
                    'processing_mode': f'gpu_scaled_{scale}',
                    'target_width': scaled_width,
                    'target_height': scaled_height,
                    'batch_size': 1
                })
                logger.warning(f"Scaling down to {scaled_width}x{scaled_height} due to GPU memory constraints")
                return params
        
        # Fall back to RAM processing
        if memory_info['can_use_ram'] and estimated_requirement < memory_info['ram_available_gb']:
            params.update({
                'use_ram': True,
                'processing_mode': 'ram_full',
                'batch_size': 1
            })
            logger.info(f"Using RAM processing due to insufficient GPU memory")
            return params
        
        # Try reduced frames
        for frame_reduction in [0.75, 0.5, 0.25]:
            reduced_frames = int(num_frames * frame_reduction)
            reduced_requirement = self.estimate_memory_requirement(width, height, reduced_frames)
            
            if memory_info['can_use_ram'] and reduced_requirement < memory_info['ram_available_gb']:
                params.update({
                    'use_ram': True,
                    'processing_mode': f'ram_reduced_frames_{frame_reduction}',
                    'target_frames': reduced_frames,
                    'batch_size': 1
                })
                logger.warning(f"Reducing frames to {reduced_frames} due to memory constraints")
                return params
        
        # Last resort: very small processing
        params.update({
            'use_ram': True,
            'processing_mode': 'ram_minimal',
            'target_width': min(width, 480),
            'target_height': min(height, 270),
            'target_frames': min(num_frames, 50),
            'batch_size': 1
        })
        logger.warning(f"Using minimal processing parameters due to severe memory constraints")
        
        return params

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
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            return {}

class VideoValidator:
    """Advanced video validation system with GPU compatibility testing and strict mode enforcement"""
    
    def __init__(self, config):
        self.config = config
        self.validation_results = {}
        
        # Create quarantine directory for corrupted files
        self.quarantine_dir = Path(os.path.expanduser(config.cache_dir)) / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for GPU testing
        self.temp_test_dir = Path(os.path.expanduser(config.cache_dir)) / "gpu_test"
        self.temp_test_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU-friendly formats and codecs
        #self.gpu_friendly_codecs = {'h264', 'avc1', 'mp4v', 'mpeg4'}
        self.gpu_problematic_codecs = {'hevc', 'h265', 'vp9', 'av1', 'vp8'}
        #self.convertible_codecs = {'hevc', 'h265', 'vp9', 'av1', 'vp8', 'xvid', 'divx'}
        self.unconvertible_codecs = {'prores', 'dnxhd', 'cineform'}
        self.gpu_friendly_codecs = {'h264', 'avc1', 'mp4v', 'mpeg4', 'hevc', 'h265'}  # Added HEVC
        self.convertible_codecs = {'vp9', 'av1', 'vp8', 'xvid', 'divx', 'mjpeg'}  # Added MJPEG

        
        
        logger.info(f"Enhanced Video Validator initialized:")
        logger.info(f"  Strict Mode: {config.strict or config.strict_fail}")
        logger.info(f"  Quarantine Directory: {self.quarantine_dir}")
        logger.info(f"  GPU Test Directory: {self.temp_test_dir}")
        if config.strict or config.strict_fail:
            logger.info(f"  GPU Compatibility Testing: Enabled")
            logger.info(f"  Conversion Feasibility Testing: Enabled")
    
    def validate_video_batch(self, video_files, quarantine_corrupted=True):
        """Validate a batch of video files with enhanced GPU compatibility testing"""
        logger.info(f"Pre-flight validation of {len(video_files)} videos...")
        if self.config.strict or self.config.strict_fail:
            logger.info("STRICT MODE: Testing GPU compatibility and conversion feasibility...")
        
        valid_videos = []
        corrupted_videos = []
        validation_details = {}
        
        # Progress bar for validation
        try:
            from tqdm import tqdm
            pbar = tqdm(video_files, desc="Validating videos", unit="video")
        except ImportError:
            pbar = video_files
        
        for video_path in pbar:
            try:
                if hasattr(pbar, 'set_postfix_str'):
                    pbar.set_postfix_str(f"Checking {Path(video_path).name[:30]}...")
                
                validation_result = self.validate_single_video(video_path)
                validation_details[video_path] = validation_result
                
                if validation_result['is_valid']:
                    valid_videos.append(video_path)
                    if hasattr(pbar, 'set_postfix_str'):
                        compatibility = validation_result.get('gpu_compatibility', 'unknown')
                        emoji = self._get_compatibility_emoji(compatibility)
                        pbar.set_postfix_str(f"{emoji} {Path(video_path).name[:25]}")
                else:
                    corrupted_videos.append(video_path)
                    if hasattr(pbar, 'set_postfix_str'):
                        pbar.set_postfix_str(f"❌ {Path(video_path).name[:25]}")
                    
                    # Handle corrupted/rejected video
                    if quarantine_corrupted and not validation_result.get('strict_rejected', False):
                        self.quarantine_video(video_path, validation_result['error'])
                    elif validation_result.get('strict_rejected', False):
                        logger.info(f"STRICT MODE: Rejected {Path(video_path).name} - {validation_result['error']}")
                        
            except Exception as e:
                logger.error(f"Error validating {video_path}: {e}")
                corrupted_videos.append(video_path)
                validation_details[video_path] = {
                    'is_valid': False, 
                    'error': str(e),
                    'validation_stage': 'exception'
                }
        
        # Print enhanced validation summary
        self.print_enhanced_validation_summary(valid_videos, corrupted_videos, validation_details)
        
        return valid_videos, corrupted_videos, validation_details
    
    def validate_single_video(self, video_path):
        """Enhanced single video validation with GPU compatibility and strict mode testing"""
        validation_result = {
            'is_valid': False,
            'error': None,
            'file_size_mb': 0,
            'duration': 0,
            'codec': None,
            'resolution': None,
            'issues': [],
            'gpu_compatibility': 'unknown',
            'conversion_feasible': None,
            'strict_rejected': False,
            'validation_stage': 'init'
        }
        
        try:
            # Stage 1: Basic file validation
            validation_result['validation_stage'] = 'basic_checks'
            
            if not os.path.exists(video_path):
                validation_result['error'] = "File does not exist"
                return validation_result
            
            file_size = os.path.getsize(video_path)
            validation_result['file_size_mb'] = file_size / (1024 * 1024)
            
            # Check if file is too small
            if file_size < 1024:
                validation_result['error'] = f"File too small: {file_size} bytes"
                return validation_result
            
            # Stage 2: Header validation
            validation_result['validation_stage'] = 'header_check'
            header_valid = self.check_video_header(video_path)
            if not header_valid:
                validation_result['issues'].append("unrecognized_header")
                # Continue to ffprobe - header check is just a hint
            
            # Stage 3: FFprobe validation
            validation_result['validation_stage'] = 'ffprobe_validation'
            probe_result = self.ffprobe_validation(video_path)
            if not probe_result['success']:
                validation_result['error'] = probe_result['error']
                return validation_result
            
            # Update with probe data
            validation_result.update(probe_result['data'])
            
            # Stage 4: GPU compatibility assessment
            validation_result['validation_stage'] = 'gpu_compatibility'
            gpu_compat = self.assess_gpu_compatibility(validation_result)
            validation_result['gpu_compatibility'] = gpu_compat
            
            # Stage 5: Strict mode validation
            validation_result['validation_stage'] = 'strict_mode_check'
            if self.config.strict or self.config.strict_fail:
                strict_valid = self.strict_mode_validation(video_path, validation_result)
                if not strict_valid:
                    validation_result['strict_rejected'] = True
                    # Error message already set by strict_mode_validation
                    return validation_result
            
            # Stage 6: Final validation
            validation_result['validation_stage'] = 'completed'
            validation_result['is_valid'] = True
            
            # Clear header error if we got this far
            if "unrecognized_header" in validation_result['issues'] and not validation_result['error']:
                validation_result['issues'].remove("unrecognized_header")
            
            return validation_result
            
        except Exception as e:
            validation_result['error'] = f"Validation exception at {validation_result['validation_stage']}: {str(e)}"
            return validation_result
    
    def check_video_header(self, video_path):
        """Comprehensive video file header check supporting all major formats"""
        try:
            with open(video_path, 'rb') as f:
                header = f.read(128)  # Read more bytes for better detection
            
            if len(header) < 8:
                return False
            
            # MP4/MOV/3GP files - look for ftyp box
            if len(header) >= 8:
                # Check for ftyp box at various positions
                for offset in range(0, min(64, len(header) - 8), 4):
                    if header[offset:offset+4] == b'ftyp':
                        return True
                    # Also check for the complete box structure
                    if offset + 8 <= len(header) and header[offset+4:offset+8] == b'ftyp':
                        return True
            
            # Comprehensive video format signatures
            video_signatures = [
                # MP4 variants with different box sizes
                b'\x00\x00\x00\x14ftyp',  # 20-byte ftyp
                b'\x00\x00\x00\x18ftyp',  # 24-byte ftyp
                b'\x00\x00\x00\x1cftyp',  # 28-byte ftyp
                b'\x00\x00\x00 ftyp',     # 32-byte ftyp
                b'\x00\x00\x00$ftyp',     # 36-byte ftyp
                b'\x00\x00\x00(ftyp',     # 40-byte ftyp
                b'\x00\x00\x00,ftyp',     # 44-byte ftyp
                
                # MKV/WebM (EBML)
                b'\x1a\x45\xdf\xa3',      # EBML header
                
                # AVI (RIFF container)
                b'RIFF',                   # RIFF header
                
                # MPEG variants
                b'\x00\x00\x01\xba',      # MPEG-PS
                b'\x00\x00\x01\xb3',      # MPEG video sequence
                b'\x00\x00\x01\x00',      # MPEG video
                
                # FLV
                b'FLV\x01',               # FLV header
                
                # Other formats
                b'\x30\x26\xb2\x75',      # ASF/WMV
                b'OggS',                   # OGG
            ]
            
            # Check all signatures
            for signature in video_signatures:
                if header.startswith(signature):
                    return True
                # Also check within first 64 bytes
                if signature in header[:64]:
                    return True
            
            # Special checks
            # AVI files
            if header.startswith(b'RIFF') and b'AVI ' in header[:32]:
                return True
            
            # QuickTime MOV
            if any(box in header[:64] for box in [b'moov', b'mdat', b'wide', b'free']):
                return True
            
            # 3GP files
            if b'3gp' in header[:32] or b'3g2' in header[:32]:
                return True
            
            # Extension-based fallback for edge cases
            video_extensions = {
                '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', 
                '.m4v', '.3gp', '.3g2', '.wmv', '.asf', '.ogv',
                '.mpg', '.mpeg', '.m2v', '.vob', '.ts', '.mts'
            }
            file_ext = os.path.splitext(video_path)[1].lower()
            if file_ext in video_extensions:
                return True
            
            return False
            
        except Exception:
            return False
    
    def ffprobe_validation(self, video_path):
        """Enhanced FFprobe validation with detailed codec and format analysis"""
        result = {
            'success': False,
            'error': None,
            'data': {}
        }
        
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,codec_long_name,profile,width,height,duration,pix_fmt,bit_rate',
                '-show_entries', 'format=format_name,duration,bit_rate,size',
                '-of', 'json', video_path
            ]
            
            proc_result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if proc_result.returncode != 0:
                error_output = proc_result.stderr.strip()
                
                # Enhanced error categorization
                error_lower = error_output.lower()
                if 'moov atom not found' in error_lower:
                    result['error'] = "Corrupted MP4: Missing metadata (moov atom)"
                elif 'invalid data found' in error_lower:
                    result['error'] = "Corrupted: Invalid video data"
                elif 'no such file' in error_lower:
                    result['error'] = "File not found"
                elif 'permission denied' in error_lower:
                    result['error'] = "Permission denied"
                elif 'protocol not found' in error_lower:
                    result['error'] = "Unsupported file format or protocol"
                elif 'invalid argument' in error_lower:
                    result['error'] = "Invalid file format or corrupted"
                elif 'end of file' in error_lower:
                    result['error'] = "Truncated or incomplete file"
                elif len(error_output) == 0:
                    result['error'] = "FFprobe failed with no error message"
                else:
                    result['error'] = f"FFprobe error: {error_output[:300]}"
                
                return result
            
            # Parse JSON output
            try:
                probe_data = json.loads(proc_result.stdout)
            except json.JSONDecodeError as e:
                result['error'] = f"Invalid FFprobe JSON output: {str(e)}"
                return result
            
            # Extract video stream info
            streams = probe_data.get('streams', [])
            if not streams:
                result['error'] = "No video streams found"
                return result
            
            video_stream = streams[0]
            format_info = probe_data.get('format', {})
            
            # Extract comprehensive video information
            codec_name = video_stream.get('codec_name', 'unknown').lower()
            duration = self._extract_duration(video_stream, format_info)
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            result['data'] = {
                'codec': codec_name,
                'codec_long_name': video_stream.get('codec_long_name', ''),
                'profile': video_stream.get('profile', ''),
                'width': width,
                'height': height,
                'duration': duration,
                'pixel_format': video_stream.get('pix_fmt', ''),
                'format_name': format_info.get('format_name', ''),
                'file_size': int(format_info.get('size', 0)),
                'bit_rate': self._extract_bit_rate(video_stream, format_info)
            }
            
            # Validation checks
            if width <= 0 or height <= 0:
                result['error'] = f"Invalid video dimensions: {width}x{height}"
                return result
            
            if width > 7680 or height > 4320:  # 8K limit
                result['error'] = f"Video resolution too high: {width}x{height} (max 8K supported)"
                return result
            
            if duration <= 0:
                logger.warning(f"No duration information for {Path(video_path).name}")
                # Not a fatal error for some formats
            
            if duration > 7200:  # 2 hours
                logger.warning(f"Very long video: {duration/60:.1f} minutes - {Path(video_path).name}")
            
            # Add resolution tuple
            result['data']['resolution'] = (width, height)
            
            result['success'] = True
            return result
            
        except subprocess.TimeoutExpired:
            result['error'] = "FFprobe timeout (file may be corrupted or very large)"
            return result
        except FileNotFoundError:
            result['error'] = "FFprobe not found - please install ffmpeg"
            return result
        except Exception as e:
            result['error'] = f"FFprobe validation failed: {str(e)}"
            return result
    
    def assess_gpu_compatibility(self, validation_result):
        """Assess GPU processing compatibility based on codec and format"""
        codec = validation_result.get('codec', '').lower()
        width = validation_result.get('width', 0)
        height = validation_result.get('height', 0)
        pixel_format = validation_result.get('pixel_format', '').lower()
        
        # GPU-friendly codecs
        if codec in self.gpu_friendly_codecs:
            if width <= 1920 and height <= 1080:
                return 'excellent'
            elif width <= 3840 and height <= 2160:
                return 'good'
            else:
                return 'fair'
        
        # Problematic but convertible codecs
        elif codec in self.gpu_problematic_codecs:
            if '10bit' in pixel_format or '10le' in pixel_format:
                return 'poor'  # 10-bit is harder for GPU
            elif width <= 1920 and height <= 1080:
                return 'fair'
            else:
                return 'poor'
        
        # Convertible codecs
        elif codec in self.convertible_codecs:
            return 'fair'
        
        # Unconvertible or unknown
        elif codec in self.unconvertible_codecs:
            return 'incompatible'
        
        else:
            return 'unknown'
    
    def strict_mode_validation(self, video_path, validation_result):
        """Strict mode validation with GPU compatibility and conversion testing"""
        gpu_compatibility = validation_result.get('gpu_compatibility', 'unknown')
        codec = validation_result.get('codec', '').lower()
        width = validation_result.get('width', 0)
        height = validation_result.get('height', 0)
        
        # Ultra strict mode - very restrictive
        if self.config.strict_fail:
            if gpu_compatibility in ['poor', 'incompatible', 'unknown']:
                validation_result['error'] = f"ULTRA STRICT: Codec '{codec}' not suitable for GPU processing"
                return False
            
            if width > 3840 or height > 2160:
                validation_result['error'] = f"ULTRA STRICT: Resolution {width}x{height} too high for reliable GPU processing"
                return False
        
        # Regular strict mode - test actual GPU compatibility
        elif self.config.strict:
            if gpu_compatibility == 'incompatible':
                validation_result['error'] = f"STRICT: Codec '{codec}' cannot be processed or converted"
                return False
            
            # Test GPU decoding for problematic videos
            if gpu_compatibility in ['poor', 'unknown']:
                gpu_test_result = self.test_gpu_decoding(video_path, validation_result)
                if not gpu_test_result['success']:
                    # Try conversion feasibility
                    conversion_result = self.test_conversion_feasibility(video_path, validation_result)
                    validation_result['conversion_feasible'] = conversion_result['feasible']
                    
                    if not conversion_result['feasible']:
                        validation_result['error'] = f"STRICT: Cannot process '{codec}' and conversion failed: {conversion_result['error']}"
                        return False
                    else:
                        logger.info(f"STRICT: {Path(video_path).name} requires conversion but is feasible")
        
        return True
    
    def test_gpu_decoding(self, video_path, validation_result):
        """Test actual GPU decoding capability"""
        result = {'success': False, 'error': None}
        
        try:
            # Quick GPU decode test - extract first frame
            test_output = self.temp_test_dir / f"gpu_test_{int(time.time() * 1000000)}.jpg"
            
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda',
                '-i', video_path,
                '-vframes', '1',
                '-f', 'image2',
                str(test_output)
            ]
            
            proc_result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if proc_result.returncode == 0 and test_output.exists() and test_output.stat().st_size > 1024:
                result['success'] = True
            else:
                error_msg = proc_result.stderr.decode() if proc_result.stderr else "Unknown GPU decode error"
                result['error'] = error_msg
            
            # Cleanup
            if test_output.exists():
                test_output.unlink()
                
        except subprocess.TimeoutExpired:
            result['error'] = "GPU decode test timeout"
        except Exception as e:
            result['error'] = f"GPU decode test failed: {str(e)}"
        
        return result
    
    def test_conversion_feasibility(self, video_path, validation_result):
        """Test if video can be converted to GPU-friendly format"""
        result = {'feasible': False, 'error': None}
        
        try:
            # Quick conversion test - convert 5 seconds to H264
            test_output = self.temp_test_dir / f"convert_test_{int(time.time() * 1000000)}.mp4"
            
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-i', video_path,
                '-t', '5',  # Only 5 seconds
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '30',
                '-an',  # No audio
                str(test_output)
            ]
            
            proc_result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if proc_result.returncode == 0 and test_output.exists() and test_output.stat().st_size > 1024:
                result['feasible'] = True
            else:
                error_msg = proc_result.stderr.decode() if proc_result.stderr else "Unknown conversion error"
                result['error'] = error_msg
            
            # Cleanup
            if test_output.exists():
                test_output.unlink()
                
        except subprocess.TimeoutExpired:
            result['error'] = "Conversion test timeout"
        except Exception as e:
            result['error'] = f"Conversion test failed: {str(e)}"
        
        return result
    
    def _extract_duration(self, video_stream, format_info):
        """Extract duration from multiple possible sources"""
        duration = 0.0
        
        if video_stream.get('duration'):
            try:
                duration = float(video_stream['duration'])
            except (ValueError, TypeError):
                pass
        
        if duration <= 0 and format_info.get('duration'):
            try:
                duration = float(format_info['duration'])
            except (ValueError, TypeError):
                pass
        
        return duration
    
    def _extract_bit_rate(self, video_stream, format_info):
        """Extract bit rate from multiple possible sources"""
        bit_rate = 0
        
        if video_stream.get('bit_rate'):
            try:
                bit_rate = int(video_stream['bit_rate'])
            except (ValueError, TypeError):
                pass
        
        if bit_rate <= 0 and format_info.get('bit_rate'):
            try:
                bit_rate = int(format_info['bit_rate'])
            except (ValueError, TypeError):
                pass
        
        return bit_rate
    
    def _get_compatibility_emoji(self, compatibility):
        """Get emoji for GPU compatibility level"""
        emoji_map = {
            'excellent': '🟢',
            'good': '🟡', 
            'fair': '🟠',
            'poor': '🔴',
            'incompatible': '❌',
            'unknown': '⚪'
        }
        return emoji_map.get(compatibility, '⚪')
    
    def quarantine_video(self, video_path, error_reason):
        """Move corrupted video to quarantine directory with enhanced info"""
        try:
            video_name = Path(video_path).name
            quarantine_path = self.quarantine_dir / video_name
            
            # If file exists, add timestamp
            if quarantine_path.exists():
                timestamp = int(time.time())
                stem = Path(video_path).stem
                suffix = Path(video_path).suffix
                quarantine_path = self.quarantine_dir / f"{stem}_{timestamp}{suffix}"
            
            # Move file
            shutil.move(video_path, quarantine_path)
            
            # Create detailed info file
            info_path = quarantine_path.with_suffix('.txt')
            with open(info_path, 'w') as f:
                f.write(f"Quarantined: {datetime.now().isoformat()}\n")
                f.write(f"Original path: {video_path}\n")
                f.write(f"Error reason: {error_reason}\n")
                f.write(f"Strict mode: {self.config.strict or self.config.strict_fail}\n")
                f.write(f"Validator version: Enhanced VideoValidator v2.0\n")
            
            logger.info(f"Quarantined video: {video_name}")
            
        except Exception as e:
            logger.error(f"Failed to quarantine {video_path}: {e}")
    
    def print_enhanced_validation_summary(self, valid_videos, corrupted_videos, validation_details):
        """Print enhanced validation summary with GPU compatibility stats"""
        total_videos = len(valid_videos) + len(corrupted_videos)
        
        # Analyze valid videos by GPU compatibility
        gpu_stats = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'incompatible': 0, 'unknown': 0}
        strict_rejected = 0
        
        for video_path, details in validation_details.items():
            if details.get('is_valid'):
                compatibility = details.get('gpu_compatibility', 'unknown')
                gpu_stats[compatibility] = gpu_stats.get(compatibility, 0) + 1
            elif details.get('strict_rejected'):
                strict_rejected += 1
        
        print(f"\n{'='*90}")
        print(f"ENHANCED VIDEO VALIDATION SUMMARY")
        print(f"{'='*90}")
        print(f"Total Videos Checked: {total_videos}")
        print(f"Valid Videos: {len(valid_videos)} ({100*len(valid_videos)/max(total_videos,1):.1f}%)")
        print(f"Corrupted/Rejected Videos: {len(corrupted_videos)} ({100*len(corrupted_videos)/max(total_videos,1):.1f}%)")
        
        if self.config.strict or self.config.strict_fail:
            mode_name = "ULTRA STRICT" if self.config.strict_fail else "STRICT"
            print(f"  - {mode_name} Mode Rejected: {strict_rejected}")
            print(f"  - Actually Corrupted: {len(corrupted_videos) - strict_rejected}")
        
        # GPU Compatibility breakdown
        if valid_videos:
            print(f"\nGPU COMPATIBILITY BREAKDOWN:")
            print(f"  🟢 Excellent (GPU-optimal): {gpu_stats['excellent']} ({100*gpu_stats['excellent']/len(valid_videos):.1f}%)")
            print(f"  🟡 Good (GPU-friendly): {gpu_stats['good']} ({100*gpu_stats['good']/len(valid_videos):.1f}%)")
            print(f"  🟠 Fair (Convertible): {gpu_stats['fair']} ({100*gpu_stats['fair']/len(valid_videos):.1f}%)")
            print(f"  🔴 Poor (Problematic): {gpu_stats['poor']} ({100*gpu_stats['poor']/len(valid_videos):.1f}%)")
            print(f"  ❌ Incompatible: {gpu_stats['incompatible']}")
            print(f"  ⚪ Unknown: {gpu_stats['unknown']}")
        
        # Show problematic videos
        if corrupted_videos:
            print(f"\nPROBLEMATIC VIDEOS:")
            shown = 0
            for video_path in corrupted_videos:
                if shown >= 15:  # Limit display
                    remaining = len(corrupted_videos) - shown
                    print(f"  ... and {remaining} more (check quarantine directory)")
                    break
                
                video_name = Path(video_path).name
                details = validation_details.get(video_path, {})
                error = details.get('error', 'Unknown error')
                file_size = details.get('file_size_mb', 0)
                stage = details.get('validation_stage', 'unknown')
                
                if details.get('strict_rejected'):
                    print(f"  🚫 {video_name[:45]:<45} | {file_size:6.1f}MB | STRICT: {error}")
                else:
                    print(f"  ❌ {video_name[:45]:<45} | {file_size:6.1f}MB | {error}")
                shown += 1
            
            if not self.config.no_quarantine and any(not d.get('strict_rejected') for d in validation_details.values() if not d.get('is_valid')):
                print(f"\nCorrupted videos quarantined to: {self.quarantine_dir}")
        
        print(f"{'='*90}")
    
    def get_validation_report(self, validation_details):
        """Generate comprehensive validation report"""
        valid_count = sum(1 for v in validation_details.values() if v['is_valid'])
        corrupted_count = len(validation_details) - valid_count
        strict_rejected_count = sum(1 for v in validation_details.values() if v.get('strict_rejected'))
        
        # GPU compatibility stats
        gpu_stats = {}
        codec_stats = {}
        
        for details in validation_details.values():
            if details.get('is_valid'):
                compatibility = details.get('gpu_compatibility', 'unknown')
                gpu_stats[compatibility] = gpu_stats.get(compatibility, 0) + 1
                
                codec = details.get('codec', 'unknown')
                codec_stats[codec] = codec_stats.get(codec, 0) + 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'validator_version': 'Enhanced VideoValidator v2.0',
            'strict_mode': self.config.strict or self.config.strict_fail,
            'ultra_strict_mode': self.config.strict_fail,
            'summary': {
                'total_videos': len(validation_details),
                'valid_videos': valid_count,
                'corrupted_videos': corrupted_count,
                'strict_rejected': strict_rejected_count,
                'actually_corrupted': corrupted_count - strict_rejected_count,
                'validation_success_rate': valid_count / max(len(validation_details), 1)
            },
            'gpu_compatibility_stats': gpu_stats,
            'codec_distribution': codec_stats,
            'details': validation_details,
            'quarantine_directory': str(self.quarantine_dir),
            'temp_test_directory': str(self.temp_test_dir)
        }
    
    def cleanup(self):
        """Cleanup temporary test files"""
        try:
            if self.temp_test_dir.exists():
                for temp_file in self.temp_test_dir.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                    except:
                        pass
            logger.info("Video validator cleanup completed")
        except Exception as e:
            logger.warning(f"Video validator cleanup failed: {e}")

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
            
            # Add diagnostics for the problematic video
            self._diagnose_video_issues(video_path, gpu_id)
            
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
    
    def _diagnose_video_issues(self, video_path: str, gpu_id: int):
        """Diagnose potential video issues"""
        try:
            # Check file size and basic properties
            file_size = os.path.getsize(video_path)
            logger.debug(f"Video file size: {file_size / (1024*1024):.1f} MB")
            
            # Check video properties with ffprobe
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_format', '-show_streams', video_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                video_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
                if video_stream:
                    logger.debug(f"Video codec: {video_stream.get('codec_name')}")
                    logger.debug(f"Video resolution: {video_stream.get('width')}x{video_stream.get('height')}")
                    logger.debug(f"Pixel format: {video_stream.get('pix_fmt')}")
                    
                    # Check for problematic formats
                    problematic_codecs = ['hevc', 'vp9', 'av1']
                    if video_stream.get('codec_name') in problematic_codecs:
                        logger.warning(f"Problematic codec detected: {video_stream.get('codec_name')}")
                        
                    if video_stream.get('pix_fmt') not in ['yuv420p', 'yuv444p', 'yuv422p']:
                        logger.warning(f"Unusual pixel format: {video_stream.get('pix_fmt')}")
            else:
                logger.warning(f"ffprobe failed for {Path(video_path).name}")
                
        except Exception as e:
            logger.debug(f"Video diagnosis failed: {e}")
    
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
                # Log more details for debugging
                if "CUDA" in error_msg or "nvidia" in error_msg.lower():
                    logger.warning(f"GPU-related error in {strategy_name}: {error_msg[:300]}")
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
    """Enhanced FFmpeg decoder with RAM fallback and memory-aware processing"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.temp_dirs = {}
        self.memory_processor = MemoryAwareProcessor(config)
        
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
    
    def decode_video_enhanced(self, video_path: str, gpu_id: int) -> Tuple[Optional[torch.Tensor], float, float]:
        """Enhanced video decoding with memory-aware processing and RAM fallback"""
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
            
            # Get optimal processing parameters
            optimal_params = self.memory_processor.get_optimal_processing_params(
                video_info['width'], video_info['height'], 
                min(self.config.max_frames, int(video_info['duration'] * video_info['fps'])), 
                gpu_id
            )
            
            logger.info(f"Processing {Path(video_path).name} with mode: {optimal_params['processing_mode']}")
            
            # Decode frames based on optimal parameters
            if optimal_params['use_gpu']:
                frames_tensor = self._decode_gpu_frames(actual_video_path, video_info, gpu_id, optimal_params)
            elif optimal_params['use_ram']:
                frames_tensor = self._decode_ram_frames(actual_video_path, video_info, gpu_id, optimal_params)
            else:
                raise RuntimeError("No suitable processing method available")
            
            if frames_tensor is None:
                raise RuntimeError("Frame decoding failed")
            
            return frames_tensor, video_info['fps'], video_info['duration']
            
        except Exception as e:
            logger.error(f"Enhanced video decoding failed for {video_path}: {e}")
            return None, 0, 0
    
    def _decode_gpu_frames(self, video_path: str, video_info: Dict, gpu_id: int, params: Dict) -> Optional[torch.Tensor]:
        """Decode frames directly to GPU with memory management"""
        try:
            temp_dir = self.temp_dirs[gpu_id]
            output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
            
            # Use optimal parameters
            target_width = params['target_width']
            target_height = params['target_height']
            target_frames = params['target_frames']
            
            # Ensure even dimensions
            if target_width % 2 != 0:
                target_width += 1
            if target_height % 2 != 0:
                target_height += 1
            
            # Calculate sampling
            total_frames = int(video_info['duration'] * video_info['fps'])
            if total_frames > target_frames:
                sample_rate = total_frames / target_frames
                vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,select=not(mod(n\\,{int(sample_rate)}))'
            else:
                vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2'
            
            # CUDA command with memory management
            cuda_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                '-i', video_path,
                '-vf', vf_filter,
                '-frames:v', str(min(target_frames, total_frames)),
                '-q:v', '2',
                '-threads', '1',
                output_pattern
            ]
            
            # Execute command
            result = subprocess.run(cuda_cmd, check=True, capture_output=True, timeout=300)
            
            # Load frames to GPU tensor with batching
            return self._load_frames_to_gpu_batched(temp_dir, gpu_id, target_width, target_height, params['batch_size'])
            
        except Exception as e:
            logger.error(f"GPU frame decoding failed: {e}")
            return None
    
    def _decode_ram_frames(self, video_path: str, video_info: Dict, gpu_id: int, params: Dict) -> Optional[torch.Tensor]:
        """Decode frames to RAM then transfer to GPU in batches"""
        try:
            temp_dir = self.temp_dirs[gpu_id]
            output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
            
            # Use optimal parameters
            target_width = params['target_width']
            target_height = params['target_height']
            target_frames = params['target_frames']
            
            # Ensure even dimensions
            if target_width % 2 != 0:
                target_width += 1
            if target_height % 2 != 0:
                target_height += 1
            
            # Calculate sampling
            total_frames = int(video_info['duration'] * video_info['fps'])
            if total_frames > target_frames:
                sample_rate = total_frames / target_frames
                vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,select=not(mod(n\\,{int(sample_rate)}))'
            else:
                vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2'
            
            # CPU command for RAM processing
            cpu_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-i', video_path,
                '-vf', vf_filter,
                '-frames:v', str(min(target_frames, total_frames)),
                '-q:v', '3',
                '-threads', '4',
                output_pattern
            ]
            
            # Execute command
            result = subprocess.run(cpu_cmd, check=True, capture_output=True, timeout=600)
            
            # Load frames to RAM first, then transfer to GPU
            return self._load_frames_ram_to_gpu(temp_dir, gpu_id, target_width, target_height)
            
        except Exception as e:
            logger.error(f"RAM frame decoding failed: {e}")
            return None
    
    def _load_frames_to_gpu_batched(self, temp_dir: str, gpu_id: int, target_width: int, target_height: int, batch_size: int) -> Optional[torch.Tensor]:
        """Load frames to GPU with batching to prevent OOM"""
        frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.jpg')))
        
        if not frame_files:
            return None
        
        device = torch.device(f'cuda:{gpu_id}')
        all_frames = []
        
        # Process in batches
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i + batch_size]
            batch_frames = []
            
            for frame_file in batch_files:
                try:
                    img = cv2.imread(frame_file)
                    if img is None:
                        continue
                    
                    # Verify dimensions
                    if img.shape[1] != target_width or img.shape[0] != target_height:
                        img = cv2.resize(img, (target_width, target_height))
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    batch_frames.append(img_tensor)
                    
                    os.remove(frame_file)
                    
                except Exception as e:
                    logger.debug(f"Failed to load frame {frame_file}: {e}")
                    continue
            
            if batch_frames:
                # Stack batch and move to GPU
                batch_tensor = torch.stack(batch_frames).permute(0, 3, 1, 2)
                try:
                    batch_tensor = batch_tensor.to(device)
                    all_frames.append(batch_tensor)
                except torch.cuda.OutOfMemoryError:
                    # If GPU OOM, keep in RAM and transfer later
                    logger.warning("GPU OOM during batch transfer, keeping in RAM")
                    all_frames.append(batch_tensor)
                
                # Clear batch from memory
                del batch_frames, batch_tensor
                torch.cuda.empty_cache()
                gc.collect()
        
        if not all_frames:
            return None
        
        # Concatenate all frames
        try:
            # Try to concatenate on GPU
            frames_tensor = torch.cat(all_frames, dim=0).unsqueeze(0)
            if frames_tensor.device != device:
                frames_tensor = frames_tensor.to(device)
        except torch.cuda.OutOfMemoryError:
            # Concatenate on CPU then move to GPU
            logger.warning("GPU OOM during concatenation, using CPU")
            cpu_frames = [f.cpu() if f.device.type == 'cuda' else f for f in all_frames]
            frames_tensor = torch.cat(cpu_frames, dim=0).unsqueeze(0)
            try:
                frames_tensor = frames_tensor.to(device)
            except torch.cuda.OutOfMemoryError:
                logger.error("Cannot fit final tensor on GPU, keeping on CPU")
                # This will cause issues downstream but better than crashing
        
        logger.debug(f"Loaded {len(frame_files)} frames to GPU {gpu_id}: {frames_tensor.shape}")
        return frames_tensor
    
    def _load_frames_ram_to_gpu(self, temp_dir: str, gpu_id: int, target_width: int, target_height: int) -> Optional[torch.Tensor]:
        """Load frames to RAM first, then carefully transfer to GPU"""
        frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.jpg')))
        
        if not frame_files:
            return None
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Load all frames to RAM first
        ram_frames = []
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
                ram_frames.append(img_tensor)
                
                os.remove(frame_file)
                
            except Exception as e:
                logger.debug(f"Failed to load frame {frame_file}: {e}")
                continue
        
        if not ram_frames:
            return None
        
        # Stack frames in RAM
        frames_tensor = torch.stack(ram_frames).permute(0, 3, 1, 2).unsqueeze(0)
        
        # Try to move to GPU
        try:
            frames_tensor = frames_tensor.to(device)
            logger.debug(f"Successfully moved {len(ram_frames)} frames from RAM to GPU {gpu_id}")
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Cannot move frames to GPU {gpu_id}, keeping in RAM")
            # Keep on CPU - this will work but be slower
        
        return frames_tensor

class EnhancedFeatureExtractor:
    """Enhanced feature extraction with RAM fallback and memory optimization"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.feature_models = {}
        self.memory_processor = MemoryAwareProcessor(config)
        
        # Initialize models for each GPU
        for gpu_id in gpu_manager.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            self.feature_models[gpu_id] = self._create_enhanced_model().to(device)
        
        logger.info("Enhanced feature extractor initialized with memory management")
    
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
        """Extract features with RAM fallback when GPU memory is insufficient"""
        try:
            device = torch.device(f'cuda:{gpu_id}')
            
            # Check if tensor is on GPU and fits in memory
            memory_info = self.memory_processor.get_available_memory(gpu_id)
            tensor_size_gb = frames_tensor.numel() * 4 / (1024**3)  # Assuming float32
            
            if not memory_info['can_use_gpu'] or tensor_size_gb > memory_info['gpu_available_gb'] * 0.5:
                logger.warning(f"Using RAM-based feature extraction due to memory constraints")
                return self._extract_features_cpu_optimized(frames_tensor, gpu_id)
            
            # Try GPU processing with memory management
            try:
                if frames_tensor.device != device:
                    frames_tensor = frames_tensor.to(device)
                
                return self._extract_features_gpu_batched(frames_tensor, gpu_id)
                
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU OOM during feature extraction, falling back to CPU")
                torch.cuda.empty_cache()
                return self._extract_features_cpu_optimized(frames_tensor, gpu_id)
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction failed: {e}")
            raise
    
    def _extract_features_gpu_batched(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract features on GPU with batching"""
        device = torch.device(f'cuda:{gpu_id}')
        model = self.feature_models[gpu_id]
        
        batch_size, num_frames = frames_tensor.shape[:2]
        
        # Determine optimal batch size for processing
        memory_info = self.memory_processor.get_available_memory(gpu_id)
        max_batch_frames = max(1, int(memory_info['gpu_available_gb'] * 200))  # Rough estimate
        
        features = {}
        
        with torch.no_grad():
            if num_frames <= max_batch_frames:
                # Process all frames at once
                frames_flat = frames_tensor.view(-1, *frames_tensor.shape[2:])
                cnn_features = model(frames_flat)
                
                for key, value in cnn_features.items():
                    value = value.view(batch_size, num_frames, -1)[0]
                    features[key] = value.cpu().numpy()
            else:
                # Process in smaller batches
                logger.info(f"Processing {num_frames} frames in batches of {max_batch_frames}")
                all_features = defaultdict(list)
                
                for start_idx in range(0, num_frames, max_batch_frames):
                    end_idx = min(start_idx + max_batch_frames, num_frames)
                    batch_frames = frames_tensor[0, start_idx:end_idx]
                    
                    cnn_features = model(batch_frames)
                    
                    for key, value in cnn_features.items():
                        all_features[key].append(value.cpu().numpy())
                    
                    # Clear GPU memory
                    del batch_frames, cnn_features
                    torch.cuda.empty_cache()
                
                # Concatenate batch results
                for key, value_list in all_features.items():
                    features[key] = np.concatenate(value_list, axis=0)
            
            # Enhanced motion features
            motion_features = self._compute_enhanced_motion(frames_tensor[0], device)
            features.update(motion_features)
            
            # Color features
            color_features = self._compute_enhanced_color(frames_tensor[0], device)
            features.update(color_features)
            
            # Edge features
            edge_features = self._compute_edge_features(frames_tensor[0], device)
            features.update(edge_features)
        
        logger.debug(f"GPU feature extraction successful: {len(features)} feature types")
        return features
    
    def _extract_features_cpu_optimized(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract features using CPU with GPU model"""
        try:
            # Move tensor to CPU if needed
            if frames_tensor.device.type == 'cuda':
                frames_tensor = frames_tensor.cpu()
            
            # Create CPU version of model
            device = torch.device('cpu')
            cpu_model = self._create_enhanced_model().to(device)
            cpu_model.eval()
            
            batch_size, num_frames = frames_tensor.shape[:2]
            features = {}
            
            with torch.no_grad():
                # Process in smaller batches to manage memory
                batch_size_cpu = 10  # Smaller batches for CPU
                all_features = defaultdict(list)
                
                for start_idx in range(0, num_frames, batch_size_cpu):
                    end_idx = min(start_idx + batch_size_cpu, num_frames)
                    batch_frames = frames_tensor[0, start_idx:end_idx]
                    
                    cnn_features = cpu_model(batch_frames)
                    
                    for key, value in cnn_features.items():
                        all_features[key].append(value.numpy())
                    
                    # Clear memory
                    del batch_frames, cnn_features
                    gc.collect()
                
                # Concatenate batch results
                for key, value_list in all_features.items():
                    features[key] = np.concatenate(value_list, axis=0)
                
                # Enhanced motion features (CPU version)
                motion_features = self._compute_enhanced_motion_cpu(frames_tensor[0])
                features.update(motion_features)
                
                # Color features (CPU version)
                color_features = self._compute_enhanced_color_cpu(frames_tensor[0])
                features.update(color_features)
                
                # Edge features (CPU version)
                edge_features = self._compute_edge_features_cpu(frames_tensor[0])
                features.update(edge_features)
            
            logger.debug(f"CPU feature extraction successful: {len(features)} feature types")
            return features
            
        except Exception as e:
            logger.error(f"CPU feature extraction failed: {e}")
            raise
    
    def _compute_enhanced_motion_cpu(self, frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """CPU version of enhanced motion computation"""
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
    
    def _compute_enhanced_color_cpu(self, frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """CPU version of enhanced color computation"""
        num_frames = frames.shape[0]
        
        # Color variance over time
        color_variance = torch.var(frames, dim=[2, 3])
        mean_color_variance = torch.mean(color_variance, dim=1).numpy()
        
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
    
    def _compute_edge_features_cpu(self, frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """CPU version of edge feature computation"""
        # Use numpy for CPU edge detection
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray_np = gray_frames.numpy()
        
        edge_density = []
        for frame in gray_np:
            # Simple gradient-based edge detection
            grad_x = np.abs(np.diff(frame, axis=1))
            grad_y = np.abs(np.diff(frame, axis=0))
            
            # Pad to match original size
            grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='constant')
            grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='constant')
            
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_density.append(np.mean(edge_magnitude))
        
        return {
            'edge_density': np.array(edge_density)
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

class ImprovedSimilarityEngine:
    """Improved similarity computation with better correlation methods"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        # Rebalanced weights for better accuracy
        self.weights = {
            'motion_dynamics': 0.50,     # Increased - most important
            'temporal_correlation': 0.25, # Reduced
            'statistical_profile': 0.25   # Reduced
        }
        
        # Improved similarity thresholds
        self.quality_thresholds = {
            'excellent': 0.85,  # Raised threshold
            'good': 0.70,       # Raised threshold
            'fair': 0.50,       # Raised threshold
            'poor': 0.30        # Raised threshold
        }
    
    def compute_enhanced_similarity(self, video_features: Dict, gpx_features: Dict) -> Dict[str, float]:
        """Compute enhanced similarity with improved methods"""
        try:
            similarities = {}
            
            # Enhanced motion dynamics similarity
            similarities['motion_dynamics'] = self._compute_improved_motion_similarity(video_features, gpx_features)
            
            # Improved temporal correlation
            similarities['temporal_correlation'] = self._compute_improved_temporal_similarity(video_features, gpx_features)
            
            # Enhanced statistical profile matching
            similarities['statistical_profile'] = self._compute_improved_statistical_similarity(video_features, gpx_features)
            
            # Apply non-linear weighting to emphasize strong correlations
            weighted_scores = []
            for key, score in similarities.items():
                if key in self.weights:
                    # Apply sigmoid transformation to emphasize strong correlations
                    enhanced_score = self._sigmoid_transform(score)
                    weighted_scores.append(enhanced_score * self.weights[key])
            
            combined_score = sum(weighted_scores)
            similarities['combined'] = float(np.clip(combined_score, 0.0, 1.0))
            similarities['quality'] = self._assess_improved_quality(similarities['combined'])
            
            return similarities
            
        except Exception as e:
            logger.error(f"Enhanced similarity computation failed: {e}")
            return self._create_zero_similarity()
    
    def _compute_improved_motion_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Improved motion similarity with better alignment"""
        try:
            # Extract motion signatures with multiple approaches
            video_signatures = self._extract_multiple_motion_signatures(video_features, 'video')
            gpx_signatures = self._extract_multiple_motion_signatures(gpx_features, 'gpx')
            
            if not video_signatures or not gpx_signatures:
                return 0.0
            
            # Try multiple correlation methods and take the best
            best_similarity = 0.0
            
            for v_sig_name, v_sig in video_signatures.items():
                for g_sig_name, g_sig in gpx_signatures.items():
                    if v_sig is None or g_sig is None or len(v_sig) < 5 or len(g_sig) < 5:
                        continue
                    
                    # Method 1: DTW similarity
                    dtw_sim = self._compute_dtw_similarity(v_sig, g_sig)
                    
                    # Method 2: Cross-correlation
                    xcorr_sim = self._compute_cross_correlation_similarity(v_sig, g_sig)
                    
                    # Method 3: Normalized correlation
                    norm_corr_sim = self._compute_normalized_correlation(v_sig, g_sig)
                    
                    # Take the best similarity from the three methods
                    method_best = max(dtw_sim, xcorr_sim, norm_corr_sim)
                    best_similarity = max(best_similarity, method_best)
            
            return float(np.clip(best_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Improved motion similarity computation failed: {e}")
            return 0.0
    
    def _extract_multiple_motion_signatures(self, features: Dict, source_type: str) -> Dict[str, Optional[np.ndarray]]:
        """Extract multiple types of motion signatures"""
        signatures = {}
        
        try:
            if source_type == 'video':
                # Primary motion signature
                if 'motion_magnitude' in features:
                    signatures['magnitude'] = self._robust_normalize(features['motion_magnitude'])
                
                # Acceleration signature
                if 'acceleration' in features:
                    signatures['acceleration'] = self._robust_normalize(features['acceleration'])
                
                # Scene change signature (derived from motion)
                if 'motion_magnitude' in features:
                    motion = features['motion_magnitude']
                    if len(motion) > 1:
                        scene_changes = np.diff(motion)
                        signatures['scene_changes'] = self._robust_normalize(scene_changes)
                        
            elif source_type == 'gpx':
                # Primary speed signature
                if 'speed' in features:
                    signatures['speed'] = self._robust_normalize(features['speed'])
                
                # Acceleration signature
                if 'acceleration' in features:
                    signatures['acceleration'] = self._robust_normalize(features['acceleration'])
                
                # Speed change signature
                if 'speed' in features:
                    speed = features['speed']
                    if len(speed) > 1:
                        speed_changes = np.diff(speed)
                        signatures['speed_changes'] = self._robust_normalize(speed_changes)
        
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
        
        return signatures
    
    def _compute_dtw_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute DTW-based similarity"""
        try:
            if FASTDTW_AVAILABLE and len(sig1) > 10 and len(sig2) > 10:
                # Use FastDTW for longer sequences
                distance, _ = fastdtw(sig1.reshape(-1, 1), sig2.reshape(-1, 1))
                max_len = max(len(sig1), len(sig2))
                normalized_distance = distance / max_len
                similarity = 1.0 / (1.0 + normalized_distance)
                return similarity
            else:
                # Simple DTW for shorter sequences
                return self._simple_dtw_similarity(sig1, sig2)
        except Exception:
            return 0.0
    
    def _simple_dtw_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Simple DTW implementation for shorter sequences"""
        try:
            n, m = len(sig1), len(sig2)
            if n == 0 or m == 0:
                return 0.0
            
            # Create cost matrix
            cost = np.full((n + 1, m + 1), np.inf)
            cost[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    dist = abs(sig1[i-1] - sig2[j-1])
                    cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            
            # Normalize by path length
            normalized_cost = cost[n, m] / (n + m)
            similarity = 1.0 / (1.0 + normalized_cost)
            
            return similarity
        except Exception:
            return 0.0
    
    def _compute_cross_correlation_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute cross-correlation based similarity"""
        try:
            if len(sig1) < 3 or len(sig2) < 3:
                return 0.0
            
            # Normalize signals
            sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
            sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)
            
            # Compute cross-correlation
            correlation = np.correlate(sig1_norm, sig2_norm, mode='full')
            max_corr = np.max(np.abs(correlation))
            
            # Normalize by signal lengths
            normalization = np.sqrt(len(sig1) * len(sig2))
            normalized_corr = max_corr / normalization
            
            return float(np.clip(normalized_corr, 0.0, 1.0))
        except Exception:
            return 0.0
    
    def _compute_normalized_correlation(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute normalized correlation"""
        try:
            # Align lengths
            min_len = min(len(sig1), len(sig2))
            if min_len < 3:
                return 0.0
            
            s1 = sig1[:min_len]
            s2 = sig2[:min_len]
            
            # Compute Pearson correlation
            correlation = np.corrcoef(s1, s2)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            # Convert to similarity (take absolute value and normalize)
            similarity = abs(correlation)
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception:
            return 0.0
    
    def _sigmoid_transform(self, score: float, steepness: float = 10.0) -> float:
        """Apply sigmoid transformation to emphasize strong correlations"""
        try:
            # Sigmoid function that emphasizes scores above 0.5
            transformed = 1.0 / (1.0 + np.exp(-steepness * (score - 0.5)))
            return float(np.clip(transformed, 0.0, 1.0))
        except Exception:
            return score
    
    def _assess_improved_quality(self, score: float) -> str:
        """Assess similarity quality with improved thresholds"""
        if score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif score >= self.quality_thresholds['good']:
            return 'good'
        elif score >= self.quality_thresholds['fair']:
            return 'fair'
        elif score >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'


# Add this to the main() function to use the improved components
def update_main_function_usage():
    """
    In your main() function, replace these instantiations:
    
    OLD:
    similarity_engine = ImprovedSimilarityEngine(config)
    
    NEW:
    similarity_engine = ImprovedSimilarityEngine(config)
    
    The decoder and feature extractor will automatically use the new memory management.
    """
    pass
    
    
def process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
    """GPU-Optimized chunked video processing with unlimited resolution/frames"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    try:
        # Use chunked processing for unlimited resolution/frames
        if getattr(config, 'enable_chunked_processing', True):
            try:
                # Import chunked processor
                from gpu_optimized_chunked_processor import ChunkedVideoProcessor
                
                processor = ChunkedVideoProcessor(gpu_manager, config)
                features = processor.process_video_chunked(video_path)
                
                if features is not None:
                    features['processing_mode'] = 'GPU_CHUNKED_OPTIMIZED'
                    
                    if powersafe_manager:
                        powersafe_manager.mark_video_features_done(video_path)
                    
                    logger.info(f"🚀 Chunked processing successful: {Path(video_path).name}")
                    return video_path, features
                else:
                    logger.warning(f"Chunked processing failed, trying fallback: {Path(video_path).name}")
                    
            except ImportError as e:
                logger.error(f"❌ ChunkedVideoProcessor not available: {e}")
                logger.error("Please ensure gpu_optimized_chunked_processor.py exists")
                return video_path, None
            except Exception as e:
                logger.warning(f"Chunked processing error, trying fallback: {e}")
        
        # Fallback to original processing if chunked fails
        logger.info(f"Using fallback processing: {Path(video_path).name}")
        return original_process_video_parallel_enhanced(args)
        
    except Exception as e:
        error_msg = f"All video processing methods failed: {str(e)}"
        logger.error(f"{error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        return video_path, None

def original_process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
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
            error_msg = f"Video decoding failed after all attempts for {Path(video_path).name}"
            
            # Handle different strict modes
            if config.strict_fail:
                # Ultra strict mode - fail entire process
                error_msg = f"ULTRA STRICT MODE: {error_msg}"
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                raise RuntimeError(error_msg)
            elif config.strict:
                # Regular strict mode - log error but continue
                logger.error(f"STRICT MODE: {error_msg}")
                logger.error(f"STRICT MODE: Skipping problematic video {Path(video_path).name} - check video integrity")
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, f"STRICT MODE: {error_msg}")
                return video_path, None
            else:
                # Normal mode - just continue
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
        
        # Handle different strict modes for exceptions
        if config.strict_fail:
            # Ultra strict mode - fail entire process
            error_msg = f"ULTRA STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {Path(video_path).name}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            raise RuntimeError(error_msg)
        elif config.strict:
            # Regular strict mode - log error but continue
            if "STRICT MODE" not in str(e):
                error_msg = f"STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {Path(video_path).name}")
            logger.error(f"STRICT MODE: Skipping failed video {Path(video_path).name}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            return video_path, None
        else:
            # Normal mode
            logger.error(f"{error_msg} for {Path(video_path).name}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
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


def monitor_chunked_performance():
    """Monitor chunked processing performance"""
    try:
        import torch
        import psutil
        
        # GPU stats
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                if allocated > 0:  # Only show active GPUs
                    logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        
        # RAM stats
        ram = psutil.virtual_memory()
        ram_used_gb = (ram.total - ram.available) / 1024**3
        ram_total_gb = ram.total / 1024**3
        
        if ram_used_gb > 50:  # Only log if significant RAM usage
            logger.info(f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent:.1f}%)")
            
    except Exception as e:
        logger.debug(f"Performance monitoring error: {e}")

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
    parser.add_argument("--max_frames", type=int, default=999999,
                       help="Maximum frames per video (default: 150)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[3840, 2160],
                       help="Target video resolution (default: 480 270)")
    parser.add_argument("--sample_rate", type=float, default=3.0,
                       help="Video sampling rate (default: 3.0)")
    parser.add_argument("--parallel_videos", type=int, default=2,
                       help="Number of videos to process in parallel (default: 1)")
    
    # GPU configuration
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1],
                       help="GPU IDs to use (default: [0, 1])")
    parser.add_argument("--gpu_timeout", type=int, default=60,
                       help="Seconds to wait for GPU availability (default: 60)")
    parser.add_argument("--max_gpu_memory", type=float, default=15.0,
                       help="Maximum GPU memory to use in GB (default: 12.0)")
    parser.add_argument("--memory_efficient", action='store_true', default=True,
                       help="Enable memory optimizations (default: True)")
    
    # Video preprocessing and caching
    parser.add_argument("--enable_preprocessing", action='store_true', default=True,
                       help="Enable GPU-based video preprocessing (default: True)")
    parser.add_argument("--ram_cache", type=float, default=100.0,
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
                       help="STRICT MODE: Enforce GPU usage, skip problematic videos")
    parser.add_argument("--strict_fail", action='store_true',
                       help="ULTRA STRICT MODE: Fail entire process if any video fails")
    
    # Power-safe mode arguments
    parser.add_argument("--powersafe", action='store_true',
                       help="Enable power-safe mode with incremental saves")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save results every N correlations in powersafe mode (default: 5)")
    
    # Video validation arguments
    parser.add_argument("--skip_validation", action='store_true',
                       help="Skip pre-flight video validation (not recommended)")
    parser.add_argument("--no_quarantine", action='store_true',
                       help="Don't quarantine corrupted videos, just skip them")
    parser.add_argument("--validation_only", action='store_true',
                       help="Only run video validation, don't process videos")
    
    args = parser.parse_args()
    
    # Update config to use correct temp directory
    args = update_config_for_temp_dir(args)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level, "production_correlation.log")
    
    if args.strict_fail:
        logger.info("Starting Production-Ready Video-GPX Correlation System [ULTRA STRICT GPU MODE]")
    elif args.strict:
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
            strict_fail=args.strict_fail,
            memory_efficient=args.memory_efficient,
            max_gpu_memory_gb=args.max_gpu_memory,
            enable_preprocessing=args.enable_preprocessing,
            ram_cache_gb=args.ram_cache,
            disk_cache_gb=args.disk_cache,
            cache_dir=args.cache_dir,
            skip_validation=args.skip_validation,
            no_quarantine=args.no_quarantine,
            validation_only=args.validation_only
        )
        
        # Validate strict mode requirements early
        if config.strict or config.strict_fail:
            mode_name = "ULTRA STRICT MODE" if config.strict_fail else "STRICT MODE"
            logger.info(f"{mode_name} ENABLED: GPU usage mandatory")
            if config.strict_fail:
                logger.info("ULTRA STRICT MODE: Process will fail if any video fails")
            else:
                logger.info("STRICT MODE: Problematic videos will be skipped")
                
            if not torch.cuda.is_available():
                raise RuntimeError(f"{mode_name}: CUDA is required but not available")
            if not cp.cuda.is_available():
                raise RuntimeError(f"{mode_name}: CuPy CUDA is required but not available")
        
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
        
        # PRE-FLIGHT VIDEO VALIDATION
        if not config.skip_validation:
            logger.info("🔍 Starting pre-flight video validation...")
            validator = VideoValidator(config)
            
            valid_videos, corrupted_videos, validation_details = validator.validate_video_batch(
                video_files, 
                quarantine_corrupted=not config.no_quarantine
            )
            
            # Save validation report
            validation_report = validator.get_validation_report(validation_details)
            validation_report_path = output_dir / "video_validation_report.json"
            with open(validation_report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"📋 Validation report saved: {validation_report_path}")
            
            # Update video_files to only include valid videos
            video_files = valid_videos
            
            if not video_files:
                print(f"\n❌ No valid videos found after validation!")
                print(f"   All {len(corrupted_videos)} videos were corrupted.")
                print(f"   Check the quarantine directory: {validator.quarantine_dir}")
                sys.exit(1)
            
            if config.validation_only:
                print(f"\n✅ Validation-only mode complete!")
                print(f"   Valid videos: {len(valid_videos)}")
                print(f"   Corrupted videos: {len(corrupted_videos)}")
                print(f"   Report saved: {validation_report_path}")
                sys.exit(0)
            
            logger.info(f"✅ Pre-flight validation complete: {len(valid_videos)} valid videos will be processed")
        else:
            logger.warning("⚠️ Skipping video validation - corrupted videos may cause failures")
        
        if not video_files:
            raise RuntimeError("No valid video files to process")
        
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
        similarity_engine = ImprovedSimilarityEngine(config)
        
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
            print(f"   • Check video file formats and corruption with --validation_only")
            print(f"   • Verify GPX files contain valid track data")
            print(f"   • Enable --debug for detailed error analysis")
            if not config.powersafe:
                print(f"   • Use --powersafe to preserve progress during debugging")
            if config.strict_fail:
                print(f"   • Remove --strict_fail flag to allow skipping problematic videos")
            elif config.strict:
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
        print(f"   • Run --validation_only to check for corrupted videos")
        print(f"   • Use --no_quarantine to keep corrupted videos in place")
        
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
          
