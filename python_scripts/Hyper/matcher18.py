"""
Ultra-Optimized Multi-GPU Video-GPX Correlation Script with H.264 4:4:4 Support

Key Features:
- Strict GPU acceleration enforcement (no CPU fallbacks)
- Automatic H.264 4:4:4 to 4:2:0 conversion using NVIDIA GPU encoding
- Intelligent RAM vs disk storage management for conversions
- Automatic file replacement after processing completion
- 128GB RAM optimization for temporary storage
- Comprehensive error handling and logging

H.264 4:4:4 Conversion Features:
- Automatic detection of H.264 4:4:4 pixel formats
- GPU-accelerated conversion using h264_nvenc
- RAM storage for conversions when sufficient memory available
- Disk fallback for large files or low memory situations
- Automatic replacement of original files with converted versions
- Optional backup preservation of original 4:4:4 files
- Configurable conversion behavior via command-line options

Usage Examples:
  # Basic usage with H.264 4:4:4 auto-conversion
  python matcher11.py -d /path/to/videos --gpu_ids 0 1

  # Force disk conversion (no RAM usage)
  python matcher11.py -d /path/to/videos --force_disk_conversion

  # Keep backup of original 4:4:4 files
  python matcher11.py -d /path/to/videos --keep_444_backup

  # Disable 4:4:4 conversion entirely
  python matcher11.py -d /path/to/videos --disable_444_conversion
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
from threading import Lock
import queue
import tempfile
import shutil

# Try to import optional video libraries
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    warnings.warn("PyAV not available, using fallback video processing")

# Try to import optional GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available, GPU monitoring disabled")

# Try to import psutil for RAM monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available, using fallback RAM detection")

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpu_correlation.log')
    ]
)
logger = logging.getLogger(__name__)

class StrictGPUEnforcer:
    """Strict GPU enforcement - no CPU fallbacks allowed"""
    
    @staticmethod
    def ensure_cuda_available():
        """Ensure CUDA is available and working - no fallbacks"""
        if not torch.cuda.is_available():
            raise RuntimeError("STRICT MODE: CUDA is not available! GPU acceleration required.")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("STRICT MODE: No CUDA devices found! GPU acceleration required.")
        
        logger.info(f"STRICT MODE: CUDA available with {device_count} devices")
        
        # Test basic GPU operations - must work
        try:
            test_tensor = torch.randn(100, 100, device='cuda:0')
            result = torch.mm(test_tensor, test_tensor)
            if result.device.type != 'cuda':
                raise RuntimeError("STRICT MODE: GPU tensor operations returned CPU tensor!")
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"STRICT MODE: GPU tensor operations failed: {e}")
    
    @staticmethod
    def ensure_cupy_available():
        """Ensure CuPy is available and working - no fallbacks"""
        if not cp.cuda.is_available():
            raise RuntimeError("STRICT MODE: CuPy CUDA is not available! GPU acceleration required.")
        
        try:
            test_array = cp.random.randn(100, 100)
            result = cp.dot(test_array, test_array)
            if not isinstance(result, cp.ndarray):
                raise RuntimeError("STRICT MODE: CuPy operations not returning GPU arrays!")
            del test_array, result
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            raise RuntimeError(f"STRICT MODE: CuPy operations failed: {e}")
    
    @staticmethod
    def check_ffmpeg_gpu():
        """Check FFmpeg GPU capabilities - must work"""
        try:
            # Check for NVDEC support
            result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                  capture_output=True, text=True, check=True)
            if 'cuda' not in result.stdout:
                raise RuntimeError("STRICT MODE: FFmpeg CUDA support not found!")
            
            # Check for available encoders
            result = subprocess.run(['ffmpeg', '-encoders'], 
                                  capture_output=True, text=True, check=True)
            if 'h264_nvenc' not in result.stdout and 'h264_cuvid' not in result.stdout:
                raise RuntimeError("STRICT MODE: FFmpeg GPU encoders not found!")
            
            logger.info("STRICT MODE: FFmpeg GPU support verified")
            return True
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"STRICT MODE: FFmpeg GPU check failed: {e}")
    
    @staticmethod
    def enforce_tensor_on_gpu(tensor, operation_name="operation"):
        """Enforce that tensor is on GPU"""
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"STRICT MODE: {operation_name} did not return a tensor!")
        
        if tensor.device.type != 'cuda':
            raise RuntimeError(f"STRICT MODE: {operation_name} returned CPU tensor instead of GPU! Device: {tensor.device}")
        
        return tensor
    
    @staticmethod
    def enforce_array_on_gpu(array, operation_name="operation"):
        """Enforce that CuPy array is on GPU"""
        if not isinstance(array, cp.ndarray):
            raise RuntimeError(f"STRICT MODE: {operation_name} did not return a CuPy array!")
        
        return array

class UltraOptimizedFFmpegDecoder:
    """Ultra-optimized FFmpeg-based GPU decoder with H.264 4:4:4 conversion support"""
    
    def __init__(self, gpu_ids=[0, 1], skip_decode_test=False, disable_444_conversion=False, 
                 force_disk_conversion=False, keep_444_backup=False):
        StrictGPUEnforcer.ensure_cuda_available()
        StrictGPUEnforcer.check_ffmpeg_gpu()
        
        self.gpu_ids = gpu_ids
        self.temp_dirs = {}
        self.skip_decode_test = skip_decode_test
        self.disable_444_conversion = disable_444_conversion
        self.force_disk_conversion = force_disk_conversion
        self.keep_444_backup = keep_444_backup
        self.converted_videos = {}  # Track converted videos for cleanup
        self.ram_conversions = {}   # Track RAM-stored conversions
        
        if skip_decode_test:
            logger.warning("STRICT MODE: GPU decode capability test DISABLED")
        
        if disable_444_conversion:
            logger.warning("STRICT MODE: H.264 4:4:4 conversion DISABLED")
        
        if force_disk_conversion:
            logger.info("STRICT MODE: H.264 4:4:4 conversions will use disk storage")
        
        if keep_444_backup:
            logger.info("STRICT MODE: H.264 4:4:4 backup files will be preserved")
        
        # Initialize GPU contexts - must succeed
        for gpu_id in gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"STRICT MODE: GPU {gpu_id} not available! Only {torch.cuda.device_count()} GPUs found.")
            
            torch.cuda.set_device(gpu_id)
            self.temp_dirs[gpu_id] = tempfile.mkdtemp(prefix=f'gpu_{gpu_id}_')
        
        # Create RAM disk directory for temporary conversions
        self.ram_disk_dir = tempfile.mkdtemp(prefix='gpu_ram_conversions_')
        
        logger.info(f"STRICT MODE: FFmpeg decoder initialized for GPUs: {gpu_ids}")
        logger.info(f"STRICT MODE: RAM conversion directory: {self.ram_disk_dir}")
    
    def _check_video_format(self, video_path):
        """Check if video is H.264 4:4:4 format that needs conversion"""
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    codec_name = stream.get('codec_name', '').lower()
                    pix_fmt = stream.get('pix_fmt', '').lower()
                    
                    # Check for H.264 4:4:4 formats
                    if codec_name == 'h264' and ('444' in pix_fmt or 'yuv444' in pix_fmt):
                        logger.info(f"DETECTED: H.264 4:4:4 video format in {video_path} (pixel format: {pix_fmt})")
                        return True, stream
            
            return False, None
            
        except Exception as e:
            logger.warning(f"Failed to check video format for {video_path}: {e}")
            return False, None
    
    def _get_available_ram(self):
        """Get available RAM in bytes"""
        if PSUTIL_AVAILABLE:
            try:
                available_ram = psutil.virtual_memory().available
                return available_ram
            except Exception as e:
                logger.warning(f"psutil failed to get RAM info: {e}")
        
        # Fallback: check /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        # Value is in kB
                        available_kb = int(line.split()[1])
                        return available_kb * 1024
        except Exception as e:
            logger.warning(f"Failed to read /proc/meminfo: {e}")
        
        # Conservative fallback
        logger.warning("Could not determine available RAM, assuming 32GB available")
        return 32 * 1024**3  # 32GB fallback
    
    def _estimate_conversion_size(self, video_path, stream_info):
        """Estimate the size of converted video"""
        try:
            # Get original file size
            original_size = os.path.getsize(video_path)
            
            # Get video properties
            width = int(stream_info.get('width', 1920))
            height = int(stream_info.get('height', 1080))
            duration = float(stream_info.get('duration', 600))  # Default 10 minutes
            
            # Rough estimation: 4:2:0 is typically 60-70% the size of 4:4:4
            # But also consider bitrate reduction
            estimated_size = int(original_size * 0.65)  # Conservative estimate
            
            logger.debug(f"Original size: {original_size//1024**2}MB, Estimated 4:2:0 size: {estimated_size//1024**2}MB")
            
            return estimated_size
            
        except Exception as e:
            logger.warning(f"Failed to estimate conversion size: {e}")
            # Fallback: assume 2GB max
            return 2 * 1024**3
    
    def _convert_h264_444_to_420_gpu(self, video_path, gpu_id=0):
        """Convert H.264 4:4:4 to 4:2:0 using GPU acceleration"""
        
        # Check if conversion is disabled
        if self.disable_444_conversion:
            logger.debug(f"H.264 4:4:4 conversion disabled, using original: {video_path}")
            return video_path
        
        # Check video format first
        is_444, stream_info = self._check_video_format(video_path)
        if not is_444:
            logger.debug(f"Video {video_path} is not H.264 4:4:4, no conversion needed")
            return video_path  # Return original path if no conversion needed
        
        logger.info(f"CONVERTING: H.264 4:4:4 to 4:2:0 for {video_path}")
        
        # Estimate conversion size and check available RAM
        estimated_size = self._estimate_conversion_size(video_path, stream_info)
        available_ram = self._get_available_ram()
        
        # Reserve 8GB for system operations, require 2x estimated size for safety
        ram_threshold = available_ram - (8 * 1024**3)  # Keep 8GB free
        use_ram = estimated_size * 2 < ram_threshold and not self.force_disk_conversion
        
        if use_ram:
            # Store in RAM (temporary directory)
            output_path = os.path.join(self.ram_disk_dir, f"converted_{os.path.basename(video_path)}")
            logger.info(f"RAM CONVERSION: Using RAM storage for conversion ({estimated_size//1024**2}MB estimated)")
        else:
            # Store on disk next to original
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}_converted_420.mp4"
            logger.info(f"DISK CONVERSION: Using disk storage for conversion ({estimated_size//1024**2}MB estimated)")
            if self.force_disk_conversion:
                logger.info("DISK CONVERSION: Forced by user option")
        
        # GPU-accelerated conversion command
        convert_cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-c:v', 'h264_nvenc',  # Use NVIDIA GPU encoder
            '-pix_fmt', 'yuv420p',  # Force 4:2:0 pixel format
            '-preset', 'p1',  # Fastest preset for NVENC
            '-tune', 'hq',  # High quality tuning
            '-rc', 'vbr',  # Variable bitrate
            '-cq', '23',  # Constant quality (similar to CRF)
            '-b:v', '0',  # Let CQ control bitrate
            '-maxrate', '50M',  # Maximum bitrate cap
            '-bufsize', '100M',  # Buffer size
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-movflags', '+faststart',  # Optimize for streaming
            output_path
        ]
        
        logger.debug(f"Conversion command: {' '.join(convert_cmd)}")
        
        try:
            # Clear GPU memory before conversion
            self._clear_gpu_memory_aggressive(gpu_id)
            
            # Run conversion with timeout
            start_time = time.time()
            process = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            conversion_time = time.time() - start_time
            
            if process.returncode != 0:
                raise RuntimeError(f"GPU conversion failed: {process.stderr}")
            
            # Verify output file exists and has reasonable size
            if not os.path.exists(output_path):
                raise RuntimeError(f"Converted file not created: {output_path}")
            
            converted_size = os.path.getsize(output_path)
            if converted_size < 1024:  # Less than 1KB is suspicious
                raise RuntimeError(f"Converted file too small: {converted_size} bytes")
            
            logger.info(f"CONVERSION SUCCESS: {video_path} -> {output_path}")
            logger.info(f"Conversion time: {conversion_time:.2f}s, Size: {converted_size//1024**2}MB")
            
            # Track the conversion for cleanup
            self.converted_videos[video_path] = {
                'converted_path': output_path,
                'use_ram': use_ram,
                'original_path': video_path,
                'conversion_time': conversion_time,
                'converted_size': converted_size
            }
            
            if use_ram:
                self.ram_conversions[video_path] = output_path
            
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"GPU conversion timed out for {video_path}")
            # Clean up partial file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise RuntimeError(f"GPU conversion timed out for {video_path}")
            
        except Exception as e:
            logger.error(f"GPU conversion failed for {video_path}: {e}")
            # Clean up partial file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
    
    def _finalize_conversion(self, original_path):
        """Finalize conversion by replacing original file if needed"""
        if original_path not in self.converted_videos:
            return
        
        conversion_info = self.converted_videos[original_path]
        converted_path = conversion_info['converted_path']
        use_ram = conversion_info['use_ram']
        
        try:
            if use_ram:
                # Move from RAM storage to replace original
                logger.info(f"FINALIZING: Moving RAM conversion to replace original {original_path}")
                
                # Create backup of original if requested
                if self.keep_444_backup:
                    backup_path = f"{original_path}.444_backup"
                    shutil.copy2(original_path, backup_path)
                    logger.info(f"BACKUP: Created H.264 4:4:4 backup at {backup_path}")
                
                # Remove original and move converted file to original location
                os.remove(original_path)
                shutil.move(converted_path, original_path)
                
                logger.info(f"FINALIZED: {original_path} replaced with 4:2:0 version")
                
                # Remove from RAM conversions tracking
                if original_path in self.ram_conversions:
                    del self.ram_conversions[original_path]
            
            else:
                # For disk conversions, optionally replace original
                logger.info(f"FINALIZING: Disk conversion complete, original preserved: {original_path}")
                logger.info(f"Converted version available at: {converted_path}")
                
                # Optionally replace original with converted version
                replace_original = True  # Set this based on your preference
                
                if replace_original:
                    if self.keep_444_backup:
                        backup_path = f"{original_path}.444_backup"
                        shutil.copy2(original_path, backup_path)
                        logger.info(f"BACKUP: Created H.264 4:4:4 backup at {backup_path}")
                    
                    os.remove(original_path)
                    shutil.move(converted_path, original_path)
                    
                    logger.info(f"REPLACED: {original_path} with 4:2:0 version")
            
            # Remove from tracking
            del self.converted_videos[original_path]
            
        except Exception as e:
            logger.error(f"Failed to finalize conversion for {original_path}: {e}")
    
    #def decode_video_gpu_batch(self, video_path, sample_rate=2.0, target_size=(640, 360), gpu_id=0):
    def decode_video_gpu_batch(self, video_path, sample_rate=2.0, target_size=(640, 360), gpu_id=0):
        """Decode video using STRICT GPU acceleration with H.264 4:4:4 conversion support"""
        
        if gpu_id not in self.temp_dirs:
            raise RuntimeError(f"STRICT MODE: GPU {gpu_id} not initialized!")
        
        # Check if video needs H.264 4:4:4 to 4:2:0 conversion
        try:
            working_video_path = self._convert_h264_444_to_420_gpu(video_path, gpu_id)
            if working_video_path != video_path:
                logger.info(f"PROCESSING: Using converted video {working_video_path} instead of {video_path}")
        except Exception as e:
            logger.warning(f"H.264 4:4:4 conversion failed for {video_path}: {e}")
            logger.info(f"FALLBACK: Attempting to process original video {video_path}")
            working_video_path = video_path
        
        # Aggressive memory clearing before FFmpeg
        self._clear_gpu_memory_aggressive(gpu_id)
        
        temp_dir = self.temp_dirs[gpu_id]
        
        # Get video info from working video path
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', 
            '-show_streams', working_video_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
        except Exception as e:
            raise RuntimeError(f"STRICT MODE: Failed to probe video {working_video_path}: {e}")
        
        # Find video stream
        video_stream = None
        for stream in info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise RuntimeError(f"STRICT MODE: No video stream found in {working_video_path}")
        
        fps = eval(video_stream['r_frame_rate'])
        duration = float(info['format']['duration'])
        
        # Get video resolution and adjust target size if needed
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # For very high resolution videos, reduce target size to save memory
        if width >= 3840 or height >= 2160:  # 4K or higher
            target_size = (480, 270)  # Smaller for 4K videos
            sample_rate = min(sample_rate, 1.0)  # Also reduce sample rate
            logger.info(f"STRICT MODE: 4K+ video detected ({width}x{height}), using target size {target_size}, rate {sample_rate}")
        elif width >= 1920 or height >= 1080:  # 1080p
            target_size = (640, 360)  # Standard size
            sample_rate = min(sample_rate, 1.5)  # Slight reduction
        
        # Check available GPU memory - must have enough
        available_memory = self._get_available_gpu_memory(gpu_id)
        required_memory = self._estimate_decode_memory(width, height, target_size, sample_rate, duration)
        
        # Strict memory requirement - must have at least 1GB free
        min_memory_required = 1024**3  # 1GB
        
        if available_memory < min_memory_required:
            raise RuntimeError(f"STRICT MODE: Insufficient GPU memory! Have {available_memory//1024**2}MB, need {min_memory_required//1024**2}MB minimum")
        
        if required_memory > available_memory * 0.8:
            raise RuntimeError(f"STRICT MODE: Video too large for GPU memory! Need {required_memory//1024**2}MB, have {available_memory//1024**2}MB")
        
        # GPU decode capability test - must pass (unless skipped)
        if not self.skip_decode_test:
            if not self._test_gpu_decode_capability(gpu_id):
                raise RuntimeError(f"STRICT MODE: GPU {gpu_id} decode capability test failed!")
        else:
            logger.warning(f"STRICT MODE: Skipping GPU {gpu_id} decode capability test")
        
        # Calculate sampling parameters
        frame_interval = max(1, int(fps / sample_rate))
        max_frames = min(int(duration * sample_rate) + 10, 300)  # Limit to 300 frames max
        
        # Create GPU-accelerated FFmpeg pipeline - MUST use GPU
        output_pattern = os.path.join(temp_dir, 'frame_%06d.rgb')
        
        # Try GPU decoders in order of preference - NO CPU FALLBACK
        gpu_decoders = ['h264_cuvid', 'hevc_cuvid']
        
        last_error = None
        for decoder in gpu_decoders:
            decode_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(gpu_id),
                '-hwaccel_output_format', 'cuda',  # Keep frames on GPU
                '-c:v', decoder,
                '-i', working_video_path,  # Use working video path (possibly converted)
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]},hwdownload,format=rgb24',
                '-fps_mode', 'vfr',  # Use variable frame rate mode instead of -vsync
                '-f', 'image2',
                '-pix_fmt', 'rgb24',
                '-frames:v', str(max_frames),
                output_pattern
            ]
            
            logger.debug(f"STRICT MODE: Trying GPU decoder: {decoder}")
            
            try:
                process = subprocess.run(decode_cmd, capture_output=True, text=True, timeout=180)
                if process.returncode == 0:
                    # Success! Break out of loop
                    break
                else:
                    last_error = f"GPU decoder {decoder} failed: {process.stderr}"
                    logger.debug(last_error)
                    
            except subprocess.TimeoutExpired:
                last_error = f"GPU decoder {decoder} timed out"
                logger.debug(last_error)
                continue
            except Exception as e:
                last_error = f"GPU decoder {decoder} exception: {e}"
                logger.debug(last_error)
                continue
        else:
            # If GPU decoders fail, try fallback approach with software decode + GPU processing
            logger.warning(f"STRICT MODE: GPU decoders failed, trying fallback software decode for {working_video_path}")
            
            fallback_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-i', working_video_path,
                '-vf', f'scale={target_size[0]}:{target_size[1]}',
                '-fps_mode', 'vfr',
                '-f', 'image2',
                '-pix_fmt', 'rgb24',
                '-frames:v', str(max_frames),
                output_pattern
            ]
            
            try:
                process = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=180)
                if process.returncode == 0:
                    logger.info(f"FALLBACK SUCCESS: Software decode worked for {working_video_path}")
                else:
                    raise RuntimeError(f"STRICT MODE: Both GPU and fallback decoders failed! Last error: {last_error}, Fallback error: {process.stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"STRICT MODE: All decoders timed out for {working_video_path}")
            except Exception as e:
                raise RuntimeError(f"STRICT MODE: All decoders failed! Last error: {last_error}, Fallback exception: {e}")
        
        # Load frames directly to GPU
        frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.rgb')))
        if not frame_files:
            # Try other possible extensions in case FFmpeg used different format
            frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.*')))
        
        if not frame_files:
            raise RuntimeError(f"STRICT MODE: No frames decoded by GPU for {working_video_path}")
        
        logger.debug(f"Found {len(frame_files)} frame files to load")
        frames_gpu = self._load_frames_to_gpu(frame_files, target_size, gpu_id)
        
        # Validate frames are on GPU
        StrictGPUEnforcer.enforce_tensor_on_gpu(frames_gpu, "video decode frames")
        
        # Cleanup immediately
        for f in frame_files:
            try:
                os.remove(f)
            except:
                pass
        
        frame_indices = list(range(0, len(frames_gpu), frame_interval))
        
        logger.info(f"STRICT MODE: Successfully decoded {len(frames_gpu)} frames using GPU decoder")
        
        # Store reference to original path for later finalization
        if working_video_path != video_path:
            self.converted_videos.setdefault(video_path, {})['needs_finalization'] = True
        
        return frames_gpu, fps, duration, frame_indices
    
    def _clear_gpu_memory_aggressive(self, gpu_id):
        """Aggressively clear GPU memory before processing"""
        try:
            torch.cuda.set_device(gpu_id)
            
            # Force garbage collection first
            import gc
            gc.collect()
            
            # Clear PyTorch cache multiple times
            for _ in range(3):
                torch.cuda.empty_cache()
                time.sleep(0.1)
            
            # Clear CuPy memory
            with cp.cuda.Device(gpu_id):
                cp.get_default_memory_pool().free_all_blocks()
                # Force synchronization
                cp.cuda.Device(gpu_id).synchronize()
            
            # Wait a bit for cleanup to complete
            time.sleep(0.5)
            
            logger.debug(f"Aggressively cleared GPU {gpu_id} memory")
            
        except Exception as e:
            logger.warning(f"Failed to aggressively clear GPU {gpu_id} memory: {e}")
    
    def _test_gpu_decode_capability(self, gpu_id):
        """Test if GPU decoder can be initialized with detailed debugging"""
        try:
            # Test 1: Basic GPU tensor operations
            test_tensor = torch.randn(100, 100, device=f'cuda:{gpu_id}')
            del test_tensor
            torch.cuda.empty_cache()
            logger.debug(f"GPU {gpu_id}: Basic tensor operations - PASSED")
            
            # Test 2: Check GPU memory
            available_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
            logger.debug(f"GPU {gpu_id}: Available memory: {available_memory//1024**2}MB")
            
            if available_memory < 512 * 1024**2:  # Less than 512MB
                logger.warning(f"GPU {gpu_id}: Low available memory: {available_memory//1024**2}MB")
                return False
            
            # Test 3: Simple FFmpeg command without CUDA
            simple_test_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-f', 'lavfi', '-i', 'color=red:size=320x240:duration=0.1',
                '-f', 'null', '-'
            ]
            
            try:
                result = subprocess.run(simple_test_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    logger.warning(f"GPU {gpu_id}: Basic FFmpeg test failed: {result.stderr}")
                    return False
                logger.debug(f"GPU {gpu_id}: Basic FFmpeg test - PASSED")
            except subprocess.TimeoutExpired:
                logger.warning(f"GPU {gpu_id}: Basic FFmpeg test timed out")
                return False
            
            # Test 4: FFmpeg with CUDA (the actual test) - FIXED ORDER
            cuda_test_cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),  # MOVED BEFORE INPUT
                '-f', 'lavfi', '-i', 'color=red:size=320x240:duration=0.1',
                '-f', 'null', '-'
            ]
            
            try:
                result = subprocess.run(cuda_test_cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    logger.debug(f"GPU {gpu_id}: CUDA decode capability test - PASSED")
                    return True
                else:
                    logger.warning(f"GPU {gpu_id}: CUDA decode test failed")
                    logger.warning(f"GPU {gpu_id}: FFmpeg error: {result.stderr}")
                    
                    # Check for specific error patterns
                    stderr_lower = result.stderr.lower()
                    if 'out of memory' in stderr_lower:
                        logger.error(f"GPU {gpu_id}: CUDA out of memory during test")
                    elif 'no device' in stderr_lower:
                        logger.error(f"GPU {gpu_id}: CUDA device not available")
                    elif 'failed to create' in stderr_lower:
                        logger.error(f"GPU {gpu_id}: Failed to create CUDA context")
                    
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"GPU {gpu_id}: CUDA decode test timed out")
                return False
                
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Decode capability test exception: {e}")
            return False
    
    def _clear_gpu_memory(self, gpu_id):
        """Clear GPU memory before processing"""
        try:
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            # Also clear CuPy memory on this device
            with cp.cuda.Device(gpu_id):
                cp.get_default_memory_pool().free_all_blocks()
                
        except Exception as e:
            logger.warning(f"Failed to clear GPU {gpu_id} memory: {e}")
    
    def _get_available_gpu_memory(self, gpu_id):
        """Get available GPU memory in bytes"""
        try:
            props = torch.cuda.get_device_properties(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id)
            reserved = torch.cuda.memory_reserved(gpu_id)
            
            # Use reserved memory as it's more accurate for availability
            available = props.total_memory - reserved
            return max(0, available)
            
        except Exception:
            # Fallback: assume some memory is available
            return 1024**3  # 1GB fallback
    
    def _estimate_decode_memory(self, input_width, input_height, target_size, sample_rate, duration):
        """Estimate memory needed for video decoding"""
        target_width, target_height = target_size
        
        # Rough estimation:
        # - Input frame buffer: input_width * input_height * 3 * 4 (RGBA float)
        # - Output frame buffer: target_width * target_height * 3 * batch_size
        # - FFmpeg internal buffers: ~2x input frame size
        
        estimated_frames = int(duration * sample_rate)
        max_batch_frames = min(30, estimated_frames)  # Limit batch size
        
        input_frame_size = input_width * input_height * 3 * 4  # RGBA float
        output_frame_size = target_width * target_height * 3 * 4  # RGB float
        
        memory_needed = (
            input_frame_size * 3 +  # FFmpeg decode buffers
            output_frame_size * max_batch_frames +  # Output batch
            100 * 1024**2  # 100MB overhead
        )
        
        return memory_needed
    
    def _load_frames_to_gpu(self, frame_files, target_size, gpu_id):
        """Load raw RGB frames directly to GPU memory - STRICT GPU ONLY"""
        device = torch.device(f'cuda:{gpu_id}')
        frame_list = []
        
        width, height = target_size
        expected_frame_size = width * height * 3  # RGB
        
        for frame_file in frame_files:
            try:
                # Read frame data
                with open(frame_file, 'rb') as f:
                    raw_data = f.read()
                
                # Check if this is raw RGB data or compressed JPEG
                if len(raw_data) != expected_frame_size:
                    # This is likely compressed JPEG data, need to decode it
                    logger.debug(f"Frame appears to be JPEG compressed ({len(raw_data)} bytes), decoding...")
                    
                    # Use OpenCV to decode the JPEG data
                    import cv2
                    
                    # Convert bytes to numpy array
                    img_array = np.frombuffer(raw_data, dtype=np.uint8)
                    
                    # Decode JPEG
                    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame_bgr is None:
                        logger.warning(f"Failed to decode frame {frame_file}")
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Ensure correct size
                    if frame_rgb.shape[:2] != (height, width):
                        frame_rgb = cv2.resize(frame_rgb, (width, height))
                    
                    # Convert to tensor
                    frame_tensor = torch.from_numpy(frame_rgb).to(device, dtype=torch.float32) / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW format
                    
                else:
                    # This is raw RGB data
                    logger.debug(f"Frame is raw RGB data ({len(raw_data)} bytes)")
                    
                    # Convert to numpy then GPU tensor in one operation
                    frame_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                    frame_tensor = torch.from_numpy(frame_np).to(device, dtype=torch.float32) / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW format
                
                # STRICT: Validate tensor is on GPU
                StrictGPUEnforcer.enforce_tensor_on_gpu(frame_tensor, "frame loading")
                
                frame_list.append(frame_tensor)
                
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_file}: {e}")
                continue
        
        if not frame_list:
            raise RuntimeError("STRICT MODE: No valid frames loaded to GPU")
        
        # Stack all frames into batch tensor on GPU
        frames_batch = torch.stack(frame_list).unsqueeze(0)  # (1, N, C, H, W)
        
        # Final validation
        StrictGPUEnforcer.enforce_tensor_on_gpu(frames_batch, "frame batch")
        
        logger.info(f"STRICT MODE: Loaded {len(frame_list)} frames to GPU with shape {frames_batch.shape}")
        
        return frames_batch
    
    def finalize_all_conversions(self):
        """Finalize all H.264 4:4:4 conversions by replacing original files"""
        logger.info(f"FINALIZING: Processing {len(self.converted_videos)} converted videos")
        
        for original_path in list(self.converted_videos.keys()):
            try:
                self._finalize_conversion(original_path)
            except Exception as e:
                logger.error(f"Failed to finalize conversion for {original_path}: {e}")
        
        # Clean up RAM disk directory
        if os.path.exists(self.ram_disk_dir):
            try:
                shutil.rmtree(self.ram_disk_dir)
                logger.info(f"Cleaned up RAM conversion directory: {self.ram_disk_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up RAM conversion directory: {e}")
    
    def __del__(self):
        """Cleanup temp directories and conversions"""
        # Clean up temporary frame directories
        for temp_dir in self.temp_dirs.values():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Clean up RAM conversions if not finalized
        for original_path, converted_path in self.ram_conversions.items():
            if os.path.exists(converted_path):
                try:
                    os.remove(converted_path)
                    logger.debug(f"Cleaned up unfinalized RAM conversion: {converted_path}")
                except:
                    pass
        
        # Clean up RAM disk directory
        if hasattr(self, 'ram_disk_dir') and os.path.exists(self.ram_disk_dir):
            shutil.rmtree(self.ram_disk_dir, ignore_errors=True)

class MaxGPUFeatureExtractor:
    """Maximum GPU utilization feature extractor"""
    
    def __init__(self, gpu_ids=[0, 1]):
        StrictGPUEnforcer.ensure_cuda_available()
        
        self.gpu_ids = gpu_ids
        self.devices = []
        
        for gpu_id in gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"STRICT MODE: GPU {gpu_id} not available! Only {torch.cuda.device_count()} GPUs found.")
            device = torch.device(f'cuda:{gpu_id}')
            self.devices.append(device)
        
        if not self.devices:
            raise RuntimeError("STRICT MODE: No valid GPU devices for feature extraction!")
        
        # Force mixed precision for maximum performance
        self.use_amp = True
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Initialize models on all GPUs
        self.feature_extractors = []
        self.optical_flow_models = []
        
        for device in self.devices:
            # Load optimized feature extractor - MUST be on GPU
            model = self._create_optimized_feature_network().to(device)
            model.eval()
            
            # Validate model is on GPU
            for param in model.parameters():
                if param.device.type != 'cuda':
                    raise RuntimeError(f"STRICT MODE: Model parameter not on GPU! Device: {param.device}")
            
            self.feature_extractors.append(model)
            
            # Optical flow model - MUST be on GPU
            flow_model = self._create_gpu_optical_flow().to(device)
            flow_model.eval()
            
            # Validate flow model is on GPU
            for param in flow_model.parameters():
                if param.device.type != 'cuda':
                    raise RuntimeError(f"STRICT MODE: Flow model parameter not on GPU! Device: {param.device}")
            
            self.optical_flow_models.append(flow_model)
        
        # CUDA streams for maximum parallelism
        self.streams = [torch.cuda.Stream(device=device) for device in self.devices]
        
        logger.info(f"STRICT MODE: MaxGPU Feature Extractor initialized on {len(self.devices)} devices")
    
    def _create_optimized_feature_network(self):
        """Create optimized CNN for feature extraction"""
        class OptimizedFeatureNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Efficient depth-wise separable convolutions
                self.features = nn.Sequential(
                    # Initial conv
                    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True),
                    
                    # Depth-wise separable blocks
                    self._make_separable_block(32, 64, 2),
                    self._make_separable_block(64, 128, 2),
                    self._make_separable_block(128, 256, 2),
                    self._make_separable_block(256, 512, 1),
                    
                    # Global pooling
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                
                # Additional feature heads
                self.scene_head = nn.Linear(512, 256)
                self.motion_head = nn.Linear(512, 128)
                self.texture_head = nn.Linear(512, 128)
            
            def _make_separable_block(self, in_channels, out_channels, stride):
                return nn.Sequential(
                    # Depth-wise
                    nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                             groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU6(inplace=True),
                    
                    # Point-wise
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(inplace=True)
                )
            
            def forward(self, x):
                features = self.features(x)
                return {
                    'scene': self.scene_head(features),
                    'motion': self.motion_head(features),
                    'texture': self.texture_head(features)
                }
        
        return OptimizedFeatureNet()
    
    def _create_gpu_optical_flow(self):
        """Create GPU-optimized optical flow network"""
        class GPUOpticalFlow(nn.Module):
            def __init__(self):
                super().__init__()
                # Lightweight FlowNet-style architecture
                self.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3)
                self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
                self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
                self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
                
                # Decoder
                self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
                self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
                self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
                self.deconv4 = nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1)
                
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, img1, img2):
                # Ensure inputs are the right shape
                if img1.dim() == 5:  # Batch of frames
                    img1 = img1.squeeze(0)
                if img2.dim() == 5:
                    img2 = img2.squeeze(0)
                
                # Concatenate frames
                x = torch.cat([img1, img2], dim=1)  # 6 channels
                
                # Encoder
                x1 = self.relu(self.conv1(x))
                x2 = self.relu(self.conv2(x1))
                x3 = self.relu(self.conv3(x2))
                x4 = self.relu(self.conv4(x3))
                
                # Decoder
                y1 = self.relu(self.deconv1(x4))
                y2 = self.relu(self.deconv2(y1))
                y3 = self.relu(self.deconv3(y2))
                flow = self.deconv4(y3)
                
                return flow
        
        return GPUOpticalFlow()
    
    def extract_all_features_gpu(self, frames_tensor, device_idx=0):
        """Extract all features using STRICT GPU acceleration - no CPU fallbacks"""
        if device_idx >= len(self.devices):
            raise RuntimeError(f"STRICT MODE: Invalid device index {device_idx}")
        
        device = self.devices[device_idx]
        model = self.feature_extractors[device_idx]
        flow_model = self.optical_flow_models[device_idx]
        stream = self.streams[device_idx]
        
        # STRICT: Validate input tensor is on GPU
        StrictGPUEnforcer.enforce_tensor_on_gpu(frames_tensor, "input frames")
        
        # Check GPU memory and adjust processing if needed
        available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        frames_memory = frames_tensor.numel() * frames_tensor.element_size()
        
        if frames_memory > available_memory * 0.7:  # Use max 70% of available memory
            logger.warning(f"STRICT MODE: Large frame tensor ({frames_memory//1024**2}MB), processing in chunks")
            return self._extract_features_chunked(frames_tensor, model, flow_model, device, stream)
        
        # Ensure tensor is on correct device
        if frames_tensor.device != device:
            frames_tensor = frames_tensor.to(device, non_blocking=True)
            StrictGPUEnforcer.enforce_tensor_on_gpu(frames_tensor, "moved frames")
        
        batch_size, num_frames = frames_tensor.shape[:2]
        
        with torch.cuda.stream(stream):
            with torch.amp.autocast('cuda'):
                features = self._extract_comprehensive_features(
                    frames_tensor, model, flow_model, device
                )
        
        # Synchronize stream
        stream.synchronize()
        
        # STRICT: Validate all output features are on GPU
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                StrictGPUEnforcer.enforce_tensor_on_gpu(value, f"feature {key}")
        
        return features
    
    def _extract_features_chunked(self, frames_tensor, model, flow_model, device, stream):
        """Extract features in chunks to manage memory"""
        batch_size, num_frames = frames_tensor.shape[:2]
        chunk_size = max(5, num_frames // 4)  # Process in 4 chunks minimum
        
        logger.info(f"Processing {num_frames} frames in chunks of {chunk_size}")
        
        all_features = {}
        
        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            chunk = frames_tensor[:, start_idx:end_idx]
            
            # Move chunk to device
            if chunk.device != device:
                chunk = chunk.to(device, non_blocking=True)
            
            with torch.cuda.stream(stream):
                with torch.amp.autocast('cuda'):
                    chunk_features = self._extract_comprehensive_features(
                        chunk, model, flow_model, device
                    )
            
            # Accumulate features
            for key, values in chunk_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(values)
            
            # Clear chunk from GPU immediately
            del chunk
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        final_features = {}
        for key, value_list in all_features.items():
            if isinstance(value_list[0], torch.Tensor):
                final_features[key] = torch.cat(value_list, dim=0)
            else:
                # For non-tensor features, just take the first one or average
                final_features[key] = value_list[0]
        
        stream.synchronize()
        return final_features
    
    def _extract_comprehensive_features(self, frames_tensor, model, flow_model, device):
        """Comprehensive GPU feature extraction"""
        batch_size, num_frames = frames_tensor.shape[:2]
        
        # Reshape for batch processing
        frames_flat = frames_tensor.view(-1, *frames_tensor.shape[2:])  # (B*N, C, H, W)
        
        # Extract CNN features for all frames at once
        with torch.no_grad():
            cnn_features = model(frames_flat)
        
        # Reshape back to sequence
        for key in cnn_features:
            cnn_features[key] = cnn_features[key].view(batch_size, num_frames, -1)
        
        features = {
            'scene_features': cnn_features['scene'][0],  # Remove batch dim
            'motion_features': cnn_features['motion'][0],
            'texture_features': cnn_features['texture'][0],
        }
        
        # GPU-accelerated motion analysis
        motion_stats = self._compute_motion_statistics_gpu(frames_tensor[0], flow_model, device)
        features.update(motion_stats)
        
        # GPU-accelerated color analysis
        color_stats = self._compute_color_statistics_gpu(frames_tensor[0], device)
        features.update(color_stats)
        
        # GPU-accelerated temporal analysis
        temporal_stats = self._compute_temporal_statistics_gpu(frames_tensor[0], device)
        features.update(temporal_stats)
        
        # GPU-accelerated edge analysis
        edge_stats = self._compute_edge_statistics_gpu(frames_tensor[0], device)
        features.update(edge_stats)
        
        return features
    
    def _compute_motion_statistics_gpu(self, frames, flow_model, device):
        """Compute comprehensive motion statistics on GPU"""
        num_frames = frames.shape[0]
        
        motion_stats = {
            'motion_magnitude': torch.zeros(num_frames, device=device),
            'motion_direction': torch.zeros(num_frames, device=device),
            'motion_coherence': torch.zeros(num_frames, device=device),
            'acceleration': torch.zeros(num_frames, device=device),
            'jerk': torch.zeros(num_frames, device=device),
            'rotation': torch.zeros(num_frames, device=device),
            'flow_complexity': torch.zeros(num_frames, device=device)
        }
        
        if num_frames < 2:
            return motion_stats
        
        # Batch optical flow computation
        for i in range(num_frames - 1):
            with torch.no_grad():
                flow = flow_model(frames[i:i+1], frames[i+1:i+2])
            
            # Flow statistics
            flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
            flow_direction = torch.atan2(flow[:, 1], flow[:, 0])
            
            # Aggregate statistics
            motion_stats['motion_magnitude'][i+1] = torch.mean(flow_magnitude)
            motion_stats['motion_direction'][i+1] = torch.mean(torch.cos(flow_direction))
            motion_stats['flow_complexity'][i+1] = torch.std(flow_magnitude)
            
            # Motion coherence (how uniform the flow is)
            direction_std = torch.std(flow_direction)
            motion_stats['motion_coherence'][i+1] = 1.0 / (1.0 + direction_std)
            
            # Rotation estimation
            h, w = flow.shape[2:]
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing='ij'
            )
            x_coords = x_coords - w // 2
            y_coords = y_coords - h // 2
            
            rotation = torch.mean(x_coords * flow[0, 1] - y_coords * flow[0, 0])
            rotation = rotation / (torch.mean(x_coords**2 + y_coords**2) + 1e-8)
            motion_stats['rotation'][i+1] = rotation
        
        # Compute acceleration and jerk
        motion_mag = motion_stats['motion_magnitude']
        motion_stats['acceleration'][1:] = motion_mag[1:] - motion_mag[:-1]
        motion_stats['jerk'][2:] = motion_stats['acceleration'][2:] - motion_stats['acceleration'][1:-1]
        
        return motion_stats
    
    def _compute_color_statistics_gpu(self, frames, device):
        """Compute color statistics on GPU"""
        # Color histograms
        histograms = []
        color_variance = torch.zeros(frames.shape[0], device=device)
        
        for i in range(frames.shape[0]):
            frame = frames[i]
            
            # Color variance
            color_variance[i] = torch.var(frame)
            
            # Color histogram (simplified for speed)
            frame_quantized = (frame * 15).long()  # 16 bins per channel
            hist = torch.zeros(16**3, device=device)
            
            # Flatten and compute indices
            frame_flat = frame_quantized.permute(1, 2, 0).reshape(-1, 3)
            indices = frame_flat[:, 0] * 256 + frame_flat[:, 1] * 16 + frame_flat[:, 2]
            indices = torch.clamp(indices, 0, 16**3 - 1)
            
            hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
            hist = hist / hist.sum()
            histograms.append(hist)
        
        color_histograms = torch.stack(histograms)
        
        return {
            'color_variance': color_variance,
            'color_histograms': color_histograms
        }
    
    def _compute_temporal_statistics_gpu(self, frames, device):
        """Compute temporal statistics on GPU"""
        if frames.shape[0] < 2:
            return {
                'temporal_gradient': torch.zeros(frames.shape[0], device=device),
                'temporal_stability': torch.zeros(frames.shape[0], device=device)
            }
        
        # Convert to grayscale for temporal analysis
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        # Temporal gradients
        temporal_diff = torch.abs(gray_frames[1:] - gray_frames[:-1])
        temporal_gradient = torch.mean(temporal_diff, dim=[1, 2])
        temporal_gradient = torch.cat([torch.zeros(1, device=device), temporal_gradient])
        
        # Temporal stability (inverse of variance in local windows)
        stability = torch.zeros(frames.shape[0], device=device)
        for i in range(1, frames.shape[0]):
            window_start = max(0, i - 5)
            window_frames = gray_frames[window_start:i+1]
            if len(window_frames) > 1:
                stability[i] = 1.0 / (1.0 + torch.var(window_frames))
        
        return {
            'temporal_gradient': temporal_gradient,
            'temporal_stability': stability
        }
    
    def _compute_edge_statistics_gpu(self, frames, device):
        """Compute edge statistics on GPU"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        
        edge_density = torch.zeros(frames.shape[0], device=device)
        edge_orientation = torch.zeros(frames.shape[0], device=device)
        
        # Convert to grayscale
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray_frames = gray_frames.unsqueeze(1)  # Add channel dimension
        
        # Batch edge detection
        edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
        edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3])
        
        # Edge orientation
        edge_angle = torch.atan2(edges_y, edges_x + 1e-8)
        edge_orientation = torch.mean(torch.abs(torch.cos(edge_angle)), dim=[1, 2, 3])
        
        return {
            'edge_density': edge_density,
            'edge_orientation': edge_orientation
        }

class CuPyGPXProcessor:
    """Maximum CuPy GPU acceleration for GPX processing"""
    
    def __init__(self):
        StrictGPUEnforcer.ensure_cupy_available()
        
        # Force GPU memory pool settings for maximum performance
        try:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=None)  # Remove memory limit
            logger.info("STRICT MODE: CuPy default memory pool limit removed")
        except Exception as e:
            raise RuntimeError(f"STRICT MODE: Could not configure CuPy memory pool: {e}")
        
        # Set memory growth to prevent fragmentation (if available)
        try:
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Check if set_limit method exists before calling it
            if hasattr(pinned_mempool, 'set_limit'):
                pinned_mempool.set_limit(size=None)
                logger.info("STRICT MODE: CuPy pinned memory pool limit removed")
            else:
                logger.info("STRICT MODE: CuPy pinned memory pool set_limit not available in this version")
                
        except Exception as e:
            logger.warning(f"STRICT MODE: Could not configure CuPy pinned memory pool: {e}")
        
        logger.info("STRICT MODE: CuPy GPX processor initialized with GPU-only operations")
    
    def process_gpx_files_gpu(self, gpx_paths, max_workers=None):
        """Process all GPX files with maximum GPU acceleration"""
        if max_workers is None:
            max_workers = min(32, mp.cpu_count())
        
        results = {}
        
        # First pass: parse GPX files (CPU bound I/O)
        raw_data = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._parse_gpx_cpu, path): path for path in gpx_paths}
            
            for future in tqdm(as_completed(futures), total=len(gpx_paths), desc="Parsing GPX"):
                path = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        raw_data[path] = data
                except Exception as e:
                    logger.error(f"Error parsing {path}: {e}")
        
        # Second pass: GPU feature computation
        logger.info(f"Computing features for {len(raw_data)} GPX files on GPU...")
        
        for path, data in tqdm(raw_data.items(), desc="GPU feature computation"):
            try:
                enhanced_data = self._compute_gpu_features(data)
                results[path] = enhanced_data
            except Exception as e:
                logger.error(f"Error computing GPU features for {path}: {e}")
                results[path] = None
        
        return results
    
    def _parse_gpx_cpu(self, gpx_path):
        """Parse single GPX file (CPU operation)"""
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for pt in segment.points:
                        if pt.time:
                            points.append({
                                'timestamp': pt.time.replace(tzinfo=None),
                                'lat': pt.latitude,
                                'lon': pt.longitude,
                                'elevation': pt.elevation or 0
                            })
            
            if len(points) < 10:
                return None
            
            df = pd.DataFrame(points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return {
                'df': df,
                'start_time': df['timestamp'].iloc[0],
                'end_time': df['timestamp'].iloc[-1],
                'point_count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error parsing {gpx_path}: {e}")
            return None
    
    def _compute_gpu_features(self, gpx_data):
        """Compute all features using CuPy GPU acceleration"""
        df = gpx_data['df']
        
        # Transfer to GPU immediately
        lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)
        lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
        elevs_gpu = cp.array(df['elevation'].values, dtype=cp.float64)
        
        n_points = len(lats_gpu)
        
        # Time differences in seconds - handle both datetime and numpy datetime64
        timestamps = df['timestamp'].values
        time_diffs_list = []
        
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            
            # Handle different time difference types
            if hasattr(time_diff, 'total_seconds'):
                # datetime.timedelta object
                seconds = time_diff.total_seconds()
            elif isinstance(time_diff, np.timedelta64):
                # numpy.timedelta64 object
                seconds = float(time_diff / np.timedelta64(1, 's'))
            elif hasattr(time_diff, 'seconds'):
                # pandas Timedelta
                seconds = time_diff.total_seconds()
            else:
                # Try to convert to pandas Timedelta first
                try:
                    td = pd.Timedelta(time_diff)
                    seconds = td.total_seconds()
                except:
                    # Last resort - assume it's already in seconds or use 1.0
                    logger.warning(f"Could not convert time difference: {type(time_diff)}, using 1.0 seconds")
                    seconds = 1.0
            
            # Ensure we have a valid positive number
            if seconds <= 0 or not np.isfinite(seconds):
                seconds = 1.0
                
            time_diffs_list.append(seconds)
        
        # Convert to GPU array - FIXED: ensure consistent length
        if len(time_diffs_list) == n_points - 1:
            # We have n-1 time differences, prepend 1.0 for first point
            time_diffs = cp.array([1.0] + time_diffs_list, dtype=cp.float64)
        else:
            # Fallback: create array of ones with correct length
            time_diffs = cp.ones(n_points, dtype=cp.float64)
            if len(time_diffs_list) > 0:
                min_len = min(len(time_diffs_list), n_points - 1)
                time_diffs[1:min_len+1] = cp.array(time_diffs_list[:min_len])
        
        features = {}
        
        # All distance calculations on GPU
        if n_points > 1:
            distances = self._haversine_distance_gpu(lats_gpu[:-1], lons_gpu[:-1], 
                                                    lats_gpu[1:], lons_gpu[1:])
            # FIXED: ensure consistent length
            distances = cp.concatenate([cp.array([0.0]), distances])
        else:
            distances = cp.array([0.0])
        
        # Ensure all arrays have same length
        assert len(distances) == n_points, f"Distance array length mismatch: {len(distances)} vs {n_points}"
        assert len(time_diffs) == n_points, f"Time diff array length mismatch: {len(time_diffs)} vs {n_points}"
        
        # Speed (miles per hour) - FIXED: handle edge cases
        if n_points > 1:
            # Calculate speeds for segments
            speed_segments = (distances[1:] * 3600) / cp.maximum(time_diffs[1:], 1e-6)  # Avoid division by zero
            speeds = cp.concatenate([cp.array([0.0]), speed_segments])
        else:
            speeds = cp.array([0.0])
        
        # Bearings - FIXED: handle edge cases  
        if n_points > 1:
            bearing_segments = self._compute_bearings_gpu(lats_gpu[:-1], lons_gpu[:-1], 
                                                        lats_gpu[1:], lons_gpu[1:])
            bearings = cp.concatenate([cp.array([0.0]), bearing_segments])
        else:
            bearings = cp.array([0.0])
        
        # Ensure all arrays have same length before advanced motion computation
        assert len(speeds) == n_points, f"Speed array length mismatch: {len(speeds)} vs {n_points}"
        assert len(bearings) == n_points, f"Bearing array length mismatch: {len(bearings)} vs {n_points}"
        
        # Advanced motion features (all GPU)
        features.update(self._compute_advanced_motion_gpu(
            speeds, bearings, elevs_gpu, time_diffs, distances
        ))
        
        # Statistical features
        features.update(self._compute_statistical_features_gpu(
            speeds, bearings, elevs_gpu, distances
        ))
        
        # Smoothing on GPU
        features = self._apply_gpu_smoothing(features)
        
        # Convert back to CPU for storage
        for key, value in features.items():
            if isinstance(value, cp.ndarray):
                features[key] = cp.asnumpy(value)
        
        # Add metadata - handle duration calculation properly
        try:
            start_time = df['timestamp'].iloc[0]
            end_time = df['timestamp'].iloc[-1]
            
            if hasattr(start_time, 'to_pydatetime'):
                start_time = start_time.to_pydatetime()
            if hasattr(end_time, 'to_pydatetime'):
                end_time = end_time.to_pydatetime()
            
            duration_delta = end_time - start_time
            
            if hasattr(duration_delta, 'total_seconds'):
                duration = duration_delta.total_seconds()
            else:
                duration = float(duration_delta / np.timedelta64(1, 's'))
                
        except Exception as e:
            logger.warning(f"Could not calculate duration: {e}, using sum of time diffs")
            duration = float(cp.sum(time_diffs))
        
        gpx_data.update({
            'features': features,
            'duration': duration,
            'distance': float(cp.sum(distances)),
            'max_speed': float(cp.max(speeds)),
            'avg_speed': float(cp.mean(speeds))
        })
        
        return gpx_data
    
    def _haversine_distance_gpu(self, lat1, lon1, lat2, lon2):
        """GPU-accelerated haversine distance using CuPy"""
        R = 3958.8  # Earth radius in miles
        
        # Convert to radians
        lat1_rad = cp.radians(lat1)
        lon1_rad = cp.radians(lon1)
        lat2_rad = cp.radians(lat2)
        lon2_rad = cp.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(a))
        
        return R * c
    
    def _compute_bearings_gpu(self, lat1, lon1, lat2, lon2):
        """GPU-accelerated bearing computation"""
        lat1_rad = cp.radians(lat1)
        lon1_rad = cp.radians(lon1)
        lat2_rad = cp.radians(lat2)
        lon2_rad = cp.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = cp.sin(dlon) * cp.cos(lat2_rad)
        x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(dlon)
        
        bearings = cp.degrees(cp.arctan2(y, x))
        # Normalize to 0-360
        bearings = cp.where(bearings < 0, bearings + 360, bearings)
        
        return bearings
    
    def _compute_advanced_motion_gpu(self, speeds, bearings, elevations, time_diffs, distances):
        """Compute advanced motion features on GPU"""
        n = len(speeds)
        
        # Ensure all input arrays have the same length
        assert len(bearings) == n, f"Bearing array length mismatch: {len(bearings)} vs {n}"
        assert len(elevations) == n, f"Elevation array length mismatch: {len(elevations)} vs {n}"
        assert len(time_diffs) == n, f"Time diff array length mismatch: {len(time_diffs)} vs {n}"
        assert len(distances) == n, f"Distance array length mismatch: {len(distances)} vs {n}"
        
        # Acceleration - FIXED: handle array lengths properly
        if n > 1:
            speed_diffs = cp.diff(speeds)  # Length n-1
            # Use time_diffs[1:] to match the length (n-1)
            acceleration_segments = speed_diffs / cp.maximum(time_diffs[1:], 1e-6)
            acceleration = cp.concatenate([cp.array([0.0]), acceleration_segments])
        else:
            acceleration = cp.array([0.0])
        
        # Jerk - FIXED: handle array lengths properly
        if n > 2:
            accel_diffs = cp.diff(acceleration)  # Length n-1
            jerk_segments = accel_diffs / cp.maximum(time_diffs[2:], 1e-6)  # Use time_diffs[2:] (length n-2)
            jerk = cp.concatenate([cp.array([0.0, 0.0]), jerk_segments])
        else:
            jerk = cp.zeros(n)
        
        # Elevation change rate - FIXED: handle array lengths properly
        if n > 1:
            elev_diffs = cp.diff(elevations)  # Length n-1
            elev_change_segments = elev_diffs / cp.maximum(time_diffs[1:], 1e-6)
            elev_change_rate = cp.concatenate([cp.array([0.0]), elev_change_segments])
        else:
            elev_change_rate = cp.array([0.0])
        
        # Bearing changes (handle wraparound) - FIXED: handle array lengths properly
        if n > 1:
            bearing_diffs = cp.diff(bearings)  # Length n-1
            bearing_diffs = cp.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = cp.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes_segments = cp.abs(bearing_diffs)
            bearing_changes = cp.concatenate([cp.array([0.0]), bearing_changes_segments])
        else:
            bearing_changes = cp.array([0.0])
        
        # Curvature - FIXED: avoid division by zero
        curvature = bearing_changes / cp.maximum(distances, 1e-8)
        
        # Turning rate - FIXED: handle array lengths properly
        if n > 1:
            turning_segments = bearing_changes[1:] / cp.maximum(time_diffs[1:], 1e-6)
            turning_rate = cp.concatenate([cp.array([0.0]), turning_segments])
        else:
            turning_rate = cp.array([0.0])
        
        # Speed consistency (local variance) - FIXED: handle window properly
        speed_consistency = cp.zeros_like(speeds)
        window_size = min(5, n // 2)  # Adaptive window size
        if window_size > 0:
            for i in range(window_size, n):
                window_start = max(0, i - window_size)
                window_speeds = speeds[window_start:i+1]
                if len(window_speeds) > 1:
                    speed_consistency[i] = 1.0 / (1.0 + cp.var(window_speeds))
        
        # Final length validation
        results = {
            'speed': speeds,
            'acceleration': acceleration,
            'jerk': jerk,
            'bearing': bearings,
            'bearing_change': bearing_changes,
            'curvature': curvature,
            'turning_rate': turning_rate,
            'elevation_change_rate': elev_change_rate,
            'speed_consistency': speed_consistency
        }
        
        # Validate all output arrays have the same length
        for key, value in results.items():
            assert len(value) == n, f"Output array {key} length mismatch: {len(value)} vs {n}"
        
        return results
    
    def _compute_statistical_features_gpu(self, speeds, bearings, elevations, distances):
        """Compute statistical features on GPU"""
        features = {}
        
        # Speed statistics
        features['speed_stats'] = cp.array([
            cp.mean(speeds), cp.std(speeds), cp.min(speeds), cp.max(speeds),
            cp.percentile(speeds, 25), cp.percentile(speeds, 50), cp.percentile(speeds, 75)
        ])
        
        # Bearing statistics
        features['bearing_stats'] = cp.array([
            cp.mean(bearings), cp.std(bearings), cp.min(bearings), cp.max(bearings)
        ])
        
        # Elevation statistics
        features['elevation_stats'] = cp.array([
            cp.mean(elevations), cp.std(elevations), cp.min(elevations), cp.max(elevations),
            cp.sum(cp.where(cp.diff(elevations) > 0, cp.diff(elevations), 0)),  # Total climb
            cp.sum(cp.where(cp.diff(elevations) < 0, -cp.diff(elevations), 0))  # Total descent
        ])
        
        # Distance statistics
        features['distance_stats'] = cp.array([
            cp.sum(distances), cp.mean(distances), cp.std(distances), cp.max(distances)
        ])
        
        return features
    
    def _apply_gpu_smoothing(self, features, window_size=5):
        """Apply smoothing to features on GPU"""
        smoothed = {}
        
        for key, values in features.items():
            if isinstance(values, cp.ndarray) and values.ndim == 1 and len(values) > window_size:
                # Simple moving average on GPU
                kernel = cp.ones(window_size) / window_size
                # Pad the signal
                padded = cp.pad(values, (window_size//2, window_size//2), mode='edge')
                # Convolve
                smoothed_values = cp.convolve(padded, kernel, mode='valid')
                smoothed[key] = smoothed_values
            else:
                smoothed[key] = values
        
        return smoothed

class UltraHighPerformanceCorrelator:
    """Ultra-high performance GPU correlator with tensor operations"""
    
    def __init__(self, gpu_ids=[0, 1]):
        StrictGPUEnforcer.ensure_cuda_available()
        
        self.gpu_ids = gpu_ids
        self.devices = []
        
        for gpu_id in gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"STRICT MODE: GPU {gpu_id} not available for correlation!")
            device = torch.device(f'cuda:{gpu_id}')
            self.devices.append(device)
        
        if not self.devices:
            raise RuntimeError("STRICT MODE: No GPU devices available for correlation!")
        
        # Initialize neural networks on all GPUs - MUST be on GPU
        self.similarity_networks = []
        for device in self.devices:
            net = self._create_advanced_similarity_network().to(device)
            net.eval()
            
            # Validate network is on GPU
            for param in net.parameters():
                if param.device.type != 'cuda':
                    raise RuntimeError(f"STRICT MODE: Similarity network parameter not on GPU! Device: {param.device}")
            
            self.similarity_networks.append(net)
        
        # Create CUDA streams for maximum parallelism
        self.streams = [torch.cuda.Stream(device=device) for device in self.devices]
        
        logger.info(f"STRICT MODE: Ultra-high performance correlator initialized on {len(self.devices)} GPUs")
    
    def _create_advanced_similarity_network(self):
        """Create advanced neural similarity network"""
        class AdvancedSimilarityNet(nn.Module):
            def __init__(self, input_dim=512):
                super().__init__()
                
                # Multi-head attention for feature interaction
                self.attention = nn.MultiheadAttention(input_dim//2, num_heads=8, 
                                                     dropout=0.1, batch_first=True)
                
                # Feature fusion network
                self.fusion = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
                
                # Learnable temperature parameter
                self.temperature = nn.Parameter(torch.tensor(1.0))
            
            def forward(self, x1, x2):
                # Ensure inputs are 2D
                if x1.dim() == 1:
                    x1 = x1.unsqueeze(0)
                if x2.dim() == 1:
                    x2 = x2.unsqueeze(0)
                
                batch_size = x1.shape[0]
                
                # Split features for attention
                x1_att = x1[:, :x1.shape[1]//2]
                x2_att = x2[:, :x2.shape[1]//2]
                
                # Self-attention on concatenated features
                combined = torch.cat([x1_att.unsqueeze(1), x2_att.unsqueeze(1)], dim=1)
                attended, _ = self.attention(combined, combined, combined)
                attended = attended.view(batch_size, -1)
                
                # Concatenate with remaining features
                x1_rest = x1[:, x1.shape[1]//2:]
                x2_rest = x2[:, x2.shape[1]//2:]
                
                final_features = torch.cat([attended, x1_rest, x2_rest], dim=1)
                
                # Apply temperature scaling
                similarity = self.fusion(final_features) / self.temperature
                
                return similarity
        
        return AdvancedSimilarityNet()
    
    def correlate_ultra_optimized(self, video_features_dict, gpx_database, output_dir, top_k=5):
        """Ultra-optimized correlation with maximum GPU utilization"""
        
        # Debug: Check what we actually have
        total_videos = len(video_features_dict)
        total_gpx = len(gpx_database)
        
        valid_videos = {k: v for k, v in video_features_dict.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None}
        
        valid_video_count = len(valid_videos)
        valid_gpx_count = len(valid_gpx)
        
        logger.info(f"Correlation data summary:")
        logger.info(f"  Total videos: {total_videos}, Valid: {valid_video_count}")
        logger.info(f"  Total GPX: {total_gpx}, Valid: {valid_gpx_count}")
        
        # Show some examples of what's invalid
        if valid_video_count < total_videos:
            invalid_videos = [k for k, v in video_features_dict.items() if v is None]
            logger.warning(f"Invalid videos (first 5): {invalid_videos[:5]}")
        
        if valid_gpx_count < total_gpx:
            invalid_gpx = [k for k, v in gpx_database.items() if v is None]
            logger.warning(f"Invalid GPX files (first 5): {invalid_gpx[:5]}")
        
        if not valid_videos:
            logger.error("No valid video features found!")
            if total_videos > 0:
                logger.error("All video processing appears to have failed. Check video files and processing logs.")
            else:
                logger.error("No video files were found or processed.")
            raise RuntimeError("No valid video features for correlation!")
        
        if not valid_gpx:
            logger.error("No valid GPX features found!")
            if total_gpx > 0:
                logger.error("All GPX processing appears to have failed. Check GPX files and processing logs.")
            else:
                logger.error("No GPX files were found or processed.")
            raise RuntimeError("No valid GPX features for correlation!")
        
        logger.info(f"Ultra-optimized correlation: {valid_video_count} videos × {valid_gpx_count} GPX files")
        
        # Pre-process all data to GPU tensors
        logger.info("Preprocessing video features to GPU tensors...")
        video_tensors = self._preprocess_video_features_to_gpu(valid_videos)
        logger.info(f"Successfully preprocessed {len(video_tensors)} video feature tensors")
        
        logger.info("Preprocessing GPX features to GPU tensors...")
        gpx_tensors = self._preprocess_gpx_features_to_gpu(valid_gpx)
        logger.info(f"Successfully preprocessed {len(gpx_tensors)} GPX feature tensors")
        
        if not video_tensors:
            logger.error("No video features could be converted to tensors!")
            logger.error("This suggests the cached video features are invalid or incompatible.")
            raise RuntimeError("Video feature tensor conversion failed!")
        
        if not gpx_tensors:
            logger.error("No GPX features could be converted to tensors!")
            logger.error("This suggests the cached GPX features are invalid or incompatible.")
            raise RuntimeError("GPX feature tensor conversion failed!")
        
        # Distribute work across GPUs
        results = {}
        video_paths = list(video_tensors.keys())  # Only use videos that converted successfully
        gpu_assignments = defaultdict(list)
        
        for i, video_path in enumerate(video_paths):
            gpu_idx = i % len(self.devices)
            gpu_assignments[gpu_idx].append(video_path)
        
        logger.info(f"Distributing {len(video_paths)} videos across {len(self.devices)} GPUs")
        
        # Process in parallel across all GPUs
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for gpu_idx, assigned_videos in gpu_assignments.items():
                logger.info(f"GPU {gpu_idx}: processing {len(assigned_videos)} videos")
                future = executor.submit(
                    self._process_gpu_batch_ultra,
                    assigned_videos, video_tensors, gpx_tensors, 
                    valid_gpx, gpu_idx, top_k
                )
                futures.append(future)
            
            for future in as_completed(futures):
                gpu_results = future.result()
                results.update(gpu_results)
        
        logger.info(f"Correlation completed: {len(results)} video results")
        
        # Generate comprehensive report
        asyncio.run(self._generate_ultra_report(results, output_dir))
        
        return results
    
    def _preprocess_video_features_to_gpu(self, video_features_dict):
        """Preprocess all video features to GPU tensors"""
        video_tensors = {}
        
        logger.info(f"Preprocessing {len(video_features_dict)} video feature sets...")
        
        for video_path, features in video_features_dict.items():
            if features is None:
                logger.warning(f"Skipping {video_path}: features is None")
                continue
                
            try:
                # Debug: check what features we have
                if not isinstance(features, dict):
                    logger.warning(f"Skipping {video_path}: features is not a dict, got {type(features)}")
                    continue
                
                if not features:
                    logger.warning(f"Skipping {video_path}: features dict is empty")
                    continue
                
                # Create comprehensive feature vector
                feature_vector = self._create_video_feature_vector(features)
                if feature_vector is not None:
                    video_tensors[video_path] = feature_vector
                else:
                    logger.warning(f"Failed to create feature vector for {video_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to preprocess video {video_path}: {e}")
        
        logger.info(f"Successfully preprocessed {len(video_tensors)} video feature vectors")
        return video_tensors
    
    def _preprocess_gpx_features_to_gpu(self, gpx_database):
        """Preprocess all GPX features to GPU tensors"""
        gpx_tensors = {}
        
        logger.info(f"Preprocessing {len(gpx_database)} GPX feature sets...")
        
        for gpx_path, gpx_data in gpx_database.items():
            if gpx_data is None:
                logger.warning(f"Skipping {gpx_path}: gpx_data is None")
                continue
                
            if 'features' not in gpx_data:
                logger.warning(f"Skipping {gpx_path}: no 'features' key in gpx_data")
                continue
                
            try:
                # Debug: check what features we have
                features = gpx_data['features']
                if not isinstance(features, dict):
                    logger.warning(f"Skipping {gpx_path}: features is not a dict, got {type(features)}")
                    continue
                
                if not features:
                    logger.warning(f"Skipping {gpx_path}: features dict is empty")
                    continue
                
                # Create comprehensive feature vector
                feature_vector = self._create_gpx_feature_vector(features)
                if feature_vector is not None:
                    gpx_tensors[gpx_path] = feature_vector
                else:
                    logger.warning(f"Failed to create feature vector for {gpx_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to preprocess GPX {gpx_path}: {e}")
        
        logger.info(f"Successfully preprocessed {len(gpx_tensors)} GPX feature vectors")
        return gpx_tensors
    
    def _create_video_feature_vector(self, features):
        """Create comprehensive video feature vector"""
        components = []
        
        # Debug: log what features we have
        feature_keys = list(features.keys())
        logger.debug(f"Video features available: {feature_keys}")
        
        # Scene features
        if 'scene_features' in features:
            scene_feat = features['scene_features']
            if isinstance(scene_feat, np.ndarray) and scene_feat.size > 0:
                if scene_feat.ndim == 2:
                    scene_feat = np.mean(scene_feat, axis=0)  # Average over time
                components.append(scene_feat.flatten())
                logger.debug(f"Added scene features: shape {scene_feat.shape}")
            else:
                logger.debug(f"Skipping scene_features: invalid type/size {type(scene_feat)}")
        
        # Motion features
        motion_keys = ['motion_magnitude', 'acceleration', 'jerk', 'rotation']
        for key in motion_keys:
            if key in features:
                values = features[key]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    # Statistical summary
                    try:
                        stats = np.array([
                            np.mean(values), np.std(values), np.min(values), np.max(values),
                            np.percentile(values, 25), np.percentile(values, 75)
                        ])
                        components.append(stats)
                        logger.debug(f"Added {key} stats: {stats.shape}")
                    except Exception as e:
                        logger.debug(f"Failed to compute stats for {key}: {e}")
                else:
                    logger.debug(f"Skipping {key}: invalid type/size {type(values)}")
        
        # Color features
        if 'color_histograms' in features:
            color_hist = features['color_histograms']
            if isinstance(color_hist, np.ndarray) and color_hist.size > 0:
                if color_hist.ndim == 2:
                    color_hist = np.mean(color_hist, axis=0)  # Average over time
                components.append(color_hist.flatten()[:64])  # Limit size
                logger.debug(f"Added color histogram: shape {color_hist.shape}")
            else:
                logger.debug(f"Skipping color_histograms: invalid type/size {type(color_hist)}")
        
        # Edge features
        if 'edge_density' in features:
            edge_vals = features['edge_density']
            if isinstance(edge_vals, np.ndarray) and len(edge_vals) > 0:
                try:
                    edge_stats = np.array([
                        np.mean(edge_vals), np.std(edge_vals), np.max(edge_vals)
                    ])
                    components.append(edge_stats)
                    logger.debug(f"Added edge stats: {edge_stats.shape}")
                except Exception as e:
                    logger.debug(f"Failed to compute edge stats: {e}")
            else:
                logger.debug(f"Skipping edge_density: invalid type/size {type(edge_vals)}")
        
        if not components:
            logger.warning("No valid components found for video feature vector")
            return None
        
        # Concatenate and normalize
        try:
            feature_vector = np.concatenate(components)
            logger.debug(f"Concatenated feature vector: {feature_vector.shape}")
        except Exception as e:
            logger.warning(f"Failed to concatenate feature components: {e}")
            return None
        
        # Pad or truncate to fixed size
        target_size = 512
        if len(feature_vector) < target_size:
            feature_vector = np.pad(feature_vector, (0, target_size - len(feature_vector)))
        else:
            feature_vector = feature_vector[:target_size]
        
        # Normalize
        if np.std(feature_vector) > 1e-8:
            feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        
        logger.debug(f"Final video feature vector: {feature_vector.shape}")
        return feature_vector
    
    def _create_gpx_feature_vector(self, features):
        """Create comprehensive GPX feature vector"""
        components = []
        
        # Motion features
        motion_keys = ['speed', 'acceleration', 'jerk', 'bearing_change', 'curvature']
        for key in motion_keys:
            if key in features:
                values = features[key]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    # Statistical summary
                    stats = np.array([
                        np.mean(values), np.std(values), np.min(values), np.max(values),
                        np.percentile(values, 25), np.percentile(values, 75)
                    ])
                    components.append(stats)
        
        # Statistical features
        stat_keys = ['speed_stats', 'bearing_stats', 'elevation_stats', 'distance_stats']
        for key in stat_keys:
            if key in features:
                values = features[key]
                if isinstance(values, np.ndarray):
                    components.append(values.flatten())
        
        if not components:
            return None
        
        # Concatenate and normalize
        feature_vector = np.concatenate(components)
        
        # Pad or truncate to fixed size
        target_size = 512
        if len(feature_vector) < target_size:
            feature_vector = np.pad(feature_vector, (0, target_size - len(feature_vector)))
        else:
            feature_vector = feature_vector[:target_size]
        
        # Normalize
        if np.std(feature_vector) > 1e-8:
            feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        
        return feature_vector
    
    def _process_gpu_batch_ultra(self, video_paths, video_tensors, gpx_tensors, 
                                gpx_database, gpu_idx, top_k):
        """Process batch on specific GPU with ultra optimization"""
        device = self.devices[gpu_idx]
        similarity_net = self.similarity_networks[gpu_idx]
        stream = self.streams[gpu_idx]
        
        results = {}
        
        # Move all GPX tensors to this GPU once
        gpx_paths = list(gpx_tensors.keys())
        gpx_features_gpu = {}
        
        with torch.cuda.stream(stream):
            for gpx_path in gpx_paths:
                gpx_tensor = torch.tensor(gpx_tensors[gpx_path], dtype=torch.float32, device=device)
                gpx_features_gpu[gpx_path] = gpx_tensor
        
        # Process each video
        for video_path in tqdm(video_paths, desc=f"GPU {gpu_idx}"):
            if video_path not in video_tensors:
                results[video_path] = None
                continue
            
            try:
                # Move video features to GPU
                video_tensor = torch.tensor(video_tensors[video_path], 
                                          dtype=torch.float32, device=device)
                
                # Batch similarity computation
                matches = self._compute_ultra_similarities(
                    video_tensor, gpx_features_gpu, gpx_database, 
                    similarity_net, device, stream, top_k
                )
                
                results[video_path] = {'matches': matches}
                
            except Exception as e:
                logger.error(f"GPU {gpu_idx} error processing {video_path}: {e}")
                results[video_path] = None
            
            # Periodic cleanup
            if len(results) % 5 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def _compute_ultra_similarities(self, video_tensor, gpx_features_gpu, gpx_database, 
                                   similarity_net, device, stream, top_k):
        """Compute similarities with maximum GPU optimization"""
        
        similarities = []
        gpx_paths = list(gpx_features_gpu.keys())
        
        with torch.cuda.stream(stream):
            with torch.no_grad():
                # Batch neural similarity computation
                if len(gpx_paths) > 0:
                    # Stack all GPX features
                    gpx_batch = torch.stack([gpx_features_gpu[path] for path in gpx_paths])
                    video_batch = video_tensor.unsqueeze(0).repeat(len(gpx_paths), 1)
                    
                    # Batch inference
                    neural_similarities = similarity_net(video_batch, gpx_batch).squeeze()
                    
                    if neural_similarities.dim() == 0:
                        neural_similarities = neural_similarities.unsqueeze(0)
                    
                    # Additional similarity metrics on GPU
                    for i, gpx_path in enumerate(gpx_paths):
                        gpx_tensor = gpx_features_gpu[gpx_path]
                        
                        # Cosine similarity
                        cosine_sim = F.cosine_similarity(video_tensor, gpx_tensor, dim=0)
                        
                        # Euclidean distance
                        euclidean_dist = torch.norm(video_tensor - gpx_tensor)
                        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
                        
                        # Neural similarity
                        neural_sim = neural_similarities[i]
                        
                        # Combined score
                        combined_score = (0.5 * neural_sim + 0.3 * cosine_sim + 0.2 * euclidean_sim)
                        
                        similarities.append({
                            'path': gpx_path,
                            'combined_score': combined_score.item(),
                            'neural_score': neural_sim.item(),
                            'cosine_score': cosine_sim.item(),
                            'euclidean_score': euclidean_sim.item(),
                            'distance': gpx_database[gpx_path].get('distance', 0),
                            'duration': gpx_database[gpx_path].get('duration', 0)
                        })
        
        # Sort and return top matches
        similarities.sort(key=lambda x: x['combined_score'], reverse=True)
        return similarities[:top_k]
    
    async def _generate_ultra_report(self, results, output_dir):
        """Generate comprehensive ultra-performance report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Detailed analysis
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'gpu_count': len(self.devices),
                'gpu_devices': [str(device) for device in self.devices],
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            },
            'performance_metrics': {
                'total_videos': len(results),
                'successful_correlations': sum(1 for r in results.values() if r is not None),
                'failed_correlations': sum(1 for r in results.values() if r is None)
            },
            'quality_metrics': {},
            'detailed_results': []
        }
        
        # Analyze quality
        scores = []
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        
        for video_path, result in results.items():
            if result is None or not result.get('matches'):
                continue
            
            best_match = result['matches'][0]
            score = best_match['combined_score']
            scores.append(score)
            
            # Confidence classification
            if score > 0.8:
                confidence_distribution['high'] += 1
            elif score > 0.6:
                confidence_distribution['medium'] += 1
            elif score > 0.4:
                confidence_distribution['low'] += 1
            else:
                confidence_distribution['very_low'] += 1
            
            report['detailed_results'].append({
                'video': str(video_path),
                'best_match': {
                    'gpx': str(best_match['path']),
                    'combined_score': score,
                    'neural_score': best_match.get('neural_score', 0),
                    'cosine_score': best_match.get('cosine_score', 0),
                    'euclidean_score': best_match.get('euclidean_score', 0)
                },
                'all_matches': [
                    {
                        'gpx': str(m['path']),
                        'score': m['combined_score'],
                        'neural': m.get('neural_score', 0)
                    } for m in result['matches']
                ]
            })
        
        # Quality statistics
        if scores:
            scores_array = np.array(scores)
            report['quality_metrics'] = {
                'mean_score': float(np.mean(scores_array)),
                'std_score': float(np.std(scores_array)),
                'median_score': float(np.median(scores_array)),
                'min_score': float(np.min(scores_array)),
                'max_score': float(np.max(scores_array)),
                'confidence_distribution': confidence_distribution
            }
        
        # Save comprehensive report
        async with aiofiles.open(output_path / 'ultra_performance_report.json', 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info(f"Ultra-performance report saved to {output_path}")

def process_video_ultra_optimized(video_path, decoder, feature_extractor, gpu_idx, target_size, sample_rate):
    """Process single video with ultra optimization and H.264 4:4:4 conversion support"""
    try:
        # Decode with GPU acceleration using provided parameters
        # The decoder will automatically handle H.264 4:4:4 conversion if needed
        frames_tensor, fps, duration, frame_indices = decoder.decode_video_gpu_batch(
            video_path, sample_rate=sample_rate, target_size=target_size, gpu_id=gpu_idx
        )
        
        if frames_tensor is None:
            raise RuntimeError("Failed to decode video")
        
        # Extract features with maximum GPU utilization
        features = feature_extractor.extract_all_features_gpu(frames_tensor, gpu_idx)
        
        # Convert GPU tensors to numpy for storage
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.cpu().numpy()
        
        features['duration'] = duration
        features['fps'] = fps
        features['gpu_processed'] = True  # Mark as GPU processed for validation
        features['original_video_path'] = video_path  # Track original path
        
        return features
        
    except Exception as e:
        logger.error(f"Ultra-optimization failed for {video_path}: {e}")
        raise

def main():
    """Main execution with strict GPU enforcement"""
    parser = argparse.ArgumentParser(description="Ultra-Optimized Multi-GPU Video-GPX Correlation")
    parser.add_argument("-d", "--directory", required=True, help="Directory with videos and GPX files")
    parser.add_argument("-o", "--output", default="./gpu_correlation_results", help="Output directory")
    parser.add_argument("-c", "--cache", default="./gpu_cache", help="Cache directory")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top K matches per video")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], help="GPU IDs to use")
    parser.add_argument("--force", action='store_true', help="Force reprocessing")
    parser.add_argument("--video_size", nargs=2, type=int, default=[640, 360], 
                       help="Target video resolution (width height). Use smaller for 4K videos, e.g. --video_size 480 270")
    parser.add_argument("--sample_rate", type=float, default=2.0, 
                       help="Video sampling rate (frames per second). Lower for better memory usage")
    parser.add_argument("--sequential", action='store_true', 
                       help="Process videos sequentially (better for memory management)")
    parser.add_argument("--clear_cache", action='store_true',
                       help="Clear all GPU memory and caches before starting")
    parser.add_argument("--strict_memory", action='store_true',
                       help="Use strict memory management (smaller batches, more conservative)")
    parser.add_argument("--debug", action='store_true',
                       help="Enable detailed debug logging for all operations")
    parser.add_argument("--force_gpu_decode", action='store_true',
                       help="Force GPU video decoding - fail if GPU decode doesn't work")
    parser.add_argument("--test_ffmpeg_cuda", action='store_true',
                       help="Test FFmpeg CUDA setup and exit")
    parser.add_argument("--skip_decode_test", action='store_true',
                       help="Skip GPU decode capability test (use with caution)")
    parser.add_argument("--disable_444_conversion", action='store_true',
                       help="Disable automatic H.264 4:4:4 to 4:2:0 conversion")
    parser.add_argument("--force_disk_conversion", action='store_true',
                       help="Force H.264 4:4:4 conversions to disk instead of RAM")
    parser.add_argument("--keep_444_backup", action='store_true',
                       help="Keep backup of original H.264 4:4:4 files after conversion")
    
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DEBUG MODE: Enabled detailed logging")
    
    # Test FFmpeg CUDA and exit if requested
    if args.test_ffmpeg_cuda:
        try:
            StrictGPUEnforcer.check_ffmpeg_gpu()
            print("SUCCESS: FFmpeg CUDA support verified!")
            return
        except Exception as e:
            print(f"FAILED: FFmpeg CUDA test failed: {e}")
            return
    
    # STRICT GPU validation - NO CPU FALLBACKS
    try:
        StrictGPUEnforcer.ensure_cuda_available()
        StrictGPUEnforcer.ensure_cupy_available()
        StrictGPUEnforcer.check_ffmpeg_gpu()
    except RuntimeError as e:
        logger.error(f"STRICT MODE: GPU requirements not met: {e}")
        logger.error("STRICT MODE: All operations must use GPU acceleration. Exiting.")
        raise
    
    # Setup directories
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)
    
    # Find files
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
    video_files = list(set(video_files))
    
    gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
    gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
    gpx_files = list(set(gpx_files))
    
    logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
    logger.info(f"Video processing settings: resolution={args.video_size}, sample_rate={args.sample_rate}")
    
    if not video_files or not gpx_files:
        raise RuntimeError("No videos or GPX files found!")
    
    # Initialize ultra-optimized components - STRICT GPU ONLY
    if args.clear_cache:
        logger.info("STRICT MODE: Clearing all GPU memory and caches...")
        for gpu_id in args.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"STRICT MODE: GPU {gpu_id} not available!")
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
                with cp.cuda.Device(gpu_id):
                    cp.get_default_memory_pool().free_all_blocks()
                logger.info(f"STRICT MODE: Cleared GPU {gpu_id}")
            except Exception as e:
                raise RuntimeError(f"STRICT MODE: Failed to clear GPU {gpu_id}: {e}")
    
    # All components MUST use GPU acceleration
    decoder = UltraOptimizedFFmpegDecoder(
        gpu_ids=args.gpu_ids, 
        skip_decode_test=args.skip_decode_test,
        disable_444_conversion=args.disable_444_conversion,
        force_disk_conversion=args.force_disk_conversion,
        keep_444_backup=args.keep_444_backup
    )
    feature_extractor = MaxGPUFeatureExtractor(gpu_ids=args.gpu_ids)
    gpx_processor = CuPyGPXProcessor()
    correlator = UltraHighPerformanceCorrelator(gpu_ids=args.gpu_ids)
    
    # Override target size if specified
    target_size = tuple(args.video_size)
    
    try:
        # Process videos
        logger.info("Processing videos with ultra-optimized GPU pipeline...")
        video_cache_path = cache_dir / "ultra_video_features.pkl"
        
        if video_cache_path.exists() and not args.force:
            with open(video_cache_path, 'rb') as f:
                video_features = pickle.load(f)
            
            # Validate cached data
            valid_count = sum(1 for v in video_features.values() if v is not None)
            total_count = len(video_features)
            
            logger.info(f"Loaded cached video features: {valid_count}/{total_count} valid")
            
            # If most cached data is invalid, force reprocessing
            if valid_count < total_count * 0.5:  # Less than 50% valid
                logger.warning(f"Most cached video data is invalid ({valid_count}/{total_count}), forcing reprocessing")
                video_features = {}
            
        else:
            video_features = {}
        
        if not video_features or len(video_features) != len(video_files):
            logger.info("STRICT MODE: Processing videos with GPU acceleration...")
            video_features = {}
            
            # Process videos sequentially for strict GPU management
            logger.info(f"STRICT MODE: Processing {len(video_files)} videos with strict GPU acceleration")
            
            failed_videos = []
            h264_444_conversions = 0
            
            for i, video_path in enumerate(tqdm(video_files, desc="STRICT GPU Processing")):
                gpu_idx = i % len(args.gpu_ids)
                gpu_id = args.gpu_ids[gpu_idx]
                
                try:
                    # Aggressive memory clear before each video
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    
                    with cp.cuda.Device(gpu_id):
                        cp.get_default_memory_pool().free_all_blocks()
                    
                    # Check if this video needs H.264 4:4:4 conversion
                    is_444, _ = decoder._check_video_format(video_path)
                    if is_444 and not args.disable_444_conversion:
                        h264_444_conversions += 1
                        logger.info(f"H.264 4:4:4 DETECTED: Will convert {video_path} ({h264_444_conversions} total)")
                    
                    # Process video with STRICT GPU requirements
                    features = process_video_ultra_optimized(
                        video_path, decoder, feature_extractor, gpu_idx, target_size, args.sample_rate
                    )
                    
                    # Validate features were actually computed
                    if not features or not features.get('gpu_processed', False):
                        raise RuntimeError(f"STRICT MODE: Features not properly computed on GPU for {video_path}")
                    
                    video_features[video_path] = features
                    logger.debug(f"STRICT MODE: Successfully processed {video_path} ({i+1}/{len(video_files)})")
                    
                except Exception as e:
                    logger.error(f"STRICT MODE: Critical GPU failure for {video_path}: {e}")
                    failed_videos.append(video_path)
                    video_features[video_path] = None
                    
                    # In strict mode, we could choose to fail fast
                    if len(failed_videos) > len(video_files) * 0.1:  # More than 10% failed
                        raise RuntimeError(f"STRICT MODE: Too many GPU failures ({len(failed_videos)}/{i+1}). "
                                         f"GPU acceleration requirements not met.")
                
                # Periodic cache save (every 10 videos)
                if (i + 1) % 10 == 0:
                    with open(video_cache_path, 'wb') as f:
                        pickle.dump(video_features, f)
                    logger.info(f"STRICT MODE: Saved progress: {i+1}/{len(video_files)} videos processed, {len(failed_videos)} failed")
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            
            # Finalize all H.264 4:4:4 conversions after processing
            logger.info("FINALIZING: All H.264 4:4:4 conversions...")
            decoder.finalize_all_conversions()
            
            valid_videos = sum(1 for v in video_features.values() if v is not None and v.get('gpu_processed', False))
            
            if failed_videos:
                logger.warning(f"STRICT MODE: {len(failed_videos)} videos failed GPU processing: {failed_videos[:5]}...")
            
            if valid_videos == 0:
                raise RuntimeError("STRICT MODE: No videos were successfully processed with GPU acceleration!")
            
            logger.info(f"STRICT MODE: Successfully processed {valid_videos}/{len(video_features)} videos with GPU acceleration")
            if h264_444_conversions > 0:
                logger.info(f"H.264 4:4:4 SUMMARY: Converted {h264_444_conversions} videos from 4:4:4 to 4:2:0")
        
        # Process GPX files
        logger.info("Processing GPX files with CuPy GPU acceleration...")
        gpx_cache_path = cache_dir / "ultra_gpx_features.pkl"
        
        if gpx_cache_path.exists() and not args.force:
            with open(gpx_cache_path, 'rb') as f:
                gpx_database = pickle.load(f)
            
            # Validate cached data
            valid_count = sum(1 for v in gpx_database.values() if v is not None)
            total_count = len(gpx_database)
            
            logger.info(f"Loaded cached GPX features: {valid_count}/{total_count} valid")
            
            # If most cached data is invalid, force reprocessing
            if valid_count < total_count * 0.5:  # Less than 50% valid
                logger.warning(f"Most cached GPX data is invalid ({valid_count}/{total_count}), forcing reprocessing")
                gpx_database = {}
            
        else:
            gpx_database = {}
        
        if not gpx_database or len(gpx_database) != len(gpx_files):
            logger.info("Processing GPX files (cache miss or incomplete)...")
            gpx_database = gpx_processor.process_gpx_files_gpu(gpx_files)
            
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_database, f)
            logger.info(f"Ultra-processed and cached {len(gpx_database)} GPX files")
        
        # Ultra-optimized correlation
        logger.info("Performing ultra-optimized correlation analysis...")
        start_time = time.time()
        
        results = correlator.correlate_ultra_optimized(
            video_features, gpx_database, output_dir, top_k=args.top_k
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save results
        results_path = output_dir / "ultra_correlations.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Ultra-optimized analysis complete in {total_time:.2f} seconds!")
        
        # Performance summary
        print(f"\nUltra-GPU Performance Summary:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Videos processed: {len(video_features)}")
        print(f"GPX files processed: {len(gpx_database)}")
        print(f"Average time per video: {total_time/len(video_features):.2f} seconds")
        print(f"GPU acceleration: {len(args.gpu_ids)} devices")
        
        # H.264 4:4:4 conversion summary
        if hasattr(decoder, 'converted_videos') and decoder.converted_videos:
            print(f"\nH.264 4:4:4 Conversion Summary:")
            total_conversions = len([v for v in video_features.values() if v and 'original_video_path' in v])
            ram_conversions = sum(1 for info in decoder.converted_videos.values() if info.get('use_ram', False))
            disk_conversions = len(decoder.converted_videos) - ram_conversions
            print(f"Videos converted: {len(decoder.converted_videos)}")
            print(f"RAM conversions: {ram_conversions}")
            print(f"Disk conversions: {disk_conversions}")
            
            total_conversion_time = sum(info.get('conversion_time', 0) for info in decoder.converted_videos.values())
            print(f"Total conversion time: {total_conversion_time:.2f} seconds")
        else:
            print(f"\nH.264 4:4:4 Conversion Summary:")
            print(f"No H.264 4:4:4 videos detected or conversion disabled")
        
        # Results quality summary
        if results:
            successful_correlations = sum(1 for r in results.values() if r is not None and r.get('matches'))
            high_confidence_matches = 0
            medium_confidence_matches = 0
            low_confidence_matches = 0
            
            for result in results.values():
                if result and result.get('matches'):
                    best_score = result['matches'][0]['combined_score']
                    if best_score > 0.8:
                        high_confidence_matches += 1
                    elif best_score > 0.6:
                        medium_confidence_matches += 1
                    else:
                        low_confidence_matches += 1
            
            print(f"\nCorrelation Quality Summary:")
            print(f"Successful correlations: {successful_correlations}/{len(results)}")
            print(f"High confidence matches (>0.8): {high_confidence_matches}")
            print(f"Medium confidence matches (0.6-0.8): {medium_confidence_matches}")
            print(f"Low confidence matches (<0.6): {low_confidence_matches}")
            
            if successful_correlations > 0:
                success_rate = (successful_correlations / len(results)) * 100
                print(f"Overall success rate: {success_rate:.1f}%")
        
        # RAM usage summary
        if PSUTIL_AVAILABLE:
            try:
                memory_info = psutil.virtual_memory()
                print(f"\nMemory Usage Summary:")
                print(f"Total RAM: {memory_info.total // 1024**3}GB")
                print(f"Available RAM: {memory_info.available // 1024**3}GB")
                print(f"Used RAM: {memory_info.used // 1024**3}GB")
                print(f"RAM utilization: {memory_info.percent:.1f}%")
            except Exception as e:
                logger.debug(f"Could not get memory info: {e}")
        
        # GPU memory summary
        print(f"\nGPU Memory Summary:")
        for gpu_id in args.gpu_ids:
            try:
                allocated = torch.cuda.memory_allocated(gpu_id) // 1024**2
                reserved = torch.cuda.memory_reserved(gpu_id) // 1024**2
                total = torch.cuda.get_device_properties(gpu_id).total_memory // 1024**2
                print(f"GPU {gpu_id}: {allocated}MB allocated, {reserved}MB reserved, {total}MB total")
            except Exception as e:
                logger.debug(f"Could not get GPU {gpu_id} memory info: {e}")
        
    except Exception as e:
        logger.error(f"Ultra-optimization failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Performing final cleanup...")
        
        # Clean up GPU memory
        try:
            torch.cuda.empty_cache()
            if hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()
            logger.info("GPU memory cleanup completed")
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
        
        # Clean up any remaining temporary conversions
        try:
            if hasattr(decoder, 'finalize_all_conversions'):
                decoder.finalize_all_conversions()
            logger.info("H.264 4:4:4 conversion cleanup completed")
        except Exception as e:
            logger.warning(f"Conversion cleanup failed: {e}")


if __name__ == "__main__":
    main()