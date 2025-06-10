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
import av
import tempfile
import shutil

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

class GPUEnforcer:
    """Strict GPU enforcement and validation"""
    
    @staticmethod
    def ensure_cuda_available():
        """Ensure CUDA is available and working"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! GPU acceleration required.")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No CUDA devices found! GPU acceleration required.")
        
        logger.info(f"CUDA available with {device_count} devices")
        
        # Test basic GPU operations
        try:
            test_tensor = torch.randn(100, 100, device='cuda:0')
            result = torch.mm(test_tensor, test_tensor)
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"GPU tensor operations failed: {e}")
    
    @staticmethod
    def ensure_cupy_available():
        """Ensure CuPy is available and working"""
        if not cp.cuda.is_available():
            raise RuntimeError("CuPy CUDA is not available! GPU acceleration required.")
        
        try:
            test_array = cp.random.randn(100, 100)
            result = cp.dot(test_array, test_array)
            del test_array, result
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            raise RuntimeError(f"CuPy operations failed: {e}")
    
    @staticmethod
    def check_ffmpeg_gpu():
        """Check FFmpeg GPU capabilities"""
        try:
            # Check for NVDEC support
            result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                  capture_output=True, text=True, check=True)
            if 'cuda' not in result.stdout:
                raise RuntimeError("FFmpeg CUDA support not found!")
            
            # Check for available encoders
            result = subprocess.run(['ffmpeg', '-encoders'], 
                                  capture_output=True, text=True, check=True)
            if 'h264_nvenc' not in result.stdout:
                logger.warning("h264_nvenc not available")
            
            logger.info("FFmpeg GPU support verified")
            return True
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg GPU check failed: {e}")

class UltraOptimizedFFmpegDecoder:
    """Ultra-optimized FFmpeg-based GPU decoder"""
    
    def __init__(self, gpu_ids=[0, 1]):
        GPUEnforcer.ensure_cuda_available()
        GPUEnforcer.check_ffmpeg_gpu()
        
        self.gpu_ids = gpu_ids
        self.temp_dirs = {}
        
        # Initialize GPU contexts
        for gpu_id in gpu_ids:
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                self.temp_dirs[gpu_id] = tempfile.mkdtemp(prefix=f'gpu_{gpu_id}_')
        
        logger.info(f"Initialized FFmpeg decoder for GPUs: {gpu_ids}")
    
    def decode_video_gpu_batch(self, video_path, sample_rate=2.0, target_size=(640, 360), gpu_id=0):
        """Decode video using FFmpeg GPU acceleration with zero-copy"""
        
        if gpu_id not in self.temp_dirs:
            raise RuntimeError(f"GPU {gpu_id} not initialized")
        
        temp_dir = self.temp_dirs[gpu_id]
        
        try:
            # Get video info first
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', 
                '-show_streams', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise RuntimeError("No video stream found")
            
            fps = eval(video_stream['r_frame_rate'])
            duration = float(info['format']['duration'])
            
            # Calculate sampling parameters
            frame_interval = max(1, int(fps / sample_rate))
            
            # Create GPU-accelerated FFmpeg pipeline
            # Use hardware decoder -> GPU processing -> direct tensor output
            output_pattern = os.path.join(temp_dir, 'frame_%06d.rgb')
            
            decode_cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(gpu_id),
                '-c:v', 'h264_cuvid',  # GPU decoder
                '-i', video_path,
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]},hwdownload,format=rgb24',
                '-r', str(sample_rate),
                '-f', 'image2',
                '-pix_fmt', 'rgb24',
                output_pattern
            ]
            
            # Execute with error checking
            process = subprocess.run(decode_cmd, capture_output=True, text=True)
            if process.returncode != 0:
                # Fallback to software decode if hardware fails
                logger.warning(f"Hardware decode failed, using software: {process.stderr}")
                return self._software_decode_fallback(video_path, sample_rate, target_size, gpu_id)
            
            # Load frames directly to GPU
            frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.rgb')))
            if not frame_files:
                raise RuntimeError("No frames decoded")
            
            frames_gpu = self._load_frames_to_gpu(frame_files, target_size, gpu_id)
            
            # Cleanup
            for f in frame_files:
                os.remove(f)
            
            frame_indices = list(range(0, len(frames_gpu), frame_interval))
            
            return frames_gpu, fps, duration, frame_indices
            
        except Exception as e:
            logger.error(f"GPU decode failed for {video_path}: {e}")
            # Try software fallback as last resort
            return self._software_decode_fallback(video_path, sample_rate, target_size, gpu_id)
    
    def _load_frames_to_gpu(self, frame_files, target_size, gpu_id):
        """Load raw RGB frames directly to GPU memory"""
        device = torch.device(f'cuda:{gpu_id}')
        frame_list = []
        
        width, height = target_size
        frame_size = width * height * 3  # RGB
        
        for frame_file in frame_files:
            try:
                # Read raw RGB data
                with open(frame_file, 'rb') as f:
                    raw_data = f.read()
                
                if len(raw_data) != frame_size:
                    logger.warning(f"Frame size mismatch: {len(raw_data)} vs {frame_size}")
                    continue
                
                # Convert to numpy then GPU tensor in one operation
                frame_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                frame_tensor = torch.from_numpy(frame_np).to(device, dtype=torch.float32) / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW format
                
                frame_list.append(frame_tensor)
                
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_file}: {e}")
                continue
        
        if not frame_list:
            raise RuntimeError("No valid frames loaded")
        
        # Stack all frames into batch tensor on GPU
        frames_batch = torch.stack(frame_list).unsqueeze(0)  # (1, N, C, H, W)
        
        return frames_batch
    
    def _software_decode_fallback(self, video_path, sample_rate, target_size, gpu_id):
        """Software decode fallback that still uses GPU for processing"""
        logger.warning("Using software decode fallback")
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Use PyAV for software decode
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        fps = float(video_stream.average_rate)
        duration = float(container.duration) / av.time_base if container.duration else 0
        
        frame_interval = max(1, int(fps / sample_rate))
        frames = []
        
        for i, frame in enumerate(container.decode(video_stream)):
            if i % frame_interval == 0:
                # Convert frame to numpy
                frame_np = frame.to_ndarray(format='rgb24')
                
                # Resize using CV2 (faster than PIL)
                frame_resized = cv2.resize(frame_np, target_size)
                
                # Convert to tensor and move to GPU immediately
                frame_tensor = torch.from_numpy(frame_resized).to(device, dtype=torch.float32) / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW
                
                frames.append(frame_tensor)
        
        container.close()
        
        if not frames:
            raise RuntimeError("No frames decoded in fallback")
        
        # Stack to batch tensor
        frames_batch = torch.stack(frames).unsqueeze(0)
        frame_indices = list(range(len(frames)))
        
        return frames_batch, fps, duration, frame_indices
    
    def __del__(self):
        """Cleanup temp directories"""
        for temp_dir in self.temp_dirs.values():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

class MaxGPUFeatureExtractor:
    """Maximum GPU utilization feature extractor"""
    
    def __init__(self, gpu_ids=[0, 1]):
        GPUEnforcer.ensure_cuda_available()
        
        self.gpu_ids = gpu_ids
        self.devices = [torch.device(f'cuda:{i}') for i in gpu_ids 
                       if i < torch.cuda.device_count()]
        
        if not self.devices:
            raise RuntimeError("No valid GPU devices for feature extraction!")
        
        # Force mixed precision for maximum performance
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize models on all GPUs
        self.feature_extractors = []
        self.optical_flow_models = []
        
        for device in self.devices:
            # Load optimized feature extractor
            model = self._create_optimized_feature_network().to(device)
            model.eval()
            self.feature_extractors.append(model)
            
            # Optical flow model
            flow_model = self._create_gpu_optical_flow().to(device)
            flow_model.eval()
            self.optical_flow_models.append(flow_model)
        
        # CUDA streams for maximum parallelism
        self.streams = [torch.cuda.Stream(device=device) for device in self.devices]
        
        logger.info(f"MaxGPU Feature Extractor initialized on {len(self.devices)} devices")
    
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
        """Extract all features using maximum GPU acceleration"""
        if device_idx >= len(self.devices):
            raise RuntimeError(f"Invalid device index {device_idx}")
        
        device = self.devices[device_idx]
        model = self.feature_extractors[device_idx]
        flow_model = self.optical_flow_models[device_idx]
        stream = self.streams[device_idx]
        
        # Ensure tensor is on correct device
        if frames_tensor.device != device:
            frames_tensor = frames_tensor.to(device, non_blocking=True)
        
        batch_size, num_frames = frames_tensor.shape[:2]
        
        with torch.cuda.stream(stream):
            with torch.cuda.amp.autocast():
                features = self._extract_comprehensive_features(
                    frames_tensor, model, flow_model, device
                )
        
        # Synchronize stream
        stream.synchronize()
        
        return features
    
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
        GPUEnforcer.ensure_cupy_available()
        
        # Force GPU memory pool settings for maximum performance
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=None)  # Remove memory limit
        
        # Set memory growth to prevent fragmentation
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.set_limit(size=None)
        
        logger.info("CuPy GPX processor initialized with unlimited memory")
    
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
        
        # Time differences in seconds
        timestamps = df['timestamp'].values
        time_diffs = cp.array([
            (timestamps[i] - timestamps[i-1]).total_seconds() 
            for i in range(1, len(timestamps))
        ], dtype=cp.float64)
        time_diffs = cp.concatenate([cp.array([1.0]), time_diffs])  # Avoid division by zero
        
        features = {}
        
        # All distance calculations on GPU
        distances = self._haversine_distance_gpu(lats_gpu[:-1], lons_gpu[:-1], 
                                                lats_gpu[1:], lons_gpu[1:])
        distances = cp.concatenate([cp.array([0.0]), distances])
        
        # Speed (miles per hour)
        speeds = (distances[1:] * 3600) / time_diffs[1:]
        speeds = cp.concatenate([cp.array([0.0]), speeds])
        
        # Bearings
        bearings = self._compute_bearings_gpu(lats_gpu[:-1], lons_gpu[:-1], 
                                            lats_gpu[1:], lons_gpu[1:])
        bearings = cp.concatenate([cp.array([0.0]), bearings])
        
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
        
        # Add metadata
        duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
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
        
        # Acceleration
        speed_diffs = cp.diff(speeds)
        acceleration = speed_diffs / time_diffs[1:]
        acceleration = cp.concatenate([cp.array([0.0]), acceleration])
        
        # Jerk
        accel_diffs = cp.diff(acceleration)
        jerk = accel_diffs / time_diffs[2:]
        jerk = cp.concatenate([cp.array([0.0, 0.0]), jerk])
        
        # Elevation change rate
        elev_diffs = cp.diff(elevations)
        elev_change_rate = elev_diffs / time_diffs[1:]
        elev_change_rate = cp.concatenate([cp.array([0.0]), elev_change_rate])
        
        # Bearing changes (handle wraparound)
        bearing_diffs = cp.diff(bearings)
        bearing_diffs = cp.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
        bearing_diffs = cp.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
        bearing_changes = cp.abs(bearing_diffs)
        bearing_changes = cp.concatenate([cp.array([0.0]), bearing_changes])
        
        # Curvature
        curvature = bearing_changes / (distances + 1e-8)
        
        # Turning rate
        turning_rate = bearing_changes / time_diffs[1:]
        turning_rate = cp.concatenate([cp.array([0.0]), turning_rate])
        
        # Speed consistency (local variance)
        speed_consistency = cp.zeros_like(speeds)
        window_size = 5
        for i in range(window_size, n):
            window_speeds = speeds[i-window_size:i+1]
            speed_consistency[i] = 1.0 / (1.0 + cp.var(window_speeds))
        
        return {
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
        GPUEnforcer.ensure_cuda_available()
        
        self.gpu_ids = gpu_ids
        self.devices = [torch.device(f'cuda:{i}') for i in gpu_ids 
                       if i < torch.cuda.device_count()]
        
        if not self.devices:
            raise RuntimeError("No GPU devices available for correlation!")
        
        # Initialize neural networks on all GPUs
        self.similarity_networks = []
        for device in self.devices:
            net = self._create_advanced_similarity_network().to(device)
            net.eval()
            self.similarity_networks.append(net)
        
        # Create CUDA streams for maximum parallelism
        self.streams = [torch.cuda.Stream(device=device) for device in self.devices]
        
        logger.info(f"Ultra-high performance correlator initialized on {len(self.devices)} GPUs")
    
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
        
        valid_videos = {k: v for k, v in video_features_dict.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None}
        
        if not valid_videos or not valid_gpx:
            raise RuntimeError("No valid video or GPX data for correlation!")
        
        logger.info(f"Ultra-optimized correlation: {len(valid_videos)} videos × {len(valid_gpx)} GPX files")
        
        # Pre-process all data to GPU tensors
        video_tensors = self._preprocess_video_features_to_gpu(valid_videos)
        gpx_tensors = self._preprocess_gpx_features_to_gpu(valid_gpx)
        
        # Distribute work across GPUs
        results = {}
        video_paths = list(valid_videos.keys())
        gpu_assignments = defaultdict(list)
        
        for i, video_path in enumerate(video_paths):
            gpu_idx = i % len(self.devices)
            gpu_assignments[gpu_idx].append(video_path)
        
        # Process in parallel across all GPUs
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for gpu_idx, assigned_videos in gpu_assignments.items():
                future = executor.submit(
                    self._process_gpu_batch_ultra,
                    assigned_videos, video_tensors, gpx_tensors, 
                    valid_gpx, gpu_idx, top_k
                )
                futures.append(future)
            
            for future in as_completed(futures):
                gpu_results = future.result()
                results.update(gpu_results)
        
        # Generate comprehensive report
        asyncio.run(self._generate_ultra_report(results, output_dir))
        
        return results
    
    def _preprocess_video_features_to_gpu(self, video_features_dict):
        """Preprocess all video features to GPU tensors"""
        video_tensors = {}
        
        for video_path, features in video_features_dict.items():
            if features is None:
                continue
                
            try:
                # Create comprehensive feature vector
                feature_vector = self._create_video_feature_vector(features)
                if feature_vector is not None:
                    video_tensors[video_path] = feature_vector
            except Exception as e:
                logger.warning(f"Failed to preprocess video {video_path}: {e}")
        
        return video_tensors
    
    def _preprocess_gpx_features_to_gpu(self, gpx_database):
        """Preprocess all GPX features to GPU tensors"""
        gpx_tensors = {}
        
        for gpx_path, gpx_data in gpx_database.items():
            if gpx_data is None or 'features' not in gpx_data:
                continue
                
            try:
                # Create comprehensive feature vector
                feature_vector = self._create_gpx_feature_vector(gpx_data['features'])
                if feature_vector is not None:
                    gpx_tensors[gpx_path] = feature_vector
            except Exception as e:
                logger.warning(f"Failed to preprocess GPX {gpx_path}: {e}")
        
        return gpx_tensors
    
    def _create_video_feature_vector(self, features):
        """Create comprehensive video feature vector"""
        components = []
        
        # Scene features
        if 'scene_features' in features:
            scene_feat = features['scene_features']
            if isinstance(scene_feat, np.ndarray):
                if scene_feat.ndim == 2:
                    scene_feat = np.mean(scene_feat, axis=0)  # Average over time
                components.append(scene_feat.flatten())
        
        # Motion features
        motion_keys = ['motion_magnitude', 'acceleration', 'jerk', 'rotation']
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
        
        # Color features
        if 'color_histograms' in features:
            color_hist = features['color_histograms']
            if isinstance(color_hist, np.ndarray):
                if color_hist.ndim == 2:
                    color_hist = np.mean(color_hist, axis=0)  # Average over time
                components.append(color_hist.flatten()[:64])  # Limit size
        
        # Edge features
        if 'edge_density' in features:
            edge_vals = features['edge_density']
            if isinstance(edge_vals, np.ndarray) and len(edge_vals) > 0:
                edge_stats = np.array([
                    np.mean(edge_vals), np.std(edge_vals), np.max(edge_vals)
                ])
                components.append(edge_stats)
        
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

def main():
    """Main execution with strict GPU enforcement"""
    parser = argparse.ArgumentParser(description="Ultra-Optimized Multi-GPU Video-GPX Correlation")
    parser.add_argument("-d", "--directory", required=True, help="Directory with videos and GPX files")
    parser.add_argument("-o", "--output", default="./gpu_correlation_results", help="Output directory")
    parser.add_argument("-c", "--cache", default="./gpu_cache", help="Cache directory")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top K matches per video")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], help="GPU IDs to use")
    parser.add_argument("--force", action='store_true', help="Force reprocessing")
    
    args = parser.parse_args()
    
    # Strict GPU validation
    try:
        GPUEnforcer.ensure_cuda_available()
        GPUEnforcer.ensure_cupy_available()
        GPUEnforcer.check_ffmpeg_gpu()
    except RuntimeError as e:
        logger.error(f"GPU requirements not met: {e}")
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
    
    if not video_files or not gpx_files:
        raise RuntimeError("No videos or GPX files found!")
    
    # Initialize ultra-optimized components
    decoder = UltraOptimizedFFmpegDecoder(gpu_ids=args.gpu_ids)
    feature_extractor = MaxGPUFeatureExtractor(gpu_ids=args.gpu_ids)
    gpx_processor = CuPyGPXProcessor()
    correlator = UltraHighPerformanceCorrelator(gpu_ids=args.gpu_ids)
    
    try:
        # Process videos
        logger.info("Processing videos with ultra-optimized GPU pipeline...")
        video_cache_path = cache_dir / "ultra_video_features.pkl"
        
        if video_cache_path.exists() and not args.force:
            with open(video_cache_path, 'rb') as f:
                video_features = pickle.load(f)
            logger.info(f"Loaded cached video features for {len(video_features)} videos")
        else:
            video_features = {}
            
            # Process videos in parallel
            with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
                futures = []
                
                for i, video_path in enumerate(video_files):
                    gpu_idx = i % len(args.gpu_ids)
                    future = executor.submit(
                        process_video_ultra_optimized, 
                        video_path, decoder, feature_extractor, gpu_idx
                    )
                    futures.append((video_path, future))
                
                for video_path, future in tqdm(futures, desc="Processing videos"):
                    try:
                        features = future.result()
                        video_features[video_path] = features
                    except Exception as e:
                        logger.error(f"Error processing {video_path}: {e}")
                        video_features[video_path] = None
            
            # Cache results
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            logger.info(f"Ultra-processed and cached {len(video_features)} videos")
        
        # Process GPX files
        logger.info("Processing GPX files with CuPy GPU acceleration...")
        gpx_cache_path = cache_dir / "ultra_gpx_features.pkl"
        
        if gpx_cache_path.exists() and not args.force:
            with open(gpx_cache_path, 'rb') as f:
                gpx_database = pickle.load(f)
            logger.info(f"Loaded cached GPX features for {len(gpx_database)} files")
        else:
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
        
    except Exception as e:
        logger.error(f"Ultra-optimization failed: {e}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        if hasattr(cp, 'get_default_memory_pool'):
            cp.get_default_memory_pool().free_all_blocks()

def process_video_ultra_optimized(video_path, decoder, feature_extractor, gpu_idx):
    """Process single video with ultra optimization"""
    try:
        # Decode with GPU acceleration
        frames_tensor, fps, duration, frame_indices = decoder.decode_video_gpu_batch(video_path, gpu_id=gpu_idx)
        
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
        
        return features
        
    except Exception as e:
        logger.error(f"Ultra-optimization failed for {video_path}: {e}")
        raise

if __na