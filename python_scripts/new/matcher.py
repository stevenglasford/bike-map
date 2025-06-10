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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle
import json
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
import decord
from decord import VideoReader, cpu, gpu

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('correlation.log')
    ]
)
logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Centralized GPU memory management"""
    def __init__(self, devices):
        self.devices = devices
        self.memory_locks = {device: Lock() for device in devices}
        self.memory_stats = {device: {'allocated': 0, 'cached': 0} for device in devices}
    
    def get_available_device(self):
        """Get device with most available memory"""
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        best_device = self.devices[0]
        max_free = 0
        
        for device in self.devices:
            if device.type == 'cuda':
                free_memory = torch.cuda.get_device_properties(device.index).total_memory - torch.cuda.memory_allocated(device.index)
                if free_memory > max_free:
                    max_free = free_memory
                    best_device = device
        
        return best_device
    
    def cleanup_device(self, device):
        """Clean up specific device memory"""
        if device.type == 'cuda':
            with self.memory_locks[device]:
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
                if hasattr(cp, 'get_default_memory_pool'):
                    cp.get_default_memory_pool().free_all_blocks()

class OptimizedVideoDecoder:
    """GPU-accelerated video decoder using decord"""
    def __init__(self, gpu_ids=[0, 1]):
        self.gpu_ids = gpu_ids
        self.decoders = {}
        
        # Initialize decord contexts
        if torch.cuda.is_available():
            for gpu_id in gpu_ids:
                try:
                    ctx = gpu(gpu_id)
                    self.decoders[gpu_id] = ctx
                    logger.info(f"Initialized GPU decoder on GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU decoder on GPU {gpu_id}: {e}")
        
        if not self.decoders:
            self.decoders[0] = cpu()
            logger.warning("Falling back to CPU decoding")
    
    def decode_video_batch(self, video_path, sample_rate=2.0, target_size=(640, 360)):
        """Decode video with GPU acceleration and smart sampling"""
        gpu_id = next(iter(self.decoders.keys()))
        ctx = self.decoders[gpu_id]
        
        try:
            vr = VideoReader(video_path, ctx=ctx)
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps if fps > 0 else 0
            
            # Adaptive sampling
            if duration > 0:
                sample_rate = min(sample_rate, max(0.5, 30 / duration))
            
            frame_interval = max(1, int(fps / sample_rate))
            frame_indices = list(range(0, total_frames, frame_interval))
            
            # Batch decode frames
            if len(frame_indices) > 0:
                frames = vr.get_batch(frame_indices)
                if ctx.device_type == 'gpu':
                    # Convert to torch tensor on GPU
                    frames_tensor = torch.from_numpy(frames.asnumpy()).cuda(gpu_id)
                else:
                    frames_tensor = torch.from_numpy(frames.asnumpy())
                
                # Resize batch
                frames_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0
                frames_resized = F.interpolate(frames_tensor, size=target_size, mode='bilinear', align_corners=False)
                
                return frames_resized, fps, duration, frame_indices
            
        except Exception as e:
            logger.error(f"Error decoding {video_path}: {e}")
            # Fallback to OpenCV
            return self._opencv_fallback(video_path, sample_rate, target_size)
        
        return None, 0, 0, []
    
    def _opencv_fallback(self, video_path, sample_rate, target_size):
        """Fallback to OpenCV decoding"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0, 0, []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        frame_interval = max(1, int(fps / sample_rate))
        frames = []
        frame_indices = []
        
        for i in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_indices.append(i)
        
        cap.release()
        
        if frames:
            frames_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames]).float() / 255.0
            return frames_tensor, fps, duration, frame_indices
        
        return None, 0, 0, []

class OptimizedGPUFeatureExtractor:
    """Optimized GPU feature extraction with mixed precision"""
    def __init__(self, gpu_ids=[0, 1], use_mixed_precision=True):
        self.gpu_ids = gpu_ids
        self.devices = [torch.device(f'cuda:{i}') for i in gpu_ids if torch.cuda.is_available() and i < torch.cuda.device_count()]
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        if not self.devices:
            self.devices = [torch.device('cpu')]
            self.use_mixed_precision = False
        
        # Initialize models with mixed precision
        self.feature_extractors = []
        self.optical_flow_models = []
        
        for device in self.devices:
            # Load pre-trained MobileNetV2 with mixed precision
            try:
                model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
                model = nn.Sequential(*list(model.features.children())[:-1])
                model.eval().to(device)
                
                if self.use_mixed_precision:
                    model = model.half()
                
                self.feature_extractors.append(model)
                
            except Exception as e:
                logger.warning(f"Using simple CNN fallback: {e}")
                model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                ).to(device)
                model.eval()
                
                if self.use_mixed_precision:
                    model = model.half()
                
                self.feature_extractors.append(model)
            
            # Initialize GPU optical flow
            if device.type == 'cuda':
                try:
                    flow_model = self._create_optical_flow_model().to(device)
                    if self.use_mixed_precision:
                        flow_model = flow_model.half()
                    self.optical_flow_models.append(flow_model)
                except Exception as e:
                    logger.warning(f"Failed to create optical flow model: {e}")
                    self.optical_flow_models.append(None)
            else:
                self.optical_flow_models.append(None)
        
        # CUDA streams for parallel processing
        self.streams = []
        if torch.cuda.is_available():
            for device in self.devices:
                if device.type == 'cuda':
                    stream = torch.cuda.Stream(device=device)
                    self.streams.append(stream)
                else:
                    self.streams.append(None)
    
    def _create_optical_flow_model(self):
        """Create lightweight optical flow estimation model"""
        class OpticalFlowNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
                self.conv4 = nn.Conv2d(32, 2, 3, padding=1)
                self.relu = nn.ReLU()
            
            def forward(self, frame1, frame2):
                # Convert RGB to grayscale
                gray1 = 0.299 * frame1[:, 0:1] + 0.587 * frame1[:, 1:2] + 0.114 * frame1[:, 2:3]
                gray2 = 0.299 * frame2[:, 0:1] + 0.587 * frame2[:, 1:2] + 0.114 * frame2[:, 2:3]
                
                # Concatenate frames
                x = torch.cat([gray1, gray2, frame1, frame2], dim=1)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                flow = self.conv4(x)
                return flow
        
        return OpticalFlowNet()
    
    def extract_features_batch(self, frames_tensor, device_idx=0):
        """Extract all features in batched GPU operations"""
        device = self.devices[device_idx]
        model = self.feature_extractors[device_idx]
        flow_model = self.optical_flow_models[device_idx]
        stream = self.streams[device_idx] if self.streams else None
        
        # Move to device with proper precision
        if self.use_mixed_precision and device.type == 'cuda':
            frames_tensor = frames_tensor.half().to(device)
        else:
            frames_tensor = frames_tensor.to(device)
        
        batch_size, num_frames = frames_tensor.shape[:2]
        
        # Initialize feature containers on GPU
        features = {
            'scene_features': [],
            'motion_magnitude': torch.zeros(num_frames, device=device),
            'motion_direction': torch.zeros(num_frames, device=device),
            'acceleration': torch.zeros(num_frames, device=device),
            'jerk': torch.zeros(num_frames, device=device),
            'rotation': torch.zeros(num_frames, device=device),
            'color_histogram': [],
            'edge_density': torch.zeros(num_frames, device=device),
            'optical_flow_complexity': torch.zeros(num_frames, device=device),
            'temporal_gradient': torch.zeros(num_frames, device=device)
        }
        
        with torch.cuda.stream(stream) if stream else torch.no_grad():
            # Batch scene feature extraction
            scene_feats = model(frames_tensor.view(-1, *frames_tensor.shape[2:]))
            if len(scene_feats.shape) == 4:
                scene_feats = F.adaptive_avg_pool2d(scene_feats, (1, 1)).squeeze()
            if len(scene_feats.shape) == 1:
                scene_feats = scene_feats.unsqueeze(0)
            
            features['scene_features'] = scene_feats
            
            # Batch color histogram computation on GPU
            color_hists = self._compute_color_histograms_gpu(frames_tensor)
            features['color_histogram'] = color_hists
            
            # Batch edge detection on GPU
            edge_densities = self._compute_edge_density_gpu(frames_tensor)
            features['edge_density'] = edge_densities
            
            # Batch optical flow and motion features
            if num_frames > 1:
                motion_feats = self._compute_motion_features_gpu(frames_tensor, flow_model)
                for key, values in motion_feats.items():
                    if key in features:
                        features[key] = values
            
            # Batch temporal gradients
            temporal_grads = self._compute_temporal_gradients_gpu(frames_tensor)
            features['temporal_gradient'] = temporal_grads
        
        # Post-process on GPU
        features = self._post_process_features_gpu(features, device)
        
        return features
    
    def _compute_color_histograms_gpu(self, frames_tensor):
        """GPU-accelerated color histogram computation"""
        device = frames_tensor.device
        batch_size, num_frames = frames_tensor.shape[:2]
        
        # Quantize colors for histogram
        quantized = (frames_tensor * 7).long()  # 8 bins per channel
        
        # Compute histograms using scatter_add
        histograms = []
        for i in range(num_frames):
            frame = quantized[i]
            # Flatten spatial dimensions
            h, w = frame.shape[1:]
            frame_flat = frame.permute(1, 2, 0).reshape(-1, 3)
            
            # Compute histogram indices
            hist_indices = frame_flat[:, 0] * 64 + frame_flat[:, 1] * 8 + frame_flat[:, 2]
            
            # Create histogram
            hist = torch.zeros(512, device=device)
            hist.scatter_add_(0, hist_indices, torch.ones_like(hist_indices, dtype=torch.float))
            hist = hist / hist.sum()  # Normalize
            
            histograms.append(hist)
        
        return torch.stack(histograms)
    
    def _compute_edge_density_gpu(self, frames_tensor):
        """GPU-accelerated edge detection"""
        # Convert to grayscale
        gray_frames = 0.299 * frames_tensor[:, :, 0] + 0.587 * frames_tensor[:, :, 1] + 0.114 * frames_tensor[:, :, 2]
        
        # Sobel edge detection on GPU
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=frames_tensor.dtype, device=frames_tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=frames_tensor.dtype, device=frames_tensor.device).view(1, 1, 3, 3)
        
        gray_frames = gray_frames.unsqueeze(1)  # Add channel dimension
        
        edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
        edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3])
        
        return edge_density
    
    def _compute_motion_features_gpu(self, frames_tensor, flow_model):
        """GPU-accelerated motion feature computation"""
        device = frames_tensor.device
        num_frames = frames_tensor.shape[1]
        
        features = {
            'motion_magnitude': torch.zeros(num_frames, device=device),
            'motion_direction': torch.zeros(num_frames, device=device),
            'acceleration': torch.zeros(num_frames, device=device),
            'jerk': torch.zeros(num_frames, device=device),
            'rotation': torch.zeros(num_frames, device=device),
            'optical_flow_complexity': torch.zeros(num_frames, device=device)
        }
        
        if flow_model is None or num_frames < 2:
            return features
        
        # Batch optical flow computation
        prev_frame = frames_tensor[0, 0:1]
        motion_mags = []
        motion_dirs = []
        rotations = []
        complexities = []
        
        for i in range(1, num_frames):
            curr_frame = frames_tensor[0, i:i+1]
            
            # Compute optical flow
            flow = flow_model(prev_frame, curr_frame)
            
            # Extract motion statistics
            magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
            direction = torch.atan2(flow[:, 1], flow[:, 0])
            
            motion_mag = torch.mean(magnitude)
            motion_dir = torch.mean(torch.abs(torch.exp(1j * direction.cpu()).real))  # Approximation
            
            # Rotation estimation on GPU
            h, w = flow.shape[2:]
            y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            x_coords = x_coords - w // 2
            y_coords = y_coords - h // 2
            
            rotation = torch.mean(x_coords * flow[0, 1] - y_coords * flow[0, 0]) / (torch.mean(x_coords**2 + y_coords**2) + 1e-8)
            
            # Flow complexity
            complexity = torch.std(magnitude)
            
            motion_mags.append(motion_mag)
            motion_dirs.append(motion_dir)
            rotations.append(rotation)
            complexities.append(complexity)
            
            prev_frame = curr_frame
        
        # Fill features
        if motion_mags:
            features['motion_magnitude'][1:] = torch.stack(motion_mags)
            features['motion_direction'][1:] = torch.stack(motion_dirs)
            features['rotation'][1:] = torch.stack(rotations)
            features['optical_flow_complexity'][1:] = torch.stack(complexities)
            
            # Compute acceleration and jerk on GPU
            motion_tensor = features['motion_magnitude']
            features['acceleration'][1:] = motion_tensor[1:] - motion_tensor[:-1]
            features['jerk'][2:] = features['acceleration'][2:] - features['acceleration'][1:-1]
        
        return features
    
    def _compute_temporal_gradients_gpu(self, frames_tensor):
        """GPU-accelerated temporal gradient computation"""
        if frames_tensor.shape[1] < 2:
            return torch.zeros(frames_tensor.shape[1], device=frames_tensor.device)
        
        # Convert to grayscale
        gray_frames = 0.299 * frames_tensor[0, :, 0] + 0.587 * frames_tensor[0, :, 1] + 0.114 * frames_tensor[0, :, 2]
        
        # Compute temporal differences
        temporal_diffs = torch.abs(gray_frames[1:] - gray_frames[:-1])
        temporal_grads = torch.mean(temporal_diffs, dim=[1, 2])
        
        # Pad first frame
        result = torch.zeros(frames_tensor.shape[1], device=frames_tensor.device)
        result[1:] = temporal_grads
        
        return result
    
    def _post_process_features_gpu(self, features, device):
        """GPU-accelerated feature post-processing"""
        processed = {}
        
        for key, values in features.items():
            if isinstance(values, torch.Tensor) and values.dim() == 1:
                # Apply smoothing on GPU
                if len(values) > 5:
                    # Simple moving average on GPU
                    kernel_size = min(7, len(values) // 3)
                    if kernel_size > 1:
                        kernel = torch.ones(kernel_size, device=device) / kernel_size
                        values_padded = F.pad(values.unsqueeze(0), (kernel_size//2, kernel_size//2), mode='reflect')
                        smoothed = F.conv1d(values_padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=0)
                        values = smoothed.squeeze()
                
                # Normalize on GPU
                if torch.std(values) > 1e-8:
                    values = (values - torch.mean(values)) / torch.std(values)
                
                processed[key] = values
            else:
                processed[key] = values
        
        return processed

class OptimizedGPXProcessor:
    """GPU-accelerated GPX processing using CuPy"""
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and cp.cuda.is_available()
        if self.use_gpu:
            logger.info("Using CuPy for GPU-accelerated GPX processing")
        else:
            logger.warning("CuPy not available, using CPU for GPX processing")
    
    def parse_gpx_batch(self, gpx_paths, max_workers=None):
        """Parse multiple GPX files with GPU acceleration"""
        if max_workers is None:
            max_workers = min(32, mp.cpu_count())
        
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._parse_single_gpx_cpu, path): path 
                      for path in gpx_paths}
            
            for future in tqdm(as_completed(futures), total=len(gpx_paths), desc="Parsing GPX files"):
                path = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Move to GPU for feature calculation if available
                        if self.use_gpu:
                            result = self._accelerate_features_gpu(result)
                        results[path] = result
                except Exception as e:
                    logger.error(f"Error parsing {path}: {e}")
        
        return results
    
    def _parse_single_gpx_cpu(self, gpx_path):
        """Parse single GPX file (CPU-bound I/O operation)"""
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
                'raw_points': points,
                'start_time': df['timestamp'].iloc[0],
                'end_time': df['timestamp'].iloc[-1],
                'point_count': len(df)
            }
        
        except Exception as e:
            logger.error(f"Error parsing {gpx_path}: {e}")
            return None
    
    def _accelerate_features_gpu(self, gpx_data):
        """Calculate features using GPU acceleration"""
        df = gpx_data['df']
        
        # Convert to GPU arrays
        if self.use_gpu:
            lats = cp.array(df['lat'].values)
            lons = cp.array(df['lon'].values)
            elevations = cp.array(df['elevation'].values)
            timestamps = df['timestamp'].values
        else:
            lats = df['lat'].values
            lons = df['lon'].values
            elevations = df['elevation'].values
            timestamps = df['timestamp'].values
        
        # Calculate time differences
        time_diffs = np.array([(timestamps[i] - timestamps[i-1]).total_seconds() 
                              for i in range(1, len(timestamps))])
        time_diffs = np.insert(time_diffs, 0, 1.0)  # Avoid division by zero
        
        if self.use_gpu:
            time_diffs_gpu = cp.array(time_diffs)
        
        features = {}
        
        # GPU-accelerated distance calculation
        if self.use_gpu:
            distances = self._haversine_gpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            distances = cp.insert(distances, 0, 0)
        else:
            distances = self._haversine_cpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            distances = np.insert(distances, 0, 0)
        
        # Speed calculation
        if self.use_gpu:
            speeds = (distances[1:] * 3600) / time_diffs_gpu[1:]  # mph
            speeds = cp.insert(speeds, 0, 0)
            features['speed'] = cp.asnumpy(speeds)
        else:
            speeds = (distances[1:] * 3600) / time_diffs[1:]
            speeds = np.insert(speeds, 0, 0)
            features['speed'] = speeds
        
        # GPU-accelerated bearing calculation
        if self.use_gpu:
            bearings = self._bearing_gpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            bearings = cp.insert(bearings, 0, 0)
            features['bearing'] = cp.asnumpy(bearings)
        else:
            bearings = self._bearing_cpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            bearings = np.insert(bearings, 0, 0)
            features['bearing'] = bearings
        
        # Additional features
        if self.use_gpu:
            # Elevation change
            elev_changes = cp.diff(elevations)
            elev_changes = cp.insert(elev_changes, 0, 0)
            features['elevation_change'] = cp.asnumpy(elev_changes)
            
            # Acceleration
            speed_diffs = cp.diff(speeds)
            acceleration = speed_diffs / time_diffs_gpu[1:]
            acceleration = cp.insert(acceleration, 0, 0)
            features['acceleration'] = cp.asnumpy(acceleration)
            
            # Bearing changes
            bearing_diffs = cp.diff(bearings)
            # Handle angle wraparound
            bearing_diffs = cp.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = cp.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = cp.abs(bearing_diffs)
            bearing_changes = cp.insert(bearing_changes, 0, 0)
            features['bearing_change'] = cp.asnumpy(bearing_changes)
            
            # Curvature
            curvature = bearing_changes / (distances + 1e-8)
            features['curvature'] = cp.asnumpy(curvature)
            
            # Jerk
            accel_diffs = cp.diff(acceleration)
            jerk = accel_diffs / time_diffs_gpu[2:]
            jerk = cp.insert(jerk, [0, 0], [0, 0])
            features['jerk'] = cp.asnumpy(jerk)
            
        else:
            # CPU fallback
            features['elevation_change'] = np.insert(np.diff(elevations), 0, 0)
            speed_diffs = np.diff(speeds)
            acceleration = speed_diffs / time_diffs[1:]
            features['acceleration'] = np.insert(acceleration, 0, 0)
            
            bearing_diffs = np.diff(bearings)
            bearing_diffs = np.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = np.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = np.abs(bearing_diffs)
            features['bearing_change'] = np.insert(bearing_changes, 0, 0)
            
            curvature = bearing_changes / (distances + 1e-8)
            features['curvature'] = curvature
            
            accel_diffs = np.diff(acceleration)
            jerk = accel_diffs / time_diffs[2:]
            features['jerk'] = np.insert(jerk, [0, 0], [0, 0])
        
        # Apply smoothing (keep on CPU for now due to signal processing complexity)
        for key in features:
            if len(features[key]) > 5:
                # Simple moving average
                window = min(5, len(features[key]) // 3)
                if window > 1:
                    features[key] = np.convolve(features[key], np.ones(window)/window, mode='same')
        
        # Calculate metadata
        duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        if self.use_gpu:
            total_distance = float(cp.sum(distances))
        else:
            total_distance = float(np.sum(distances))
        
        gpx_data.update({
            'features': features,
            'duration': duration,
            'distance': total_distance
        })
        
        return gpx_data
    
    def _haversine_gpu(self, lat1, lon1, lat2, lon2):
        """GPU-accelerated haversine distance"""
        R = 3958.8  # Earth radius in miles
        
        lat1, lon1, lat2, lon2 = [cp.radians(x) for x in [lat1, lon1, lat2, lon2]]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(a))
        
        return R * c
    
    def _haversine_cpu(self, lat1, lon1, lat2, lon2):
        """CPU haversine distance"""
        R = 3958.8
        
        lat1, lon1, lat2, lon2 = [np.radians(x) for x in [lat1, lon1, lat2, lon2]]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _bearing_gpu(self, lat1, lon1, lat2, lon2):
        """GPU-accelerated bearing calculation"""
        lat1, lon1, lat2, lon2 = [cp.radians(x) for x in [lat1, lon1, lat2, lon2]]
        dlon = lon2 - lon1
        
        y = cp.sin(dlon) * cp.cos(lat2)
        x = cp.cos(lat1) * cp.sin(lat2) - cp.sin(lat1) * cp.cos(lat2) * cp.cos(dlon)
        
        return cp.degrees(cp.arctan2(y, x))
    
    def _bearing_cpu(self, lat1, lon1, lat2, lon2):
        """CPU bearing calculation"""
        lat1, lon1, lat2, lon2 = [np.radians(x) for x in [lat1, lon1, lat2, lon2]]
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        return np.degrees(np.arctan2(y, x))

class HighPerformanceCorrelator:
    """High-performance GPU correlator with batched operations"""
    def __init__(self, gpu_ids=[0, 1], use_mixed_precision=True):
        self.devices = [torch.device(f'cuda:{i}') for i in gpu_ids if torch.cuda.is_available() and i < torch.cuda.device_count()]
        self.use_mixed_precision = use_mixed_precision
        
        if not self.devices:
            self.devices = [torch.device('cpu')]
            self.use_mixed_precision = False
        
        self.memory_manager = GPUMemoryManager(self.devices)
        
        # Initialize similarity networks with mixed precision
        self.similarity_networks = []
        for device in self.devices:
            net = self._create_similarity_network().to(device)
            if self.use_mixed_precision and device.type == 'cuda':
                net = net.half()
            self.similarity_networks.append(net)
    
    def _create_similarity_network(self):
        """Optimized similarity network"""
        class FastSimilarityNetwork(nn.Module):
            def __init__(self, input_dim=256):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim * 2, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x1, x2):
                x = torch.cat([x1, x2], dim=-1)
                return self.encoder(x)
        
        return FastSimilarityNetwork()
    
    def correlate_all_optimized(self, video_features_dict, gpx_database, output_dir, top_k=5):
        """Optimized correlation with massive parallelization"""
        results = {}
        
        # Prepare data for batch processing
        valid_videos = {k: v for k, v in video_features_dict.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None}
        
        if not valid_videos or not valid_gpx:
            logger.error("No valid video or GPX data found")
            return results
        
        logger.info(f"Processing {len(valid_videos)} videos against {len(valid_gpx)} GPX files")
        
        # Split work among GPUs
        video_paths = list(valid_videos.keys())
        gpu_assignments = defaultdict(list)
        for i, video_path in enumerate(video_paths):
            gpu_idx = i % len(self.devices)
            gpu_assignments[gpu_idx].append(video_path)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for gpu_idx, videos in gpu_assignments.items():
                future = executor.submit(
                    self._process_batch_on_gpu,
                    videos, valid_videos, valid_gpx, gpu_idx, top_k
                )
                futures.append(future)
            
            for future in as_completed(futures):
                gpu_results = future.result()
                results.update(gpu_results)
        
        # Generate optimized report
        asyncio.run(self._generate_report_async(results, output_dir))
        
        return results
    
    def _process_batch_on_gpu(self, video_paths, video_features, gpx_database, gpu_idx, top_k):
        """Process video batch on specific GPU with maximum efficiency"""
        device = self.devices[gpu_idx]
        similarity_net = self.similarity_networks[gpu_idx]
        
        results = {}
        
        # Pre-process all GPX signatures on GPU
        gpx_signatures = self._preprocess_gpx_signatures(gpx_database, device)
        
        for video_path in tqdm(video_paths, desc=f"GPU {gpu_idx}"):
            try:
                video_features_data = video_features[video_path]
                
                # Create video signature on GPU
                video_sig = self._create_gpu_signature(video_features_data, device)
                
                # Batch correlation against all GPX files
                matches = self._batch_correlate_gpu(
                    video_sig, gpx_signatures, gpx_database, 
                    device, similarity_net, top_k
                )
                
                results[video_path] = {
                    'matches': matches,
                    'video_duration': video_features_data.get('duration', 0)
                }
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results[video_path] = None
            
            # Periodic cleanup
            if len(results) % 10 == 0:
                self.memory_manager.cleanup_device(device)
        
        return results
    
    def _preprocess_gpx_signatures(self, gpx_database, device):
        """Pre-process all GPX signatures on GPU for batch operations"""
        signatures = {}
        
        # Collect all features for batch processing
        all_features = []
        gpx_paths = []
        
        for gpx_path, gpx_data in gpx_database.items():
            if gpx_data and 'features' in gpx_data:
                features = gpx_data['features']
                all_features.append(features)
                gpx_paths.append(gpx_path)
        
        if not all_features:
            return signatures
        
        # Batch create signatures on GPU
        try:
            # Create embeddings for all GPX files at once
            embeddings = []
            for features in all_features:
                embedding = self._create_embedding_vector(features)
                embeddings.append(embedding)
            
            if embeddings:
                # Move all embeddings to GPU at once
                embeddings_tensor = torch.stack([
                    torch.tensor(emb, dtype=torch.float16 if self.use_mixed_precision else torch.float32)
                    for emb in embeddings
                ]).to(device)
                
                # Store signatures with batch-processed embeddings
                for i, gpx_path in enumerate(gpx_paths):
                    sig = self._create_gpu_signature(all_features[i], device, precomputed_embedding=embeddings_tensor[i])
                    signatures[gpx_path] = sig
        
        except Exception as e:
            logger.warning(f"Batch GPX preprocessing failed, falling back to individual: {e}")
            # Fallback to individual processing
            for gpx_path, gpx_data in gpx_database.items():
                if gpx_data and 'features' in gpx_data:
                    sig = self._create_gpu_signature(gpx_data['features'], device)
                    signatures[gpx_path] = sig
        
        return signatures
    
    def _create_gpu_signature(self, features, device, precomputed_embedding=None):
        """Create comprehensive signature on GPU"""
        signature = {}
        
        for key, values in features.items():
            if isinstance(values, (np.ndarray, torch.Tensor)) and len(values) > 0:
                # Convert to GPU tensor
                if isinstance(values, np.ndarray):
                    if values.ndim > 1:
                        continue  # Skip multidimensional arrays
                    tensor = torch.tensor(values, dtype=torch.float16 if self.use_mixed_precision else torch.float32, device=device)
                else:
                    tensor = values.to(device)
                    if self.use_mixed_precision and device.type == 'cuda':
                        tensor = tensor.half()
                
                # Skip empty or invalid tensors
                if tensor.numel() == 0:
                    continue
                
                # Normalize
                if torch.std(tensor) > 1e-8:
                    tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
                
                # FFT on GPU
                if len(tensor) > 1:
                    fft = torch.fft.fft(tensor)
                    fft_mag = torch.abs(fft)[:len(fft)//2]
                    signature[f'{key}_fft'] = fft_mag
                
                # Statistics on GPU
                stats = torch.stack([
                    torch.mean(tensor),
                    torch.std(tensor),
                    torch.min(tensor),
                    torch.max(tensor),
                    torch.quantile(tensor, 0.25),
                    torch.quantile(tensor, 0.5),
                    torch.quantile(tensor, 0.75),
                    torch.sum(torch.abs(torch.diff(tensor))) if len(tensor) > 1 else torch.tensor(0.0, device=device),
                    torch.tensor(float(len(tensor)), device=device)
                ])
                signature[f'{key}_stats'] = stats
                
                # Downsampled signal
                if len(tensor) > 100:
                    indices = torch.linspace(0, len(tensor)-1, 100, device=device).long()
                    downsampled = tensor[indices]
                else:
                    downsampled = tensor
                signature[f'{key}_signal'] = downsampled
        
        # Create embedding vector
        if precomputed_embedding is not None:
            signature['embedding'] = precomputed_embedding
        else:
            embedding = self._create_embedding_vector(features)
            if embedding is not None:
                embedding_tensor = torch.tensor(
                    embedding, 
                    dtype=torch.float16 if self.use_mixed_precision else torch.float32,
                    device=device
                )
                signature['embedding'] = embedding_tensor
        
        return signature
    
    def _create_embedding_vector(self, features):
        """Create fixed-size embedding vector"""
        embedding_parts = []
        
        for key in ['speed', 'motion_magnitude', 'acceleration']:
            if key in features:
                values = features[key]
                if isinstance(values, (np.ndarray, torch.Tensor)) and len(values) > 0:
                    if isinstance(values, torch.Tensor):
                        values = values.cpu().numpy()
                    
                    # Basic statistics
                    if len(values) > 0 and np.std(values) > 1e-8:
                        stats = np.array([
                            np.mean(values),
                            np.std(values),
                            np.min(values),
                            np.max(values),
                            np.percentile(values, 25),
                            np.percentile(values, 50),
                            np.percentile(values, 75)
                        ])
                        embedding_parts.append(stats)
        
        if embedding_parts:
            embedding = np.concatenate(embedding_parts)
            # Pad or truncate to fixed size
            if len(embedding) < 256:
                embedding = np.pad(embedding, (0, 256 - len(embedding)))
            else:
                embedding = embedding[:256]
            return embedding
        
        return None
    
    def _batch_correlate_gpu(self, video_sig, gpx_signatures, gpx_database, device, similarity_net, top_k):
        """Batch correlation on GPU for maximum efficiency"""
        if not gpx_signatures:
            return []
        
        # Prepare batch data
        gpx_paths = list(gpx_signatures.keys())
        
        # Batch neural similarity if possible
        neural_scores = {}
        if 'embedding' in video_sig:
            video_emb = video_sig['embedding']
            gpx_embeddings = []
            valid_paths = []
            
            for path in gpx_paths:
                sig = gpx_signatures[path]
                if 'embedding' in sig:
                    gpx_embeddings.append(sig['embedding'])
                    valid_paths.append(path)
            
            if gpx_embeddings:
                # Batch neural similarity computation
                with torch.no_grad():
                    gpx_emb_batch = torch.stack(gpx_embeddings)
                    video_emb_batch = video_emb.unsqueeze(0).repeat(len(gpx_embeddings), 1)
                    
                    if self.use_mixed_precision and device.type == 'cuda':
                        gpx_emb_batch = gpx_emb_batch.half()
                        video_emb_batch = video_emb_batch.half()
                    
                    # Batch inference
                    similarities = similarity_net(video_emb_batch, gpx_emb_batch).squeeze()
                    
                    for i, path in enumerate(valid_paths):
                        neural_scores[path] = similarities[i].item() if similarities.dim() > 0 else similarities.item()
        
        # Calculate other correlation scores
        candidates = []
        for gpx_path in gpx_paths:
            gpx_sig = gpx_signatures[gpx_path]
            gpx_data = gpx_database[gpx_path]
            
            scores = {}
            
            # FFT correlation on GPU
            fft_score = self._batch_fft_correlation(video_sig, gpx_sig)
            if fft_score > 0:
                scores['fft'] = fft_score
            
            # Statistical similarity on GPU
            stats_score = self._batch_statistical_similarity(video_sig, gpx_sig)
            if stats_score > 0:
                scores['stats'] = stats_score
            
            # Neural score
            if gpx_path in neural_scores:
                scores['neural'] = neural_scores[gpx_path]
            
            # DTW similarity (simplified for speed)
            dtw_score = self._fast_dtw_similarity(video_sig, gpx_sig)
            if dtw_score > 0:
                scores['dtw'] = dtw_score
            
            if scores:
                # Adaptive weight combination
                if len(scores) >= 3:
                    weights = {'fft': 0.3, 'stats': 0.2, 'neural': 0.3, 'dtw': 0.2}
                else:
                    weights = {k: 1.0/len(scores) for k in scores}
                
                combined_score = sum(weights.get(k, 0) * scores.get(k, 0) for k in scores)
                
                candidates.append({
                    'path': gpx_path,
                    'combined_score': combined_score,
                    'scores': scores,
                    'distance': gpx_data.get('distance', 0),
                    'duration': gpx_data.get('duration', 0)
                })
        
        # Sort and return top candidates
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:top_k]
    
    def _batch_fft_correlation(self, sig1, sig2):
        """Batch FFT correlation on GPU"""
        scores = []
        
        for key in sig1:
            if key.endswith('_fft') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                if len(s1) > 0 and len(s2) > 0:
                    min_len = min(len(s1), len(s2))
                    s1_norm = s1[:min_len]
                    s2_norm = s2[:min_len]
                    
                    # Normalized correlation on GPU
                    corr = torch.sum(s1_norm * s2_norm) / (torch.norm(s1_norm) * torch.norm(s2_norm) + 1e-8)
                    scores.append(abs(corr.item()))
        
        return np.mean(scores) if scores else 0.0
    
    def _batch_statistical_similarity(self, sig1, sig2):
        """Batch statistical similarity on GPU"""
        scores = []
        
        for key in sig1:
            if key.endswith('_stats') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                # Euclidean distance in stats space
                dist = torch.norm(s1 - s2)
                similarity = 1.0 / (1.0 + dist.item())
                scores.append(similarity)
        
        return np.mean(scores) if scores else 0.0
    
    def _fast_dtw_similarity(self, sig1, sig2, max_len=50):
        """Fast DTW approximation on GPU"""
        scores = []
        
        for key in sig1:
            if key.endswith('_signal') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                # Downsample for speed
                if len(s1) > max_len:
                    indices = torch.linspace(0, len(s1)-1, max_len, device=s1.device).long()
                    s1 = s1[indices]
                if len(s2) > max_len:
                    indices = torch.linspace(0, len(s2)-1, max_len, device=s2.device).long()
                    s2 = s2[indices]
                
                if len(s1) > 0 and len(s2) > 0:
                    # Simple correlation as DTW approximation
                    s1_norm = (s1 - torch.mean(s1)) / (torch.std(s1) + 1e-8)
                    s2_norm = (s2 - torch.mean(s2)) / (torch.std(s2) + 1e-8)
                    
                    corr = F.conv1d(
                        s1_norm.unsqueeze(0).unsqueeze(0),
                        s2_norm.flip(0).unsqueeze(0).unsqueeze(0),
                        padding=len(s2_norm)-1
                    )
                    
                    max_corr = torch.max(corr).item()
                    similarity = max_corr / (len(s1_norm) * len(s2_norm))**0.5
                    scores.append(abs(similarity))
        
        return np.mean(scores) if scores else 0.0
    
    async def _generate_report_async(self, results, output_dir):
        """Generate report asynchronously"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate statistics
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(results),
            'successful_correlations': 0,
            'failed_correlations': 0,
            'score_distribution': [],
            'confidence_levels': {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
            'detailed_results': []
        }
        
        for video_path, result in results.items():
            if result is None or not result.get('matches'):
                report['failed_correlations'] += 1
                continue
            
            report['successful_correlations'] += 1
            best_match = result['matches'][0]
            score = best_match['combined_score']
            report['score_distribution'].append(score)
            
            # Confidence levels
            if score > 0.4:
                report['confidence_levels']['high'] += 1
            elif score > 0.2:
                report['confidence_levels']['medium'] += 1
            elif score > 0.1:
                report['confidence_levels']['low'] += 1
            else:
                report['confidence_levels']['very_low'] += 1
            
            report['detailed_results'].append({
                'video': str(video_path),
                'best_match': str(best_match['path']),
                'score': score,
                'sub_scores': best_match.get('scores', {}),
                'all_matches': [{'gpx': str(m['path']), 'score': m['combined_score']} 
                               for m in result['matches'][:5]]
            })
        
        # Calculate statistics
        if report['score_distribution']:
            scores = np.array(report['score_distribution'])
            report['statistics'] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'median_score': float(np.median(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores))
            }
        
        # Save reports asynchronously
        async with aiofiles.open(output_path / 'accuracy_report.json', 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        # Generate summary
        summary = f"""Video-GPX Correlation Performance Report
================================================
Generated: {report['timestamp']}
Total Videos: {report['total_videos']}
Successful: {report['successful_correlations']}
Failed: {report['failed_correlations']}

Score Statistics:
  Mean: {report.get('statistics', {}).get('mean_score', 0):.4f}
  Median: {report.get('statistics', {}).get('median_score', 0):.4f}
  Std: {report.get('statistics', {}).get('std_score', 0):.4f}

Confidence Distribution:
  High (>0.4): {report['confidence_levels']['high']} ({report['confidence_levels']['high']/report['total_videos']*100:.1f}%)
  Medium (0.2-0.4): {report['confidence_levels']['medium']} ({report['confidence_levels']['medium']/report['total_videos']*100:.1f}%)
  Low (0.1-0.2): {report['confidence_levels']['low']} ({report['confidence_levels']['low']/report['total_videos']*100:.1f}%)
  Very Low (<0.1): {report['confidence_levels']['very_low']} ({report['confidence_levels']['very_low']/report['total_videos']*100:.1f}%)
"""
        
        async with aiofiles.open(output_path / 'summary.txt', 'w') as f:
            await f.write(summary)
        
        logger.info(f"Report generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Multi-GPU Video-GPX Correlation System")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing videos and GPX files")
    parser.add_argument("-o", "--output", default="./correlation_results", help="Output directory")
    parser.add_argument("-c", "--cache", default="./cache", help="Cache directory")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top K matches per video")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], help="GPU IDs to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--force", action='store_true', help="Force reprocessing")
    parser.add_argument("--no_mixed_precision", action='store_true', help="Disable mixed precision")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)
    
    use_mixed_precision = not args.no_mixed_precision and torch.cuda.is_available()
    
    # Find files
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
    video_files = list(dict.fromkeys(video_files))  # Remove duplicates
    
    gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
    gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
    gpx_files = list(dict.fromkeys(gpx_files))
    
    logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
    
    if not video_files or not gpx_files:
        logger.error("No videos or GPX files found!")
        return
    
    # Initialize optimized components
    decoder = OptimizedVideoDecoder(gpu_ids=args.gpu_ids)
    feature_extractor = OptimizedGPUFeatureExtractor(gpu_ids=args.gpu_ids, use_mixed_precision=use_mixed_precision)
    gpx_processor = OptimizedGPXProcessor(use_gpu=True)
    correlator = HighPerformanceCorrelator(gpu_ids=args.gpu_ids, use_mixed_precision=use_mixed_precision)
    
    # Process videos with optimized pipeline
    logger.info("Processing videos with optimized GPU pipeline...")
    video_cache_path = cache_dir / "optimized_video_features.pkl"
    
    if video_cache_path.exists() and not args.force:
        with open(video_cache_path, 'rb') as f:
            video_features = pickle.load(f)
        logger.info(f"Loaded cached video features for {len(video_features)} videos")
    else:
        video_features = {}
        
        # Process videos in parallel batches
        with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
            futures = []
            
            for i, video_path in enumerate(video_files):
                gpu_idx = i % len(args.gpu_ids)
                future = executor.submit(process_single_video, video_path, decoder, feature_extractor, gpu_idx)
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
        logger.info(f"Processed and cached {len(video_features)} videos")
    
    # Process GPX files
    logger.info("Processing GPX files with GPU acceleration...")
    gpx_cache_path = cache_dir / "optimized_gpx_features.pkl"
    
    if gpx_cache_path.exists() and not args.force:
        with open(gpx_cache_path, 'rb') as f:
            gpx_database = pickle.load(f)
        logger.info(f"Loaded cached GPX features for {len(gpx_database)} files")
    else:
        gpx_database = gpx_processor.parse_gpx_batch(gpx_files)
        
        with open(gpx_cache_path, 'wb') as f:
            pickle.dump(gpx_database, f)
        logger.info(f"Processed and cached {len(gpx_database)} GPX files")
    
    # Perform optimized correlation
    logger.info("Performing optimized correlation analysis...")
    start_time = time.time()
    
    results = correlator.correlate_all_optimized(
        video_features, gpx_database, output_dir, top_k=args.top_k
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
    results_path = output_dir / "optimized_correlations.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Optimized analysis complete in {total_time:.2f} seconds!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print performance summary
    print(f"\nPerformance Summary:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Videos processed: {len(video_features)}")
    print(f"GPX files processed: {len(gpx_database)}")
    print(f"Average time per video: {total_time/len(video_features):.2f} seconds")

def process_single_video(video_path, decoder, feature_extractor, gpu_idx):
    """Process a single video with optimized pipeline"""
    try:
        # Decode video
        frames_tensor, fps, duration, frame_indices = decoder.decode_video_batch(video_path)
        
        if frames_tensor is None:
            return None
        
        # Extract features
        features = feature_extractor.extract_features_batch(frames_tensor, gpu_idx)
        
        # Convert GPU tensors to numpy for storage
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.cpu().numpy()
        
        features['duration'] = duration
        features['fps'] = fps
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        return None

if __name__ == "__main__":
    main()
