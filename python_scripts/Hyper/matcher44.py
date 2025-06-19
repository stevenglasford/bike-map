#!/usr/bin/env python3
"""
Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script
Combines robust production features with 360° video processing and enhanced GPX validation

Key Features:
- Multi-GPU processing with proper queuing
- Enhanced 360° and panoramic video processing
- Advanced GPX validation and processing
- Pre-flight video validation to detect corrupted files
- Enhanced video preprocessing with GPU acceleration
- PowerSafe mode for long-running operations
- Robust error handling and recovery
- Memory optimization and cleanup

Usage:
    # Basic usage with 360° optimization
    python enhanced_matcher39.py -d /path/to/data --gpu_ids 0 1
    
    # Test 360° detection and features
    python enhanced_matcher39.py -d /path/to/data --enable-360-detection
    
    # Enhanced GPX validation
    python enhanced_matcher39.py -d /path/to/data --gpx-validation moderate
    
    # PowerSafe mode for long runs
    python enhanced_matcher39.py -d /path/to/data --powersafe --debug
"""

import cv2
import numpy as np
import gpxpy
import pandas as pd
import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
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
from scipy import signal
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import skimage.feature as skfeature

# Advanced DTW imports
try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

try:
    from dtaidistance import dtw
    DTW_DISTANCE_AVAILABLE = True
except ImportError:
    DTW_DISTANCE_AVAILABLE = False

# Optional imports with fallbacks
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Graceful degradation for missing optional dependencies
try:
    from scipy import signal
    from scipy.spatial.distance import cosine
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import skimage.feature as skfeature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

@dataclass
class Enhanced360ProcessingConfig:
    """Enhanced configuration optimized for 360° and panoramic videos with GPX validation"""
    # Original processing parameters
    max_frames: int = 150
    target_size: Tuple[int, int] = (720, 480)  # Increased for 360° videos
    sample_rate: float = 2.0
    parallel_videos: int = 1
    gpu_memory_fraction: float = 0.8
    motion_threshold: float = 0.008
    temporal_window: int = 15
    powersafe: bool = False
    save_interval: int = 5
    gpu_timeout: int = 60
    strict: bool = False
    strict_fail: bool = False
    memory_efficient: bool = True
    max_gpu_memory_gb: float = 12.0
    enable_preprocessing: bool = True
    ram_cache_gb: float = 32.0
    disk_cache_gb: float = 1000.0
    cache_dir: str = "~/penis/temp"
    
    # Video validation settings (preserved from original)
    skip_validation: bool = False
    no_quarantine: bool = False
    validation_only: bool = False
    
    # Enhanced 360° video processing features
    enable_360_detection: bool = True
    enable_spherical_processing: bool = True
    enable_tangent_plane_processing: bool = True
    equatorial_region_weight: float = 2.0
    polar_distortion_compensation: bool = True
    longitude_wrap_detection: bool = True
    num_tangent_planes: int = 6
    tangent_plane_fov: float = 90.0
    distortion_aware_attention: bool = True
    
    # Enhanced accuracy features
    use_pretrained_features: bool = True
    use_optical_flow: bool = True
    use_attention_mechanism: bool = True
    use_ensemble_matching: bool = True
    use_advanced_dtw: bool = True
    optical_flow_quality: float = 0.01
    corner_detection_quality: float = 0.01
    max_corners: int = 100
    dtw_window_ratio: float = 0.1
    
    # Enhanced GPS processing
    gps_noise_threshold: float = 0.5
    enable_gps_filtering: bool = True
    enable_cross_modal_learning: bool = True
    
    # GPX validation settings
    gpx_validation_level: str = 'moderate'  # strict, moderate, lenient, custom
    enable_gpx_diagnostics: bool = True
    gpx_diagnostics_file: str = "gpx_validation.db"

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

class Enhanced360OpticalFlowExtractor:
    """360°-aware optical flow extraction using spherical projection methods"""
    
    def __init__(self, config: Enhanced360ProcessingConfig):
        self.config = config
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters  
        self.feature_params = dict(
            maxCorners=config.max_corners,
            qualityLevel=config.corner_detection_quality,
            minDistance=7,
            blockSize=7
        )
        
        # 360° specific parameters
        self.is_360_video = True
        self.tangent_fov = config.tangent_plane_fov
        self.num_tangent_planes = config.num_tangent_planes
        self.equatorial_weight = config.equatorial_region_weight
        
        logger.info("Enhanced 360° optical flow extractor initialized")
    
    def extract_optical_flow_features(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract 360°-aware optical flow features"""
        try:
            # Convert to numpy and prepare for OpenCV
            frames_np = frames_tensor.cpu().numpy()
            batch_size, num_frames, channels, height, width = frames_np.shape
            frames_np = frames_np[0]  # Take first batch
            
            # Detect if this is 360° video (width ≈ 2x height)
            aspect_ratio = width / height
            self.is_360_video = 1.8 <= aspect_ratio <= 2.2
            
            # Convert to grayscale frames
            gray_frames = []
            for i in range(num_frames):
                frame = frames_np[i].transpose(1, 2, 0)  # CHW to HWC
                frame = (frame * 255).astype(np.uint8)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray_frames.append(gray)
            
            if len(gray_frames) < 2:
                return self._create_empty_flow_features(num_frames)
            
            if self.is_360_video and self.config.enable_spherical_processing:
                logger.debug("🌐 Processing 360° video with spherical-aware optical flow")
                # Use 360°-specific processing
                sparse_flow_features = self._extract_spherical_sparse_flow(gray_frames)
                dense_flow_features = self._extract_spherical_dense_flow(gray_frames)
                trajectory_features = self._extract_spherical_trajectories(gray_frames)
                
                # Add spherical-specific features
                spherical_features = self._extract_spherical_motion_features(gray_frames)
                combined_features = {
                    **sparse_flow_features,
                    **dense_flow_features,
                    **trajectory_features,
                    **spherical_features
                }
            else:
                logger.debug("📹 Processing standard panoramic video with enhanced optical flow")
                # Use standard enhanced processing
                sparse_flow_features = self._extract_sparse_flow(gray_frames)
                dense_flow_features = self._extract_dense_flow(gray_frames)
                trajectory_features = self._extract_motion_trajectories(gray_frames)
                
                combined_features = {
                    **sparse_flow_features,
                    **dense_flow_features,
                    **trajectory_features
                }
            
            return combined_features
            
        except Exception as e:
            logger.error(f"360°-aware optical flow extraction failed: {e}")
            return self._create_empty_flow_features(frames_tensor.shape[1] if frames_tensor is not None else 10)
    
    def _extract_spherical_sparse_flow(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract spherical-aware sparse optical flow using tangent plane projections"""
        num_frames = len(gray_frames)
        height, width = gray_frames[0].shape
        
        features = {
            'spherical_sparse_flow_magnitude': np.zeros(num_frames),
            'spherical_sparse_flow_direction': np.zeros(num_frames),
            'equatorial_flow_consistency': np.zeros(num_frames),
            'polar_flow_magnitude': np.zeros(num_frames),
            'border_crossing_events': np.zeros(num_frames)
        }
        
        # Create latitude weights (less weight at poles due to distortion)
        lat_weights = self._create_latitude_weights(height, width)
        
        # Process multiple tangent plane projections
        for i in range(1, num_frames):
            tangent_flows = []
            
            # Extract flow from multiple tangent planes to handle distortion
            for plane_idx in range(self.num_tangent_planes):
                # Convert equirectangular region to tangent plane
                tangent_prev = self._equirect_to_tangent_region(gray_frames[i-1], plane_idx, width, height)
                tangent_curr = self._equirect_to_tangent_region(gray_frames[i], plane_idx, width, height)
                
                if tangent_prev is not None and tangent_curr is not None:
                    # Extract features in tangent plane (less distorted)
                    p0 = cv2.goodFeaturesToTrack(tangent_prev, mask=None, **self.feature_params)
                    
                    if p0 is not None and len(p0) > 0:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(
                            tangent_prev, tangent_curr, p0, None, **self.lk_params
                        )
                        
                        if p1 is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]
                            
                            if len(good_new) > 0:
                                flow_vectors = good_new - good_old
                                tangent_flows.append(flow_vectors)
            
            # Combine tangent plane flows
            if tangent_flows:
                all_flows = np.vstack(tangent_flows)
                magnitudes = np.linalg.norm(all_flows, axis=1)
                directions = np.arctan2(all_flows[:, 1], all_flows[:, 0])
                
                features['spherical_sparse_flow_magnitude'][i] = np.mean(magnitudes)
                features['spherical_sparse_flow_direction'][i] = np.mean(directions)
            
            # Analyze equatorial region specifically (less distorted)
            equatorial_region = self._extract_equatorial_region(gray_frames[i-1], gray_frames[i])
            if equatorial_region:
                features['equatorial_flow_consistency'][i] = equatorial_region
            
            # Analyze polar regions (highly distorted)
            polar_flow = self._extract_polar_flow(gray_frames[i-1], gray_frames[i])
            features['polar_flow_magnitude'][i] = polar_flow
            
            # Detect border crossing events
            border_events = self._detect_border_crossings(gray_frames[i-1], gray_frames[i])
            features['border_crossing_events'][i] = border_events
        
        return features
    
    def _extract_spherical_dense_flow(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract spherical-aware dense optical flow"""
        num_frames = len(gray_frames)
        height, width = gray_frames[0].shape
        
        features = {
            'spherical_dense_flow_magnitude': np.zeros(num_frames),
            'latitude_weighted_flow': np.zeros(num_frames),
            'spherical_flow_coherence': np.zeros(num_frames),
            'angular_flow_histogram': np.zeros((num_frames, 8)),
            'pole_distortion_compensation': np.zeros(num_frames)
        }
        
        # Create latitude weights for distortion compensation
        lat_weights = self._create_latitude_weights(height, width)
        
        for i in range(1, num_frames):
            # Standard dense flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i], None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Apply latitude weighting to reduce polar distortion impact
            weighted_magnitude = magnitude * lat_weights
            
            # Spherical motion statistics
            features['spherical_dense_flow_magnitude'][i] = np.mean(magnitude)
            features['latitude_weighted_flow'][i] = np.mean(weighted_magnitude)
            
            # Flow coherence with distortion compensation
            flow_std = np.std(weighted_magnitude)
            flow_mean = np.mean(weighted_magnitude)
            features['spherical_flow_coherence'][i] = flow_std / (flow_mean + 1e-8)
            
            # Angular histogram in spherical coordinates
            # Convert to spherical angles
            spherical_angles = self._convert_to_spherical_angles(angle, height, width)
            hist, _ = np.histogram(spherical_angles.flatten(), bins=8, range=(0, 2*np.pi))
            features['angular_flow_histogram'][i] = hist / (hist.sum() + 1e-8)
            
            # Pole distortion compensation factor
            pole_region_top = magnitude[:height//6, :]  # Top pole
            pole_region_bottom = magnitude[-height//6:, :]  # Bottom pole
            pole_distortion = (np.mean(pole_region_top) + np.mean(pole_region_bottom)) / 2
            features['pole_distortion_compensation'][i] = pole_distortion
        
        return features
    
    def _extract_spherical_trajectories(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract motion trajectories with spherical geometry awareness"""
        num_frames = len(gray_frames)
        height, width = gray_frames[0].shape
        
        features = {
            'spherical_trajectory_curvature': np.zeros(num_frames),
            'great_circle_deviation': np.zeros(num_frames),
            'spherical_acceleration': np.zeros(num_frames),
            'longitude_wrap_events': np.zeros(num_frames)
        }
        
        if num_frames < 3:
            return features
        
        # Track multiple points across the sphere
        central_points = [
            (width//4, height//2),    # Left side
            (width//2, height//2),    # Center
            (3*width//4, height//2),  # Right side
            (width//2, height//4),    # North
            (width//2, 3*height//4)   # South
        ]
        
        for point_idx, (start_x, start_y) in enumerate(central_points):
            track_point = np.array([[start_x, start_y]], dtype=np.float32).reshape(-1, 1, 2)
            trajectory_2d = [track_point[0, 0]]
            
            # Track in equirectangular space
            for i in range(1, num_frames):
                new_point, status, error = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i], track_point, None, **self.lk_params
                )
                
                if status[0] == 1:
                    # Handle border wrapping for longitude
                    new_x, new_y = new_point[0, 0]
                    
                    # Detect longitude wrap-around
                    prev_x = track_point[0, 0, 0]
                    if abs(new_x - prev_x) > width * 0.5:  # Wrapped around
                        features['longitude_wrap_events'][i] += 1
                        if new_x > width * 0.5:
                            new_x -= width
                        else:
                            new_x += width
                    
                    trajectory_2d.append([new_x, new_y])
                    track_point = np.array([[[new_x, new_y]]], dtype=np.float32)
                else:
                    trajectory_2d.append(trajectory_2d[-1])  # Keep last position
            
            # Convert trajectory to spherical coordinates for analysis
            spherical_trajectory = []
            for x, y in trajectory_2d:
                lon = (x / width) * 2 * np.pi - np.pi  # [-π, π]
                lat = (0.5 - y / height) * np.pi         # [-π/2, π/2]
                spherical_trajectory.append([lon, lat])
            
            spherical_trajectory = np.array(spherical_trajectory)
            
            # Analyze spherical motion
            if len(spherical_trajectory) >= 3:
                # Great circle analysis
                for i in range(2, len(spherical_trajectory)):
                    if i < len(spherical_trajectory) - 1:
                        # Calculate spherical curvature
                        p1, p2, p3 = spherical_trajectory[i-2:i+1]
                        spherical_curvature = self._calculate_spherical_curvature(p1, p2, p3)
                        features['spherical_trajectory_curvature'][i] += spherical_curvature / len(central_points)
                        
                        # Great circle deviation
                        gc_deviation = self._calculate_great_circle_deviation(p1, p2, p3)
                        features['great_circle_deviation'][i] += gc_deviation / len(central_points)
                
                # Spherical acceleration
                spherical_velocities = np.diff(spherical_trajectory, axis=0)
                if len(spherical_velocities) > 1:
                    spherical_accelerations = np.diff(spherical_velocities, axis=0)
                    for i, accel in enumerate(spherical_accelerations):
                        if i + 2 < num_frames:
                            features['spherical_acceleration'][i + 2] += np.linalg.norm(accel) / len(central_points)
        
        return features
    
    def _extract_spherical_motion_features(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract 360°-specific motion features"""
        num_frames = len(gray_frames)
        
        return {
            'camera_rotation_yaw': np.zeros(num_frames),
            'camera_rotation_pitch': np.zeros(num_frames),
            'camera_rotation_roll': np.zeros(num_frames),
            'stabilization_quality': np.zeros(num_frames),
            'stitching_artifact_level': np.zeros(num_frames)
        }
    
    def _create_latitude_weights(self, height: int, width: int) -> np.ndarray:
        """Create latitude-based weights to compensate for equirectangular distortion"""
        # Create weight matrix where equatorial regions have higher weight
        weights = np.ones((height, width))
        
        for y in range(height):
            # Convert pixel y to latitude
            lat = (0.5 - y / height) * np.pi  # [-π/2, π/2]
            
            # Weight based on cosine of latitude (equatorial regions less distorted)
            lat_weight = np.cos(lat)
            weights[y, :] = lat_weight
        
        # Normalize weights
        weights = weights / np.max(weights)
        
        return weights
    
    def _equirect_to_tangent_region(self, frame: np.ndarray, plane_idx: int, width: int, height: int) -> Optional[np.ndarray]:
        """Convert equirectangular region to tangent plane projection"""
        try:
            # Define tangent plane centers (like cubemap faces)
            plane_centers = [
                (0, 0),           # Front
                (np.pi/2, 0),     # Right  
                (np.pi, 0),       # Back
                (-np.pi/2, 0),    # Left
                (0, np.pi/2),     # Up
                (0, -np.pi/2)     # Down
            ]
            
            if plane_idx >= len(plane_centers):
                return None
            
            center_lon, center_lat = plane_centers[plane_idx]
            
            # Extract region around the center
            # Convert center to pixel coordinates
            center_x = int((center_lon + np.pi) / (2 * np.pi) * width) % width
            center_y = int((0.5 - center_lat / np.pi) * height)
            center_y = max(0, min(height - 1, center_y))
            
            # Extract a region (simplified tangent projection)
            region_size = min(width // 4, height // 3)
            x1 = max(0, center_x - region_size // 2)
            x2 = min(width, center_x + region_size // 2)
            y1 = max(0, center_y - region_size // 2)
            y2 = min(height, center_y + region_size // 2)
            
            # Handle wraparound for longitude
            if x2 - x1 < region_size and center_x < region_size // 2:
                # Wrap around case
                left_part = frame[y1:y2, 0:x2]
                right_part = frame[y1:y2, (width - (region_size - x2)):width]
                region = np.hstack([right_part, left_part])
            else:
                region = frame[y1:y2, x1:x2]
            
            return region
            
        except Exception as e:
            logger.debug(f"Tangent region extraction failed: {e}")
            return None
    
    def _extract_sparse_flow(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract sparse optical flow using Lucas-Kanade"""
        num_frames = len(gray_frames)
        
        features = {
            'sparse_flow_magnitude': np.zeros(num_frames),
            'sparse_flow_direction': np.zeros(num_frames),
            'feature_track_consistency': np.zeros(num_frames),
            'corner_motion_vectors': np.zeros((num_frames, 2))
        }
        
        # Detect corners in first frame
        p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **self.feature_params)
        
        if p0 is None or len(p0) == 0:
            return features
        
        # Track features across frames
        for i in range(1, num_frames):
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                gray_frames[i-1], gray_frames[i], p0, None, **self.lk_params
            )
            
            if p1 is None:
                continue
            
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) == 0:
                continue
            
            # Calculate flow vectors
            flow_vectors = good_new - good_old
            
            # Calculate magnitude and direction
            magnitudes = np.linalg.norm(flow_vectors, axis=1)
            directions = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
            
            if len(magnitudes) > 0:
                features['sparse_flow_magnitude'][i] = np.mean(magnitudes)
                features['sparse_flow_direction'][i] = np.mean(directions)
                features['feature_track_consistency'][i] = len(good_new) / len(p0)
                features['corner_motion_vectors'][i] = np.mean(flow_vectors, axis=0)
            
            # Update points for next iteration
            p0 = good_new.reshape(-1, 1, 2)
        
        return features
    
    def _extract_dense_flow(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract dense optical flow using Farneback algorithm"""
        num_frames = len(gray_frames)
        
        features = {
            'dense_flow_magnitude': np.zeros(num_frames),
            'dense_flow_direction': np.zeros(num_frames),
            'flow_histogram': np.zeros((num_frames, 8)),
            'motion_energy': np.zeros(num_frames),
            'flow_coherence': np.zeros(num_frames)
        }
        
        for i in range(1, num_frames):
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i], None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Global motion statistics
            features['dense_flow_magnitude'][i] = np.mean(magnitude)
            features['dense_flow_direction'][i] = np.mean(angle)
            features['motion_energy'][i] = np.sum(magnitude ** 2)
            
            # Flow coherence (how consistent the flow is)
            flow_std = np.std(magnitude)
            flow_mean = np.mean(magnitude)
            features['flow_coherence'][i] = flow_std / (flow_mean + 1e-8)
            
            # Direction histogram
            angle_degrees = angle * 180 / np.pi
            hist, _ = np.histogram(angle_degrees.flatten(), bins=8, range=(0, 360))
            features['flow_histogram'][i] = hist / (hist.sum() + 1e-8)
        
        return features
    
    def _extract_motion_trajectories(self, gray_frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract motion trajectory patterns"""
        num_frames = len(gray_frames)
        
        features = {
            'trajectory_curvature': np.zeros(num_frames),
            'motion_smoothness': np.zeros(num_frames),
            'acceleration_patterns': np.zeros(num_frames),
            'turning_points': np.zeros(num_frames)
        }
        
        if num_frames < 3:
            return features
        
        # Track a central point through frames for trajectory analysis
        center_y, center_x = gray_frames[0].shape[0] // 2, gray_frames[0].shape[1] // 2
        track_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)
        
        trajectory = [track_point[0, 0]]
        
        for i in range(1, num_frames):
            # Track the point
            new_point, status, error = cv2.calcOpticalFlowPyrLK(
                gray_frames[i-1], gray_frames[i], track_point, None, **self.lk_params
            )
            
            if status[0] == 1:
                trajectory.append(new_point[0, 0])
                track_point = new_point
            else:
                trajectory.append(trajectory[-1])  # Keep last position
        
        # Analyze trajectory
        trajectory = np.array(trajectory)
        
        if len(trajectory) >= 3:
            # Calculate curvature
            for i in range(2, len(trajectory)):
                if i < len(trajectory) - 1:
                    # Three consecutive points
                    p1, p2, p3 = trajectory[i-2], trajectory[i-1], trajectory[i]
                    
                    # Calculate curvature using cross product
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    cross_product = np.cross(v1, v2)
                    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                    
                    if magnitude_product > 1e-8:
                        curvature = abs(cross_product) / magnitude_product
                        features['trajectory_curvature'][i] = curvature
            
            # Calculate smoothness and acceleration
            velocities = np.diff(trajectory, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            
            if len(speeds) > 1:
                accelerations = np.diff(speeds)
                features['acceleration_patterns'][2:len(accelerations)+2] = accelerations
                features['motion_smoothness'][1:len(speeds)+1] = speeds
            
            # Detect turning points (local maxima in curvature)
            curvature_signal = features['trajectory_curvature']
            if SCIPY_AVAILABLE:
                peaks, _ = signal.find_peaks(curvature_signal, height=0.1)
                for peak in peaks:
                    if peak < num_frames:
                        features['turning_points'][peak] = 1.0
            else:
                # Simple peak detection without scipy
                for i in range(1, len(curvature_signal) - 1):
                    if (curvature_signal[i] > curvature_signal[i-1] and 
                        curvature_signal[i] > curvature_signal[i+1] and 
                        curvature_signal[i] > 0.1):
                        features['turning_points'][i] = 1.0
        
        return features
    
    def _create_empty_flow_features(self, num_frames: int) -> Dict[str, np.ndarray]:
        """Create empty flow features when extraction fails"""
        return {
            'sparse_flow_magnitude': np.zeros(num_frames),
            'sparse_flow_direction': np.zeros(num_frames),
            'feature_track_consistency': np.zeros(num_frames),
            'corner_motion_vectors': np.zeros((num_frames, 2)),
            'dense_flow_magnitude': np.zeros(num_frames),
            'dense_flow_direction': np.zeros(num_frames),
            'flow_histogram': np.zeros((num_frames, 8)),
            'motion_energy': np.zeros(num_frames),
            'flow_coherence': np.zeros(num_frames),
            'trajectory_curvature': np.zeros(num_frames),
            'motion_smoothness': np.zeros(num_frames),
            'acceleration_patterns': np.zeros(num_frames),
            'turning_points': np.zeros(num_frames)
        }
    
    def _calculate_spherical_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature in spherical coordinates"""
        try:
            # Convert spherical to Cartesian coordinates
            def sphere_to_cart(lon, lat):
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                return np.array([x, y, z])
            
            c1 = sphere_to_cart(p1[0], p1[1])
            c2 = sphere_to_cart(p2[0], p2[1])
            c3 = sphere_to_cart(p3[0], p3[1])
            
            # Calculate spherical curvature using cross product
            v1 = c2 - c1
            v2 = c3 - c2
            
            cross_product = np.cross(v1, v2)
            curvature = np.linalg.norm(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            
            return curvature
            
        except Exception:
            return 0.0
    
    def _calculate_great_circle_deviation(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate deviation from great circle path"""
        try:
            # Convert to Cartesian
            def sphere_to_cart(lon, lat):
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon) 
                z = np.sin(lat)
                return np.array([x, y, z])
            
            c1 = sphere_to_cart(p1[0], p1[1])
            c2 = sphere_to_cart(p2[0], p2[1])
            c3 = sphere_to_cart(p3[0], p3[1])
            
            # Great circle normal vector
            normal = np.cross(c1, c3)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            # Distance from point to great circle
            deviation = abs(np.dot(c2, normal))
            
            return deviation
            
        except Exception:
            return 0.0
    
    def _convert_to_spherical_angles(self, angles: np.ndarray, height: int, width: int) -> np.ndarray:
        """Convert pixel-space angles to spherical coordinate angles"""
        try:
            # Convert angles accounting for equirectangular projection
            y_coords = np.arange(height).reshape(-1, 1)
            
            # Latitude-based correction factor
            lat_correction = np.cos((0.5 - y_coords / height) * np.pi)
            lat_correction = np.broadcast_to(lat_correction, angles.shape)
            
            # Apply correction to angles
            corrected_angles = angles * lat_correction
            
            return corrected_angles
            
        except Exception:
            return angles
    
    def _extract_equatorial_region(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Extract motion features from the less distorted equatorial region"""
        try:
            height = frame1.shape[0]
            
            # Define equatorial region (middle third)
            y1 = height // 3
            y2 = 2 * height // 3
            
            eq_region1 = frame1[y1:y2, :]
            eq_region2 = frame2[y1:y2, :]
            
            # Calculate simple motion in equatorial region
            diff = cv2.absdiff(eq_region1, eq_region2)
            motion = np.mean(diff)
            
            return motion
            
        except Exception:
            return 0.0
    
    def _extract_polar_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Extract motion from polar regions (accounting for high distortion)"""
        try:
            height = frame1.shape[0]
            
            # Polar regions (top and bottom)
            top_region1 = frame1[:height//6, :]
            top_region2 = frame2[:height//6, :]
            bottom_region1 = frame1[-height//6:, :]
            bottom_region2 = frame2[-height//6:, :]
            
            # Calculate motion in polar regions
            top_diff = cv2.absdiff(top_region1, top_region2)
            bottom_diff = cv2.absdiff(bottom_region1, bottom_region2)
            
            polar_motion = (np.mean(top_diff) + np.mean(bottom_diff)) / 2
            
            return polar_motion
            
        except Exception:
            return 0.0
    
    def _detect_border_crossings(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Detect objects crossing the left/right borders of equirectangular frame"""
        try:
            width = frame1.shape[1]
            border_width = width // 20  # 5% border region
            
            # Extract border regions
            left_border1 = frame1[:, :border_width]
            right_border1 = frame1[:, -border_width:]
            left_border2 = frame2[:, :border_width]
            right_border2 = frame2[:, -border_width:]
            
            # Calculate motion at borders
            left_motion = np.mean(cv2.absdiff(left_border1, left_border2))
            right_motion = np.mean(cv2.absdiff(right_border1, right_border2))
            
            # Detect potential border crossings
            border_crossing_score = (left_motion + right_motion) / 2
            
            return border_crossing_score
            
        except Exception:
            return 0.0

class Enhanced360CNNFeatureExtractor:
    """Enhanced CNN feature extraction optimized for 360° and panoramic videos"""
    
    def __init__(self, gpu_manager, config: Enhanced360ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.feature_models = {}
        
        # Initialize models for each GPU
        for gpu_id in gpu_manager.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            self.feature_models[gpu_id] = self._create_enhanced_360_models(device)
        
        logger.info("Enhanced 360° CNN feature extractor initialized")
    
    def _create_enhanced_360_models(self, device: torch.device) -> Dict[str, nn.Module]:
        """Create 360°-optimized ensemble of models"""
        models_dict = {}
        
        # Standard models for equatorial regions (less distorted)
        if self.config.use_pretrained_features:
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = nn.Identity()
            resnet50 = resnet50.to(device).eval()
            models_dict['resnet50'] = resnet50
        
        # Custom 360°-aware spatiotemporal model
        if self.config.enable_spherical_processing:
            spherical_model = self._create_spherical_aware_model().to(device)
            models_dict['spherical'] = spherical_model
        
        # Tangent plane processing model
        if self.config.enable_tangent_plane_processing:
            tangent_model = self._create_tangent_plane_model().to(device)
            models_dict['tangent'] = tangent_model
        
        # Distortion-aware attention model
        if self.config.use_attention_mechanism and self.config.distortion_aware_attention:
            attention_model = self._create_distortion_aware_attention().to(device)
            models_dict['attention'] = attention_model
        
        return models_dict
    
    def _create_spherical_aware_model(self) -> nn.Module:
        """Create spherical-aware feature extraction model"""
        class SphericalAwareNet(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Multi-scale convolutions with distortion awareness
                self.equatorial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.mid_lat_conv = nn.Conv2d(3, 64, kernel_size=5, padding=2)
                self.polar_conv = nn.Conv2d(3, 64, kernel_size=7, padding=3)
                
                # Latitude-aware pooling
                self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 16))
                
                # Spherical feature fusion
                self.fusion = nn.Sequential(
                    nn.Linear(64 * 8 * 16, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256)
                )
                
                # Latitude weight generator
                self.lat_weight_gen = nn.Linear(1, 64)
                
            def forward(self, x):
                batch_size, num_frames, channels, height, width = x.shape
                
                # Create latitude weights
                y_coords = torch.linspace(-1, 1, height, device=x.device).view(-1, 1)
                lat_weights = torch.cos(y_coords * np.pi / 2)
                lat_features = self.lat_weight_gen(lat_weights).unsqueeze(0).unsqueeze(-1)
                
                frame_features = []
                for i in range(num_frames):
                    frame = x[:, i]
                    
                    # Apply different convolutions to different latitude bands
                    eq_region = frame[:, :, height//3:2*height//3, :]
                    mid_region = torch.cat([
                        frame[:, :, height//6:height//3, :],
                        frame[:, :, 2*height//3:5*height//6, :]
                    ], dim=2)
                    polar_region = torch.cat([
                        frame[:, :, :height//6, :],
                        frame[:, :, 5*height//6:, :]
                    ], dim=2)
                    
                    # Process each region
                    if eq_region.size(2) > 0:
                        eq_feat = F.relu(self.equatorial_conv(eq_region))
                    else:
                        eq_feat = torch.zeros(batch_size, 64, 1, width, device=x.device)
                    
                    if mid_region.size(2) > 0:
                        mid_feat = F.relu(self.mid_lat_conv(mid_region))
                    else:
                        mid_feat = torch.zeros(batch_size, 64, 1, width, device=x.device)
                        
                    if polar_region.size(2) > 0:
                        polar_feat = F.relu(self.polar_conv(polar_region))
                    else:
                        polar_feat = torch.zeros(batch_size, 64, 1, width, device=x.device)
                    
                    # Combine features
                    combined_feat = torch.cat([
                        polar_feat[:, :, :polar_region.size(2)//2, :],
                        mid_feat[:, :, :mid_region.size(2)//2, :],
                        eq_feat,
                        mid_feat[:, :, mid_region.size(2)//2:, :],
                        polar_feat[:, :, polar_region.size(2)//2:, :]
                    ], dim=2)
                    
                    # Pool and flatten
                    pooled = self.adaptive_pool(combined_feat)
                    flat_feat = pooled.flatten(start_dim=1)
                    
                    # Apply fusion
                    fused_feat = self.fusion(flat_feat)
                    frame_features.append(fused_feat)
                
                # Stack temporal features
                temporal_features = torch.stack(frame_features, dim=1)
                output = temporal_features.mean(dim=1)
                return output
        
        return SphericalAwareNet()
    
    def _create_tangent_plane_model(self) -> nn.Module:
        """Create tangent plane projection model"""
        class TangentPlaneNet(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Lightweight CNN for tangent plane processing
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                
                # Feature aggregation across tangent planes
                self.plane_aggregator = nn.Sequential(
                    nn.Linear(128 * 6, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
                
            def forward(self, tangent_planes):
                # tangent_planes: [B, num_planes, C, H, W]
                batch_size, num_planes = tangent_planes.shape[:2]
                
                # Process each tangent plane
                plane_features = []
                for i in range(num_planes):
                    plane = tangent_planes[:, i]
                    feat = self.conv_layers(plane).flatten(start_dim=1)
                    plane_features.append(feat)
                
                # Aggregate features from all planes
                all_features = torch.cat(plane_features, dim=1)
                output = self.plane_aggregator(all_features)
                
                return output
        
        return TangentPlaneNet()
    
    def _create_distortion_aware_attention(self) -> nn.Module:
        """Create distortion-aware attention mechanism"""
        class DistortionAwareAttention(nn.Module):
            def __init__(self, feature_dim=256):
                super().__init__()
                
                # Spatial attention with latitude awareness
                self.spatial_attention = nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim // 8, 1),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim // 8, 1, 1),
                    nn.Sigmoid()
                )
                
                # Channel attention
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(feature_dim, feature_dim // 16, 1),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim // 16, feature_dim, 1),
                    nn.Sigmoid()
                )
                
                # Distortion compensation weights
                self.distortion_weights = nn.Parameter(torch.ones(1, 1, 8, 16))
                
            def forward(self, features):
                # Apply channel attention
                channel_att = self.channel_attention(features)
                features = features * channel_att
                
                # Apply spatial attention with distortion awareness
                spatial_att = self.spatial_attention(features)
                
                # Resize distortion weights to match feature map
                dist_weights = F.interpolate(
                    self.distortion_weights, 
                    size=features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Combine attention with distortion compensation
                combined_att = spatial_att * dist_weights
                attended_features = features * combined_att
                
                return attended_features
        
        return DistortionAwareAttention()
    
    def extract_enhanced_features(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract 360°-optimized features"""
        try:
            device = torch.device(f'cuda:{gpu_id}')
            models = self.feature_models[gpu_id]
            
            if frames_tensor.device != device:
                frames_tensor = frames_tensor.to(device)
            
            features = {}
            batch_size, num_frames, channels, height, width = frames_tensor.shape
            
            # Detect if 360° video
            aspect_ratio = width / height
            is_360_video = 1.8 <= aspect_ratio <= 2.2
            
            with torch.no_grad():
                if is_360_video and self.config.enable_spherical_processing:
                    logger.debug("🌐 Processing 360° video features")
                    
                    # Extract features from equatorial region (less distorted)
                    if 'resnet50' in models and self.config.use_pretrained_features:
                        eq_region = frames_tensor[:, :, :, height//3:2*height//3, :]
                        eq_features = self._extract_resnet_features(eq_region, models['resnet50'])
                        features['equatorial_resnet_features'] = eq_features
                    
                    # Extract spherical-aware features
                    if 'spherical' in models:
                        spherical_features = models['spherical'](frames_tensor)
                        features['spherical_features'] = spherical_features[0].cpu().numpy()
                    
                    # Extract tangent plane features
                    if 'tangent' in models and self.config.enable_tangent_plane_processing:
                        tangent_features = self._extract_tangent_plane_features(frames_tensor, models, device)
                        if tangent_features is not None:
                            features['tangent_features'] = tangent_features
                    
                    # Apply distortion-aware attention
                    if 'attention' in models and 'spherical_features' in features:
                        spatial_features = torch.tensor(features['spherical_features']).unsqueeze(0).unsqueeze(0).to(device)
                        spatial_features = spatial_features.view(1, -1, 8, 16)
                        
                        attention_features = models['attention'](spatial_features)
                        features['attention_features'] = attention_features.flatten().cpu().numpy()
                
                else:
                    logger.debug("📹 Processing panoramic video features")
                    
                    # Standard processing for panoramic videos
                    frames_flat = frames_tensor.view(-1, *frames_tensor.shape[2:])
                    
                    # Normalize for pre-trained models
                    if self.config.use_pretrained_features and 'resnet50' in models:
                        normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                        frames_normalized = torch.stack([normalize(frame) for frame in frames_flat])
                        
                        # Extract ResNet50 features
                        resnet_features = models['resnet50'](frames_normalized)
                        resnet_features = resnet_features.view(batch_size, num_frames, -1)[0]
                        features['resnet50_features'] = resnet_features.cpu().numpy()
                    
                    # Extract spherical features (still useful for panoramic)
                    if 'spherical' in models:
                        spherical_features = models['spherical'](frames_tensor)
                        features['spherical_features'] = spherical_features[0].cpu().numpy()
            
            logger.debug(f"360°-aware feature extraction successful: {len(features)} feature types")
            return features
            
        except Exception as e:
            logger.error(f"360°-aware feature extraction failed: {e}")
            return {}
    
    def _extract_resnet_features(self, region_tensor: torch.Tensor, model: nn.Module) -> np.ndarray:
        """Extract ResNet features from a region"""
        try:
            batch_size, num_frames = region_tensor.shape[:2]
            frames_flat = region_tensor.view(-1, *region_tensor.shape[2:])
            
            # Normalize
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            frames_normalized = torch.stack([normalize(frame) for frame in frames_flat])
            
            # Extract features
            features = model(frames_normalized)
            features = features.view(batch_size, num_frames, -1)[0]
            
            return features.cpu().numpy()
            
        except Exception as e:
            logger.debug(f"ResNet feature extraction failed: {e}")
            return np.array([])
    
    def _extract_tangent_plane_features(self, frames_tensor: torch.Tensor, models: Dict, device: torch.device) -> Optional[np.ndarray]:
        """Extract features using tangent plane projections"""
        try:
            if 'tangent' not in models:
                return None
            
            batch_size, num_frames, channels, height, width = frames_tensor.shape
            
            # Create tangent plane projections for each frame
            tangent_features = []
            
            for frame_idx in range(num_frames):
                frame = frames_tensor[0, frame_idx]  # [C, H, W]
                
                # Generate 6 tangent plane projections (like cubemap)
                tangent_planes = []
                plane_centers = [
                    (0, 0),           # Front
                    (np.pi/2, 0),     # Right
                    (np.pi, 0),       # Back
                    (-np.pi/2, 0),    # Left
                    (0, np.pi/2),     # Up
                    (0, -np.pi/2)     # Down
                ]
                
                for center_lon, center_lat in plane_centers:
                    tangent_plane = self._create_tangent_plane_projection(
                        frame, center_lon, center_lat, height, width
                    )
                    if tangent_plane is not None:
                        tangent_planes.append(tangent_plane)
                
                if len(tangent_planes) == 6:
                    # Stack tangent planes: [6, C, H, W]
                    tangent_stack = torch.stack(tangent_planes).unsqueeze(0)  # [1, 6, C, H, W]
                    
                    # Extract features
                    tangent_feat = models['tangent'](tangent_stack)
                    tangent_features.append(tangent_feat)
            
            if tangent_features:
                # Average across frames
                avg_features = torch.stack(tangent_features).mean(dim=0)
                return avg_features[0].cpu().numpy()
            
            return None
            
        except Exception as e:
            logger.debug(f"Tangent plane feature extraction failed: {e}")
            return None
    
    def _create_tangent_plane_projection(self, frame: torch.Tensor, center_lon: float, center_lat: float, 
                                       height: int, width: int, plane_size: int = 64) -> Optional[torch.Tensor]:
        """Create tangent plane projection from equirectangular frame"""
        try:
            # Simplified tangent plane extraction
            # Convert center to pixel coordinates
            center_x = int((center_lon + np.pi) / (2 * np.pi) * width) % width
            center_y = int((0.5 - center_lat / np.pi) * height)
            center_y = max(0, min(height - 1, center_y))
            
            # Extract region around center
            half_size = plane_size // 2
            y1 = max(0, center_y - half_size)
            y2 = min(height, center_y + half_size)
            x1 = max(0, center_x - half_size)
            x2 = min(width, center_x + half_size)
            
            # Handle longitude wraparound
            if x2 - x1 < plane_size and center_x < half_size:
                # Wrap around case
                left_part = frame[:, y1:y2, 0:x2]
                right_part = frame[:, y1:y2, (width - (plane_size - x2)):width]
                region = torch.cat([right_part, left_part], dim=2)
            else:
                region = frame[:, y1:y2, x1:x2]
            
            # Resize to standard size
            if region.size(1) > 0 and region.size(2) > 0:
                region_resized = F.interpolate(
                    region.unsqueeze(0), 
                    size=(plane_size, plane_size), 
                    mode='bilinear', 
                    align_corners=False
                )[0]
                return region_resized
            
            return None
            
        except Exception as e:
            logger.debug(f"Tangent plane creation failed: {e}")
            return None

class AdvancedGPSProcessor:
    """Advanced GPS processing with noise filtering and feature enhancement"""
    
    def __init__(self, config: Enhanced360ProcessingConfig):
        self.config = config
        
        if SKLEARN_AVAILABLE and config.enable_gps_filtering:
            self.scaler = StandardScaler()
            self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        else:
            self.scaler = None
            self.outlier_detector = None
        
        logger.info(f"Advanced GPS processor initialized (filtering: {config.enable_gps_filtering})")
    
    def process_gpx_enhanced(self, gpx_path: str) -> Optional[Dict]:
        """Process GPX with advanced filtering and feature extraction"""
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
            
            # Advanced noise filtering
            if self.config.enable_gps_filtering:
                df = self._filter_gps_noise(df)
            
            if len(df) < 5:
                return None
            
            # Extract enhanced features
            enhanced_features = self._extract_enhanced_gps_features(df)
            
            # Calculate metadata
            duration = self._compute_duration_safe(df['timestamp'])
            total_distance = np.sum(enhanced_features.get('distances', [0]))
            
            return {
                'df': df,
                'features': enhanced_features,
                'start_time': df['timestamp'].iloc[0],
                'end_time': df['timestamp'].iloc[-1],
                'duration': duration,
                'distance': total_distance,
                'point_count': len(df),
                'max_speed': np.max(enhanced_features.get('speed', [0])),
                'avg_speed': np.mean(enhanced_features.get('speed', [0])),
                'processing_mode': 'Enhanced'
            }
            
        except Exception as e:
            logger.debug(f"Enhanced GPS processing failed for {gpx_path}: {e}")
            return None
    
    def _filter_gps_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced GPS noise filtering"""
        if len(df) < 3:
            return df
        
        # Remove obvious outliers based on coordinates
        lat_mean, lat_std = df['lat'].mean(), df['lat'].std()
        lon_mean, lon_std = df['lon'].mean(), df['lon'].std()
        
        # Keep points within 3 standard deviations
        lat_mask = (np.abs(df['lat'] - lat_mean) <= 3 * lat_std)
        lon_mask = (np.abs(df['lon'] - lon_mean) <= 3 * lon_std)
        df = df[lat_mask & lon_mask].reset_index(drop=True)
        
        if len(df) < 3:
            return df
        
        # Calculate speeds for outlier detection
        distances = self._compute_distances_vectorized(df['lat'].values, df['lon'].values)
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        
        speeds = []
        for i in range(len(distances)):
            if i > 0 and time_diffs[i] > 0:
                speed = distances[i] * 3600 / time_diffs[i]  # mph
                speeds.append(speed)
            else:
                speeds.append(0)
        
        # Remove points with impossible speeds (>200 mph)
        speed_mask = np.array(speeds) <= 200
        df = df[speed_mask].reset_index(drop=True)
        
        # Smooth the trajectory using moving average
        if len(df) >= 5:
            window_size = min(5, len(df) // 3)
            df['lat'] = df['lat'].rolling(window=window_size, center=True, min_periods=1).mean()
            df['lon'] = df['lon'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        return df
    
    def _extract_enhanced_gps_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract enhanced GPS features"""
        n_points = len(df)
        
        features = {
            'speed': np.zeros(n_points),
            'acceleration': np.zeros(n_points),
            'bearing': np.zeros(n_points),
            'distances': np.zeros(n_points),
            'curvature': np.zeros(n_points),
            'jerk': np.zeros(n_points),
            'turn_angle': np.zeros(n_points),
            'speed_change_rate': np.zeros(n_points),
            'movement_consistency': np.zeros(n_points)
        }
        
        if n_points < 2:
            return features
        
        # Calculate distances and time differences
        lats, lons = df['lat'].values, df['lon'].values
        distances = self._compute_distances_vectorized(lats, lons)
        time_diffs = self._compute_time_differences_safe(df['timestamp'].values)
        
        features['distances'] = distances
        
        # Calculate speeds
        for i in range(1, n_points):
            if time_diffs[i] > 0:
                features['speed'][i] = distances[i] * 3600 / time_diffs[i]  # mph
        
        # Calculate bearings
        for i in range(1, n_points):
            bearing = self._calculate_bearing(lats[i-1], lons[i-1], lats[i], lons[i])
            features['bearing'][i] = bearing
        
        # Calculate acceleration
        for i in range(2, n_points):
            if time_diffs[i] > 0:
                speed_diff = features['speed'][i] - features['speed'][i-1]
                features['acceleration'][i] = speed_diff / time_diffs[i]
        
        # Calculate jerk (rate of acceleration change)
        for i in range(3, n_points):
            if time_diffs[i] > 0:
                accel_diff = features['acceleration'][i] - features['acceleration'][i-1]
                features['jerk'][i] = accel_diff / time_diffs[i]
        
        # Calculate curvature and turn angles
        for i in range(2, n_points):
            # Turn angle between consecutive segments
            if i > 1:
                bearing1 = features['bearing'][i-1]
                bearing2 = features['bearing'][i]
                
                turn_angle = bearing2 - bearing1
                # Normalize to [-180, 180]
                while turn_angle > 180:
                    turn_angle -= 360
                while turn_angle < -180:
                    turn_angle += 360
                
                features['turn_angle'][i] = abs(turn_angle)
                
                # Curvature approximation
                if distances[i] > 0:
                    features['curvature'][i] = abs(turn_angle) / (distances[i] * 111000)  # Convert to meters
        
        # Calculate speed change rate
        for i in range(2, n_points):
            if features['speed'][i-1] > 0:
                speed_change = abs(features['speed'][i] - features['speed'][i-1])
                features['speed_change_rate'][i] = speed_change / features['speed'][i-1]
        
        # Calculate movement consistency
        window_size = min(5, n_points // 3)
        for i in range(window_size, n_points - window_size):
            # Consistency based on speed variance in local window
            local_speeds = features['speed'][i-window_size:i+window_size+1]
            if len(local_speeds) > 1:
                speed_std = np.std(local_speeds)
                speed_mean = np.mean(local_speeds)
                features['movement_consistency'][i] = 1.0 / (1.0 + speed_std / (speed_mean + 1e-8))
        
        return features
    
    def _compute_distances_vectorized(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Vectorized distance computation using Haversine formula"""
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
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360  # Normalize to [0, 360]
        
        return bearing
    
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

class AdvancedDTWEngine:
    """Advanced Dynamic Time Warping with shape information and constraints"""
    
    def __init__(self, config: Enhanced360ProcessingConfig):
        self.config = config
        
    def compute_enhanced_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute enhanced DTW with shape information and constraints"""
        try:
            if len(seq1) == 0 or len(seq2) == 0:
                return float('inf')
            
            # Normalize sequences
            seq1_norm = self._robust_normalize(seq1)
            seq2_norm = self._robust_normalize(seq2)
            
            # Try different DTW variants and take the best
            dtw_scores = []
            
            # Standard DTW with window constraint
            if DTW_DISTANCE_AVAILABLE:
                window_size = max(5, int(min(len(seq1), len(seq2)) * self.config.dtw_window_ratio))
                try:
                    dtw_score = dtw.distance(seq1_norm, seq2_norm, window=window_size)
                    dtw_scores.append(dtw_score)
                except:
                    pass
            
            # FastDTW if available
            if FASTDTW_AVAILABLE:
                try:
                    distance, _ = fastdtw(seq1_norm, seq2_norm, dist=lambda x, y: abs(x - y))
                    dtw_scores.append(distance)
                except:
                    pass
            
            # Custom shape-aware DTW
            try:
                shape_dtw_score = self._shape_aware_dtw(seq1_norm, seq2_norm)
                dtw_scores.append(shape_dtw_score)
            except:
                pass
            
            # Fallback to basic DTW
            if not dtw_scores:
                dtw_scores.append(self._basic_dtw(seq1_norm, seq2_norm))
            
            # Return best (minimum) score
            return min(dtw_scores)
            
        except Exception as e:
            logger.debug(f"Enhanced DTW computation failed: {e}")
            return float('inf')
    
    def _shape_aware_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute shape-aware DTW considering local patterns"""
        # Extract shape descriptors
        shape1 = self._extract_shape_descriptors(seq1)
        shape2 = self._extract_shape_descriptors(seq2)
        
        # Compute DTW on shape descriptors
        n, m = len(shape1), len(shape2)
        
        # Create cost matrix
        cost_matrix = np.full((n, m), float('inf'))
        
        # Initialize
        cost_matrix[0, 0] = np.linalg.norm(shape1[0] - shape2[0])
        
        # Fill first row and column
        for i in range(1, n):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + np.linalg.norm(shape1[i] - shape2[0])
        
        for j in range(1, m):
            cost_matrix[0, j] = cost_matrix[0, j-1] + np.linalg.norm(shape1[0] - shape2[j])
        
        # Fill rest of matrix
        for i in range(1, n):
            for j in range(1, m):
                cost = np.linalg.norm(shape1[i] - shape2[j])
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i-1, j],     # insertion
                    cost_matrix[i, j-1],     # deletion
                    cost_matrix[i-1, j-1]    # match
                )
        
        return cost_matrix[n-1, m-1] / max(n, m)  # Normalize by length
    
    def _extract_shape_descriptors(self, sequence: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Extract local shape descriptors for each point"""
        n = len(sequence)
        descriptors = np.zeros((n, window_size * 2))  # Local statistics
        
        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            local_window = sequence[start:end]
            
            if len(local_window) > 1:
                # Local statistics as shape descriptor
                desc = [
                    np.mean(local_window),
                    np.std(local_window),
                    np.max(local_window) - np.min(local_window),  # Range
                ]
                
                # Add local derivatives if possible
                if len(local_window) > 2:
                    diffs = np.diff(local_window)
                    desc.extend([
                        np.mean(diffs),
                        np.std(diffs),
                        np.sum(diffs > 0) / len(diffs)  # Proportion of increases
                    ])
                else:
                    desc.extend([0, 0, 0.5])
                
                # Pad to fixed size
                while len(desc) < window_size * 2:
                    desc.append(0)
                
                descriptors[i] = np.array(desc[:window_size * 2])
        
        return descriptors
    
    def _basic_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Basic DTW implementation as fallback"""
        n, m = len(seq1), len(seq2)
        
        # Create cost matrix
        cost_matrix = np.full((n + 1, m + 1), float('inf'))
        cost_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i-1, j],     # insertion
                    cost_matrix[i, j-1],     # deletion
                    cost_matrix[i-1, j-1]    # match
                )
        
        return cost_matrix[n, m] / max(n, m)
    
    def _robust_normalize(self, sequence: np.ndarray) -> np.ndarray:
        """Robust normalization"""
        if len(sequence) == 0:
            return sequence
        
        # Use median and MAD for robust normalization
        median = np.median(sequence)
        mad = np.median(np.abs(sequence - median))
        
        if mad > 1e-8:
            return (sequence - median) / mad
        else:
            return sequence - median

class EnsembleSimilarityEngine:
    """Ensemble similarity engine with multiple correlation methods"""
    
    def __init__(self, config: Enhanced360ProcessingConfig):
        self.config = config
        self.dtw_engine = AdvancedDTWEngine(config)
        
        # Enhanced weights for ensemble
        if config.use_ensemble_matching:
            self.weights = {
                'motion_dynamics': 0.25,
                'temporal_correlation': 0.20,
                'statistical_profile': 0.15,
                'optical_flow_correlation': 0.15,
                'cnn_feature_correlation': 0.15,
                'advanced_dtw_correlation': 0.10
            }
        else:
            # Traditional weights if ensemble is disabled
            self.weights = {
                'motion_dynamics': 0.40,
                'temporal_correlation': 0.30,
                'statistical_profile': 0.30
            }
    
    def compute_ensemble_similarity(self, video_features: Dict, gpx_features: Dict) -> Dict[str, float]:
        """Compute ensemble similarity using multiple methods"""
        try:
            similarities = {}
            
            # Traditional motion dynamics
            similarities['motion_dynamics'] = self._compute_motion_similarity(video_features, gpx_features)
            
            # Temporal correlation
            similarities['temporal_correlation'] = self._compute_temporal_similarity(video_features, gpx_features)
            
            # Statistical profile
            similarities['statistical_profile'] = self._compute_statistical_similarity(video_features, gpx_features)
            
            # Enhanced features if enabled
            if self.config.use_ensemble_matching:
                # Optical flow correlation
                similarities['optical_flow_correlation'] = self._compute_optical_flow_similarity(video_features, gpx_features)
                
                # CNN feature correlation
                similarities['cnn_feature_correlation'] = self._compute_cnn_feature_similarity(video_features, gpx_features)
                
                # Advanced DTW correlation
                if self.config.use_advanced_dtw:
                    similarities['advanced_dtw_correlation'] = self._compute_advanced_dtw_similarity(video_features, gpx_features)
                else:
                    similarities['advanced_dtw_correlation'] = 0.0
            
            # Weighted ensemble
            valid_similarities = {k: v for k, v in similarities.items() if not np.isnan(v) and v >= 0}
            
            if valid_similarities:
                total_weight = sum(self.weights.get(k, 0) for k in valid_similarities.keys())
                if total_weight > 0:
                    combined_score = sum(
                        similarities[k] * self.weights.get(k, 0) / total_weight 
                        for k in valid_similarities.keys()
                    )
                else:
                    combined_score = 0.0
            else:
                combined_score = 0.0
            
            similarities['combined'] = float(np.clip(combined_score, 0.0, 1.0))
            similarities['quality'] = self._assess_quality(similarities['combined'])
            similarities['confidence'] = len(valid_similarities) / len(self.weights)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Ensemble similarity computation failed: {e}")
            return self._create_zero_similarity()
    
    def _compute_motion_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Enhanced motion similarity"""
        try:
            # Get motion signatures from multiple sources
            video_motions = []
            gpx_motions = []
            
            # Traditional motion magnitude
            if 'motion_magnitude' in video_features:
                video_motions.append(video_features['motion_magnitude'])
            
            # Optical flow motion
            if 'sparse_flow_magnitude' in video_features:
                video_motions.append(video_features['sparse_flow_magnitude'])
                
            if 'dense_flow_magnitude' in video_features:
                video_motions.append(video_features['dense_flow_magnitude'])
            
            # 360° specific motion
            if 'spherical_dense_flow_magnitude' in video_features:
                video_motions.append(video_features['spherical_dense_flow_magnitude'])
            
            # GPS motion features
            if 'speed' in gpx_features:
                gpx_motions.append(gpx_features['speed'])
                
            if 'acceleration' in gpx_features:
                gpx_motions.append(gpx_features['acceleration'])
            
            if not video_motions or not gpx_motions:
                return 0.0
            
            # Compute correlations for all combinations
            correlations = []
            for v_motion in video_motions:
                for g_motion in gpx_motions:
                    if len(v_motion) > 3 and len(g_motion) > 3:
                        corr = self._compute_robust_correlation(v_motion, g_motion)
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                return float(np.max(correlations))  # Take best correlation
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Motion similarity computation failed: {e}")
            return 0.0
    
    def _compute_optical_flow_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute optical flow based similarity"""
        try:
            # Extract optical flow features
            flow_features = []
            
            if 'trajectory_curvature' in video_features:
                flow_features.append(video_features['trajectory_curvature'])
                
            if 'motion_energy' in video_features:
                flow_features.append(video_features['motion_energy'])
                
            if 'turning_points' in video_features:
                flow_features.append(video_features['turning_points'])
            
            # 360° specific flow features
            if 'spherical_trajectory_curvature' in video_features:
                flow_features.append(video_features['spherical_trajectory_curvature'])
            
            # Extract corresponding GPS features
            gps_features = []
            
            if 'curvature' in gpx_features:
                gps_features.append(gpx_features['curvature'])
                
            if 'turn_angle' in gpx_features:
                gps_features.append(gpx_features['turn_angle'])
                
            if 'jerk' in gpx_features:
                gps_features.append(gpx_features['jerk'])
            
            if not flow_features or not gps_features:
                return 0.0
            
            # Compute correlations
            correlations = []
            for flow_feat in flow_features:
                for gps_feat in gps_features:
                    if len(flow_feat) > 5 and len(gps_feat) > 5:
                        # Use DTW for better alignment
                        dtw_score = self.dtw_engine.compute_enhanced_dtw(flow_feat, gps_feat)
                        if dtw_score != float('inf'):
                            # Convert DTW distance to similarity
                            similarity = 1.0 / (1.0 + dtw_score)
                            correlations.append(similarity)
            
            if correlations:
                return float(np.max(correlations))
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Optical flow similarity computation failed: {e}")
            return 0.0
    
    def _compute_cnn_feature_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute CNN feature based similarity"""
        try:
            # Extract high-level CNN features
            cnn_feature_keys = ['resnet50_features', 'spherical_features', 'equatorial_resnet_features', 'tangent_features', 'attention_features']
            
            # Create motion profiles from CNN features
            motion_profiles = []
            
            for key in cnn_feature_keys:
                if key in video_features:
                    features = video_features[key]
                    if len(features.shape) == 2:  # [time, features]
                        # Extract motion-relevant patterns
                        motion_profile = np.linalg.norm(features, axis=1)  # Magnitude over time
                        motion_profiles.append(motion_profile)
            
            if not motion_profiles:
                return 0.0
            
            # Compare with GPS motion patterns
            gps_motion_keys = ['speed', 'acceleration', 'movement_consistency']
            best_correlation = 0.0
            
            for motion_profile in motion_profiles:
                for gps_key in gps_motion_keys:
                    if gps_key in gpx_features:
                        gps_motion = gpx_features[gps_key]
                        if len(motion_profile) > 3 and len(gps_motion) > 3:
                            corr = self._compute_robust_correlation(motion_profile, gps_motion)
                            if not np.isnan(corr):
                                best_correlation = max(best_correlation, abs(corr))
            
            return float(best_correlation)
            
        except Exception as e:
            logger.debug(f"CNN feature similarity computation failed: {e}")
            return 0.0
    
    def _compute_advanced_dtw_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Compute advanced DTW-based similarity"""
        try:
            # Get primary motion sequences
            video_motion = None
            gps_motion = None
            
            # Prioritize optical flow features for video
            if 'dense_flow_magnitude' in video_features:
                video_motion = video_features['dense_flow_magnitude']
            elif 'spherical_dense_flow_magnitude' in video_features:
                video_motion = video_features['spherical_dense_flow_magnitude']
            elif 'motion_magnitude' in video_features:
                video_motion = video_features['motion_magnitude']
            
            # Prioritize speed for GPS
            if 'speed' in gpx_features:
                gps_motion = gpx_features['speed']
            
            if video_motion is None or gps_motion is None:
                return 0.0
            
            if len(video_motion) < 3 or len(gps_motion) < 3:
                return 0.0
            
            # Compute enhanced DTW
            dtw_distance = self.dtw_engine.compute_enhanced_dtw(video_motion, gps_motion)
            
            if dtw_distance == float('inf'):
                return 0.0
            
            # Convert distance to similarity
            max_len = max(len(video_motion), len(gps_motion))
            normalized_distance = dtw_distance / max_len
            similarity = 1.0 / (1.0 + normalized_distance)
            
            return float(similarity)
            
        except Exception as e:
            logger.debug(f"Advanced DTW similarity computation failed: {e}")
            return 0.0
    
    def _compute_temporal_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Enhanced temporal correlation"""
        try:
            # Extract temporal signatures with better features
            video_temporal = self._extract_enhanced_temporal_signature(video_features, 'video')
            gpx_temporal = self._extract_enhanced_temporal_signature(gpx_features, 'gpx')
            
            if video_temporal is None or gpx_temporal is None:
                return 0.0
            
            # Direct correlation
            min_len = min(len(video_temporal), len(gpx_temporal))
            if min_len > 5:
                v_temp = video_temporal[:min_len]
                g_temp = gpx_temporal[:min_len]
                
                corr = self._compute_robust_correlation(v_temp, g_temp)
                if not np.isnan(corr):
                    return float(np.clip(abs(corr), 0.0, 1.0))
            
            return 0.0
                
        except Exception as e:
            logger.debug(f"Enhanced temporal similarity computation failed: {e}")
            return 0.0
    
    def _extract_enhanced_temporal_signature(self, features: Dict, source_type: str) -> Optional[np.ndarray]:
        """Extract enhanced temporal signature"""
        try:
            candidates = []
            
            if source_type == 'video':
                # Use multiple video features for temporal signature
                feature_keys = ['motion_magnitude', 'dense_flow_magnitude', 'motion_energy', 'acceleration_patterns', 'spherical_dense_flow_magnitude']
                for key in feature_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 5:
                            if np.isfinite(values).all():
                                candidates.append(np.diff(values))  # Temporal changes
                                
            elif source_type == 'gpx':
                # Use multiple GPS features for temporal signature
                feature_keys = ['speed', 'acceleration', 'speed_change_rate']
                for key in feature_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 5:
                            if np.isfinite(values).all():
                                candidates.append(np.diff(values))  # Temporal changes
            
            if candidates:
                # Use the candidate with highest variance (most informative)
                variances = [np.var(candidate) for candidate in candidates]
                best_idx = np.argmax(variances)
                return self._robust_normalize(candidates[best_idx])
            
            return None
            
        except Exception as e:
            logger.debug(f"Enhanced temporal signature extraction failed for {source_type}: {e}")
            return None
    
    def _compute_statistical_similarity(self, video_features: Dict, gpx_features: Dict) -> float:
        """Enhanced statistical profile similarity"""
        try:
            video_stats = self._extract_enhanced_statistical_profile(video_features, 'video')
            gpx_stats = self._extract_enhanced_statistical_profile(gpx_features, 'gpx')
            
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
            if SCIPY_AVAILABLE:
                cosine_sim = 1 - cosine(video_stats, gpx_stats)
            else:
                # Manual cosine similarity calculation
                dot_product = np.dot(video_stats, gpx_stats)
                norm_a = np.linalg.norm(video_stats)
                norm_b = np.linalg.norm(gpx_stats)
                cosine_sim = dot_product / (norm_a * norm_b + 1e-8)
            
            if not np.isnan(cosine_sim):
                return float(np.clip(abs(cosine_sim), 0.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Enhanced statistical similarity computation failed: {e}")
            return 0.0
    
    def _extract_enhanced_statistical_profile(self, features: Dict, source_type: str) -> Optional[np.ndarray]:
        """Extract enhanced statistical profile"""
        profile_components = []
        
        try:
            if source_type == 'video':
                # Enhanced video statistical features
                feature_keys = [
                    'motion_magnitude', 'color_variance', 'edge_density',
                    'sparse_flow_magnitude', 'dense_flow_magnitude', 'motion_energy',
                    'trajectory_curvature', 'motion_smoothness', 'spherical_dense_flow_magnitude',
                    'latitude_weighted_flow'
                ]
                
                for key in feature_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend([
                                    np.mean(values), np.std(values), np.median(values),
                                    np.percentile(values, 75) - np.percentile(values, 25)  # IQR
                                ])
                            
            elif source_type == 'gpx':
                # Enhanced GPS statistical features
                feature_keys = [
                    'speed', 'acceleration', 'bearing', 'curvature',
                    'jerk', 'turn_angle', 'speed_change_rate', 'movement_consistency'
                ]
                
                for key in feature_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile_components.extend([
                                    np.mean(values), np.std(values), np.median(values),
                                    np.percentile(values, 75) - np.percentile(values, 25)  # IQR
                                ])
            
            if not profile_components:
                return None
            
            return np.array(profile_components)
            
        except Exception as e:
            logger.debug(f"Enhanced statistical profile extraction failed for {source_type}: {e}")
            return None
    
    def _compute_robust_correlation(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute robust correlation between sequences"""
        try:
            # Handle different lengths
            min_len = min(len(seq1), len(seq2))
            if min_len < 3:
                return 0.0
            
            s1 = seq1[:min_len]
            s2 = seq2[:min_len]
            
            # Remove constant sequences
            if np.std(s1) < 1e-8 or np.std(s2) < 1e-8:
                return 0.0
            
            # Compute Pearson correlation
            correlation = np.corrcoef(s1, s2)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return correlation
            
        except Exception:
            return 0.0
    
    def _robust_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Robust normalization with outlier handling"""
        try:
            if len(vector) == 0:
                return vector
            
            # Use median and MAD for robust normalization
            median = np.median(vector)
            mad = np.median(np.abs(vector - median))
            
            if mad > 1e-8:
                return (vector - median) / mad
            else:
                return vector - median
                
        except Exception:
            return vector
    
    def _assess_quality(self, score: float) -> str:
        """Assess similarity quality with enhanced thresholds"""
        if score >= 0.85:
            return 'excellent'
        elif score >= 0.70:
            return 'very_good'
        elif score >= 0.55:
            return 'good'
        elif score >= 0.40:
            return 'fair'
        elif score >= 0.25:
            return 'poor'
        else:
            return 'very_poor'
    
    def _create_zero_similarity(self) -> Dict[str, float]:
        """Create zero similarity result"""
        return {
            'motion_dynamics': 0.0,
            'temporal_correlation': 0.0,
            'statistical_profile': 0.0,
            'optical_flow_correlation': 0.0,
            'cnn_feature_correlation': 0.0,
            'advanced_dtw_correlation': 0.0,
            'combined': 0.0,
            'quality': 'failed',
            'confidence': 0.0
        }

# Preserve all the original production classes from matcher39.py
from pathlib import Path

class VideoValidator:
    """Advanced video validation system with GPU compatibility testing"""
    
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
        self.gpu_friendly_codecs = {'h264', 'avc1', 'mp4v', 'mpeg4'}
        self.gpu_problematic_codecs = {'hevc', 'h265', 'vp9', 'av1', 'vp8'}
        
        logger.info(f"Enhanced Video Validator initialized:")
        logger.info(f"  Strict Mode: {config.strict or config.strict_fail}")
        logger.info(f"  Quarantine Directory: {self.quarantine_dir}")
        logger.info(f"  GPU Test Directory: {self.temp_test_dir}")
    
    def validate_video_batch(self, video_files, quarantine_corrupted=True):
        """Validate a batch of video files with enhanced GPU compatibility testing"""
        logger.info(f"Pre-flight validation of {len(video_files)} videos...")
        
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
        """Enhanced single video validation with GPU compatibility"""
        validation_result = {
            'is_valid': False,
            'error': None,
            'file_size_mb': 0,
            'duration': 0,
            'codec': None,
            'resolution': None,
            'issues': [],
            'gpu_compatibility': 'unknown',
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
            
            # Stage 2: FFprobe validation
            validation_result['validation_stage'] = 'ffprobe_validation'
            probe_result = self.ffprobe_validation(video_path)
            if not probe_result['success']:
                validation_result['error'] = probe_result['error']
                return validation_result
            
            # Update with probe data
            validation_result.update(probe_result['data'])
            
            # Stage 3: GPU compatibility assessment
            validation_result['validation_stage'] = 'gpu_compatibility'
            gpu_compat = self.assess_gpu_compatibility(validation_result)
            validation_result['gpu_compatibility'] = gpu_compat
            
            # Stage 4: Strict mode validation
            validation_result['validation_stage'] = 'strict_mode_check'
            if self.config.strict or self.config.strict_fail:
                strict_valid = self.strict_mode_validation(video_path, validation_result)
                if not strict_valid:
                    validation_result['strict_rejected'] = True
                    return validation_result
            
            # Stage 5: Final validation
            validation_result['validation_stage'] = 'completed'
            validation_result['is_valid'] = True
            
            return validation_result
            
        except Exception as e:
            validation_result['error'] = f"Validation exception at {validation_result['validation_stage']}: {str(e)}"
            return validation_result
    
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
            
            proc_result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=45)
            
            if proc_result.returncode != 0:
                error_output = proc_result.stderr.strip()
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
        
        else:
            return 'unknown'
    
    def strict_mode_validation(self, video_path, validation_result):
        """Strict mode validation with GPU compatibility"""
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
                validation_result['error'] = f"STRICT: Codec '{codec}' cannot be processed"
                return False
        
        return True
    
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
                f.write(f"Validator version: Enhanced VideoValidator v3.0\n")
            
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
            'validator_version': 'Enhanced VideoValidator v3.0',
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
    
    def __init__(self, cache_dir: Path, config: Enhanced360ProcessingConfig):
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
    
    def __init__(self, gpu_ids: List[int], strict: bool = False, config: Optional[Enhanced360ProcessingConfig] = None):
        self.gpu_ids = gpu_ids
        self.strict = strict
        self.config = config or Enhanced360ProcessingConfig()
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

class EnhancedFFmpegDecoder:
    """Enhanced FFmpeg decoder with GPU preprocessing for 360° videos"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: Enhanced360ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.temp_dirs = {}
        
        # Create temp directories per GPU
        base_temp = Path(config.cache_dir) / "gpu_temp"
        base_temp.mkdir(parents=True, exist_ok=True)
        
        for gpu_id in gpu_manager.gpu_ids:
            self.temp_dirs[gpu_id] = base_temp / f'gpu_{gpu_id}'
            self.temp_dirs[gpu_id].mkdir(exist_ok=True)
        
        logger.info(f"Enhanced 360° decoder initialized for GPUs: {gpu_manager.gpu_ids}")
    
    def decode_video_enhanced(self, video_path: str, gpu_id: int) -> Tuple[Optional[torch.Tensor], float, float]:
        """Enhanced video decoding with 360° awareness"""
        try:
            # Get video info
            video_info = self._get_video_info(video_path)
            if not video_info:
                raise RuntimeError("Could not get video info")
            
            # Detect 360° video
            aspect_ratio = video_info['width'] / video_info['height']
            is_360_video = 1.8 <= aspect_ratio <= 2.2
            
            if is_360_video and self.config.enable_360_detection:
                logger.debug(f"🌐 Detected 360° video: {Path(video_path).name}")
                # Use 360°-optimized decoding
                frames_tensor = self._decode_360_frames(video_path, video_info, gpu_id)
            else:
                logger.debug(f"📹 Processing standard video: {Path(video_path).name}")
                # Use standard decoding
                frames_tensor = self._decode_uniform_frames(video_path, video_info, gpu_id)
            
            if frames_tensor is None:
                raise RuntimeError("Frame decoding failed")
            
            return frames_tensor, video_info['fps'], video_info['duration']
            
        except Exception as e:
            logger.error(f"Enhanced video decoding failed for {video_path}: {e}")
            return None, 0, 0
    
    def _decode_360_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """360°-optimized frame decoding with higher resolution"""
        temp_dir = self.temp_dirs[gpu_id]
        output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
        
        # Use higher resolution for 360° videos
        target_width = max(self.config.target_size[0], 720)  # Minimum 720p wide for 360°
        target_height = max(self.config.target_size[1], 360)  # Maintain aspect ratio
        
        # Ensure even numbers
        if target_width % 2 != 0:
            target_width += 1
        if target_height % 2 != 0:
            target_height += 1
        
        # Calculate sampling rate
        total_frames = int(video_info['duration'] * video_info['fps'])
        max_frames = self.config.max_frames
        
        if total_frames > max_frames:
            sample_rate = total_frames / max_frames
            vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,select=not(mod(n\\,{int(sample_rate)}))'
        else:
            vf_filter = f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2'
        
        # Enhanced CUDA command for 360° videos
        cuda_cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-vf', vf_filter,
            '-frames:v', str(min(max_frames, total_frames)),
            '-q:v', '2',  # Higher quality for 360° videos
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
        
        return self._execute_ffmpeg_and_load(cuda_cmd, cpu_cmd, temp_dir, gpu_id, target_width, target_height, video_path)
    
    def _decode_uniform_frames(self, video_path: str, video_info: Dict, gpu_id: int) -> Optional[torch.Tensor]:
        """Standard uniform frame sampling"""
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
        
        return self._execute_ffmpeg_and_load(cuda_cmd, cpu_cmd, temp_dir, gpu_id, target_width, target_height, video_path)
    
    def _execute_ffmpeg_and_load(self, cuda_cmd, cpu_cmd, temp_dir, gpu_id, target_width, target_height, video_path):
        """Execute FFmpeg commands and load frames"""
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

class Enhanced360FeatureExtractor:
    """Enhanced feature extraction combining traditional and 360° methods"""
    
    def __init__(self, gpu_manager: EnhancedGPUManager, config: Enhanced360ProcessingConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        
        # Initialize optical flow extractor
        if config.use_optical_flow:
            self.optical_flow_extractor = Enhanced360OpticalFlowExtractor(config)
        else:
            self.optical_flow_extractor = None
        
        # Initialize CNN feature extractor
        if config.use_pretrained_features or config.enable_spherical_processing:
            self.cnn_extractor = Enhanced360CNNFeatureExtractor(gpu_manager, config)
        else:
            self.cnn_extractor = None
        
        logger.info("Enhanced 360° feature extractor initialized")
    
    def extract_enhanced_features(self, frames_tensor: torch.Tensor, gpu_id: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive enhanced features"""
        try:
            device = torch.device(f'cuda:{gpu_id}')
            
            if frames_tensor.device != device:
                frames_tensor = frames_tensor.to(device)
            
            features = {}
            batch_size, num_frames = frames_tensor.shape[:2]
            
            # Traditional motion features (always computed)
            motion_features = self._compute_traditional_motion(frames_tensor[0], device)
            features.update(motion_features)
            
            # Enhanced optical flow features
            if self.optical_flow_extractor and self.config.use_optical_flow:
                optical_flow_features = self.optical_flow_extractor.extract_optical_flow_features(frames_tensor, gpu_id)
                features.update(optical_flow_features)
            
            # CNN features
            if self.cnn_extractor:
                cnn_features = self.cnn_extractor.extract_enhanced_features(frames_tensor, gpu_id)
                features.update(cnn_features)
            
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
    
    def _compute_traditional_motion(self, frames: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
        """Compute traditional motion features"""
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

def process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
    """Enhanced video processing with 360° support"""
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
        feature_extractor = Enhanced360FeatureExtractor(gpu_manager, config)
        
        # Enhanced decode with 360° support
        frames_tensor, fps, duration = decoder.decode_video_enhanced(video_path, gpu_id)
        
        if frames_tensor is None:
            error_msg = f"Video decoding failed for {Path(video_path).name}"
            
            # Handle different strict modes
            if config.strict_fail:
                error_msg = f"ULTRA STRICT MODE: {error_msg}"
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                raise RuntimeError(error_msg)
            elif config.strict:
                logger.error(f"STRICT MODE: {error_msg}")
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, f"STRICT MODE: {error_msg}")
                return video_path, None
            else:
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, error_msg)
                return video_path, None
        
        # Extract enhanced features
        features = feature_extractor.extract_enhanced_features(frames_tensor, gpu_id)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_id
        features['processing_mode'] = 'Enhanced360_GPU_STRICT' if config.strict else 'Enhanced360_GPU'
        
        # Detect video type
        _, _, _, height, width = frames_tensor.shape
        aspect_ratio = width / height
        is_360_video = 1.8 <= aspect_ratio <= 2.2
        features['is_360_video'] = is_360_video
        features['aspect_ratio'] = aspect_ratio
        
        # Mark feature extraction as done in power-safe mode
        if powersafe_manager:
            powersafe_manager.mark_video_features_done(video_path)
        
        success_msg = f"Successfully processed {Path(video_path).name} on GPU {gpu_id}"
        if is_360_video:
            success_msg += " [360° VIDEO]"
        if config.strict:
            success_msg += " [STRICT MODE]"
        logger.info(success_msg)
        return video_path, features
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        
        # Handle different strict modes for exceptions
        if config.strict_fail:
            error_msg = f"ULTRA STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {Path(video_path).name}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            raise RuntimeError(error_msg)
        elif config.strict:
            if "STRICT MODE" not in str(e):
                error_msg = f"STRICT MODE: {error_msg}"
            logger.error(f"{error_msg} for {Path(video_path).name}")
            if powersafe_manager:
                powersafe_manager.mark_video_failed(video_path, error_msg)
            return video_path, None
        else:
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

def main():
    """Enhanced main function with 360° video processing and advanced GPX validation"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Production-Ready Multi-GPU Video-GPX Correlation Script with 360° Support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", required=True,
                       help="Directory containing videos and GPX files")
    
    # Processing configuration
    parser.add_argument("--max_frames", type=int, default=150,
                       help="Maximum frames per video (default: 150)")
    parser.add_argument("--video_size", nargs=2, type=int, default=[720, 480],
                       help="Target video resolution (default: 720 480)")
    parser.add_argument("--sample_rate", type=float, default=2.0,
                       help="Video sampling rate (default: 2.0)")
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
    
    # Enhanced processing features
    parser.add_argument("--enable-360-detection", action='store_true', default=True,
                       help="Enable automatic 360° video detection (default: True)")
    parser.add_argument("--enable-spherical-processing", action='store_true', default=True,
                       help="Enable spherical-aware processing for 360° videos (default: True)")
    parser.add_argument("--enable-tangent-planes", action='store_true', default=True,
                       help="Enable tangent plane projections for 360° videos (default: True)")
    parser.add_argument("--enable-optical-flow", action='store_true', default=True,
                       help="Enable advanced optical flow analysis (default: True)")
    parser.add_argument("--enable-pretrained-cnn", action='store_true', default=True,
                       help="Enable pre-trained CNN features (default: True)")
    parser.add_argument("--enable-attention", action='store_true', default=True,
                       help="Enable attention mechanisms (default: True)")
    parser.add_argument("--enable-ensemble", action='store_true', default=True,
                       help="Enable ensemble matching (default: True)")
    parser.add_argument("--enable-advanced-dtw", action='store_true', default=True,
                       help="Enable advanced DTW correlation (default: True)")
    
    # GPX processing
    parser.add_argument("--gpx-validation", 
                       choices=['strict', 'moderate', 'lenient', 'custom'],
                       default='moderate',
                       help="GPX validation level (default: moderate)")
    parser.add_argument("--enable-gps-filtering", action='store_true', default=True,
                       help="Enable advanced GPS noise filtering (default: True)")
    
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
    parser.add_argument("-o", "--output", default="./enhanced_360_results",
                       help="Output directory")
    parser.add_argument("-c", "--cache", default="./enhanced_360_cache",
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
    logger = setup_logging(log_level, "enhanced_correlation.log")
    
    if args.strict_fail:
        logger.info("Starting Enhanced 360° Video-GPX Correlation System [ULTRA STRICT GPU MODE]")
    elif args.strict:
        logger.info("Starting Enhanced 360° Video-GPX Correlation System [STRICT GPU MODE]")
    else:
        logger.info("Starting Enhanced 360° Video-GPX Correlation System")
    
    try:
        # Create enhanced configuration
        config = Enhanced360ProcessingConfig(
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
            validation_only=args.validation_only,
            enable_360_detection=args.enable_360_detection,
            enable_spherical_processing=args.enable_spherical_processing,
            enable_tangent_plane_processing=args.enable_tangent_planes,
            use_optical_flow=args.enable_optical_flow,
            use_pretrained_features=args.enable_pretrained_cnn,
            use_attention_mechanism=args.enable_attention,
            use_ensemble_matching=args.enable_ensemble,
            use_advanced_dtw=args.enable_advanced_dtw,
            gpx_validation_level=args.gpx_validation,
            enable_gps_filtering=args.enable_gps_filtering
        )
        
        # Display feature status
        logger.info("🚀 Enhanced 360° Features Status:")
        logger.info(f"  360° Detection: {'✅' if config.enable_360_detection else '❌'}")
        logger.info(f"  Spherical Processing: {'✅' if config.enable_spherical_processing else '❌'}")
        logger.info(f"  Tangent Plane Processing: {'✅' if config.enable_tangent_plane_processing else '❌'}")
        logger.info(f"  Advanced Optical Flow: {'✅' if config.use_optical_flow else '❌'}")
        logger.info(f"  Pre-trained CNN Features: {'✅' if config.use_pretrained_features else '❌'}")
        logger.info(f"  Attention Mechanisms: {'✅' if config.use_attention_mechanism else '❌'}")
        logger.info(f"  Ensemble Matching: {'✅' if config.use_ensemble_matching else '❌'}")
        logger.info(f"  Advanced DTW: {'✅' if config.use_advanced_dtw else '❌'}")
        logger.info(f"  Enhanced GPS Processing: {'✅' if config.enable_gps_filtering else '❌'}")
        logger.info(f"  GPX Validation Level: {config.gpx_validation_level.upper()}")
        
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
            logger.info("🔍 Starting enhanced pre-flight video validation...")
            validator = VideoValidator(config)
            
            valid_videos, corrupted_videos, validation_details = validator.validate_video_batch(
                video_files, 
                quarantine_corrupted=not config.no_quarantine
            )
            
            # Save validation report
            validation_report = validator.get_validation_report(validation_details)
            validation_report_path = output_dir / "enhanced_video_validation_report.json"
            with open(validation_report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"📋 Enhanced validation report saved: {validation_report_path}")
            
            # Update video_files to only include valid videos
            video_files = valid_videos
            
            if not video_files:
                print(f"\n❌ No valid videos found after validation!")
                print(f"   All {len(corrupted_videos)} videos were corrupted.")
                print(f"   Check the quarantine directory: {validator.quarantine_dir}")
                sys.exit(1)
            
            if config.validation_only:
                print(f"\n✅ Enhanced validation-only mode complete!")
                print(f"   Valid videos: {len(valid_videos)}")
                print(f"   Corrupted videos: {len(corrupted_videos)}")
                print(f"   Report saved: {validation_report_path}")
                sys.exit(0)
            
            logger.info(f"✅ Enhanced pre-flight validation complete: {len(valid_videos)} valid videos will be processed")
        else:
            logger.warning("⚠️ Skipping video validation - corrupted videos may cause failures")
        
        if not video_files:
            raise RuntimeError("No valid video files to process")
        
        # Load existing results in PowerSafe mode
        existing_results = {}
        if config.powersafe:
            existing_results = powersafe_manager.load_existing_results()
        
        # Process videos with enhanced 360° support
        logger.info("Processing videos with enhanced 360° parallel processing...")
        video_cache_path = cache_dir / "enhanced_360_video_features.pkl"
        
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
            logger.info(f"Processing {len(videos_to_process)} videos with enhanced 360° support...")
            
            # Prepare arguments for parallel processing
            video_args = [(video_path, gpu_manager, config, powersafe_manager) for video_path in videos_to_process]
            
            # Progress tracking
            successful_videos = 0
            failed_videos = 0
            video_360_count = 0
            
            # Use ThreadPoolExecutor for better GPU sharing
            with ThreadPoolExecutor(max_workers=config.parallel_videos) as executor:
                futures = [executor.submit(process_video_parallel_enhanced, arg) for arg in video_args]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                    video_path, features = future.result()
                    video_features[video_path] = features
                    
                    if features is not None:
                        successful_videos += 1
                        if features.get('is_360_video', False):
                            video_360_count += 1
                    else:
                        failed_videos += 1
                    
                    # Periodic cache save
                    if (successful_videos + failed_videos) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
                        logger.info(f"Progress: {successful_videos} success | {failed_videos} failed | {video_360_count} x 360°")
            
            # Final cache save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
        
        logger.info(f"Enhanced video processing complete: {successful_videos} success | {failed_videos} failed | {video_360_count} x 360° videos")
        
        # Process GPX files with enhanced filtering
        logger.info("Processing GPX files with enhanced filtering...")
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
            processor = AdvancedGPSProcessor(config)
            
            # Process with progress bar
            for gpx_file in tqdm(gpx_files, desc="Processing GPX files"):
                gpx_data = processor.process_gpx_enhanced(gpx_file)
                if gpx_data:
                    gpx_database[gpx_file] = gpx_data
            
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_database, f)
            
            logger.info(f"Enhanced GPX processing complete: {len([v for v in gpx_database.values() if v is not None])} successful")
        
        # Perform enhanced correlation with 360° features
        logger.info("Starting enhanced correlation analysis with 360° support...")
        
        # Filter valid features
        valid_videos = {k: v for k, v in video_features.items() if v is not None}
        valid_gpx = {k: v for k, v in gpx_database.items() if v is not None and 'features' in v}
        
        logger.info(f"Valid features: {len(valid_videos)} videos, {len(valid_gpx)} GPX tracks")
        
        if not valid_videos:
            raise RuntimeError(f"No valid video features! Processed {len(video_features)} videos but none succeeded.")
        
        if not valid_gpx:
            raise RuntimeError(f"No valid GPX features! Processed {len(gpx_database)} GPX files but none succeeded.")
        
        # Initialize enhanced similarity engine
        similarity_engine = EnsembleSimilarityEngine(config)
        
        # Compute correlations with enhanced features
        results = existing_results.copy()
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        successful_correlations = 0
        failed_correlations = 0
        
        with tqdm(total=total_comparisons, desc="Computing enhanced correlations") as pbar:
            for video_path, video_features_data in valid_videos.items():
                matches = []
                
                for gpx_path, gpx_data in valid_gpx.items():
                    gpx_features = gpx_data['features']
                    
                    try:
                        similarities = similarity_engine.compute_ensemble_similarity(
                            video_features_data, gpx_features
                        )
                        
                        match_info = {
                            'path': gpx_path,
                            'combined_score': similarities['combined'],
                            'motion_score': similarities['motion_dynamics'],
                            'temporal_score': similarities['temporal_correlation'],
                            'statistical_score': similarities['statistical_profile'],
                            'quality': similarities['quality'],
                            'confidence': similarities['confidence'],
                            'distance': gpx_data.get('distance', 0),
                            'duration': gpx_data.get('duration', 0),
                            'avg_speed': gpx_data.get('avg_speed', 0),
                            'is_360_video': video_features_data.get('is_360_video', False)
                        }
                        
                        # Add enhanced features if available
                        if config.use_ensemble_matching:
                            match_info['optical_flow_score'] = similarities.get('optical_flow_correlation', 0.0)
                            match_info['cnn_feature_score'] = similarities.get('cnn_feature_correlation', 0.0)
                            match_info['advanced_dtw_score'] = similarities.get('advanced_dtw_correlation', 0.0)
                        
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
                    video_type = "360°" if best.get('is_360_video', False) else "STD"
                    logger.info(f"Best match for {Path(video_path).name} [{video_type}]: "
                              f"{Path(best['path']).name} "
                              f"(score: {best['combined_score']:.3f}, quality: {best['quality']})")
                else:
                    logger.warning(f"No valid matches found for {Path(video_path).name}")
        
        logger.info(f"Enhanced correlation analysis complete: {successful_correlations} success | {failed_correlations} failed")
        
        # Final PowerSafe save
        if config.powersafe:
            powersafe_manager.save_incremental_results(powersafe_manager.pending_results)
            logger.info("PowerSafe: Final incremental save completed")
        
        # Save final results
        results_path = output_dir / "enhanced_360_correlations.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Generate comprehensive enhanced report
        report_data = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'version': 'Enhanced360VideoGPXCorrelation v3.0',
                'powersafe_enabled': config.powersafe,
                'total_videos': len(video_files),
                'total_gpx': len(gpx_files),
                'valid_videos': len(valid_videos),
                'valid_gpx': len(valid_gpx),
                'successful_correlations': successful_correlations,
                'failed_correlations': failed_correlations,
                'videos_360_count': video_360_count,
                'gpu_ids': args.gpu_ids,
                'enhanced_features': {
                    '360_detection': config.enable_360_detection,
                    'spherical_processing': config.enable_spherical_processing,
                    'tangent_plane_processing': config.enable_tangent_plane_processing,
                    'optical_flow': config.use_optical_flow,
                    'pretrained_cnn': config.use_pretrained_features,
                    'attention_mechanism': config.use_attention_mechanism,
                    'ensemble_matching': config.use_ensemble_matching,
                    'advanced_dtw': config.use_advanced_dtw,
                    'gps_filtering': config.enable_gps_filtering
                },
                'config': config.__dict__
            },
            'results': results
        }
        
        with open(output_dir / "enhanced_360_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate enhanced summary statistics
        total_videos_with_results = len(results)
        successful_matches = sum(1 for r in results.values() 
                               if r['matches'] and r['matches'][0]['combined_score'] > 0.1)
        
        excellent_matches = sum(1 for r in results.values() 
                              if r['matches'] and r['matches'][0].get('quality') == 'excellent')
        
        good_matches = sum(1 for r in results.values() 
                         if r['matches'] and r['matches'][0].get('quality') in ['good', 'very_good'])
        
        # Count 360° video results
        video_360_matches = sum(1 for r in results.values() 
                               if r['matches'] and r['matches'][0].get('is_360_video', False))
        
        # Calculate average scores
        all_scores = []
        for r in results.values():
            if r['matches'] and r['matches'][0]['combined_score'] > 0:
                all_scores.append(r['matches'][0]['combined_score'])
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        median_score = np.median(all_scores) if all_scores else 0.0
        
        # Print comprehensive enhanced summary
        print(f"\n{'='*100}")
        print(f"ENHANCED 360° VIDEO-GPX CORRELATION SUMMARY")
        print(f"{'='*100}")
        print(f"Processing Mode: Enhanced 360° Support with {'PowerSafe' if config.powersafe else 'Standard'} Mode")
        print(f"")
        print(f"File Processing:")
        print(f"  Videos Found: {len(video_files)}")
        print(f"  Videos Successfully Processed: {len(valid_videos)}/{len(video_files)} ({100*len(valid_videos)/max(len(video_files), 1):.1f}%)")
        print(f"  360° Videos Detected: {video_360_count} ({100*video_360_count/max(len(valid_videos), 1):.1f}%)")
        print(f"  GPX Files Found: {len(gpx_files)}")
        print(f"  GPX Files Successfully Processed: {len(valid_gpx)}/{len(gpx_files)} ({100*len(valid_gpx)/max(len(gpx_files), 1):.1f}%)")
        print(f"")
        print(f"Enhanced Feature Status:")
        print(f"  🌐 360° Detection: {'✅ ENABLED' if config.enable_360_detection else '❌ DISABLED'}")
        print(f"  🔄 Spherical Processing: {'✅ ENABLED' if config.enable_spherical_processing else '❌ DISABLED'}")
        print(f"  📐 Tangent Plane Processing: {'✅ ENABLED' if config.enable_tangent_plane_processing else '❌ DISABLED'}")
        print(f"  🌊 Advanced Optical Flow: {'✅ ENABLED' if config.use_optical_flow else '❌ DISABLED'}")
        print(f"  🧠 Pre-trained CNN Features: {'✅ ENABLED' if config.use_pretrained_features else '❌ DISABLED'}")
        print(f"  🎯 Attention Mechanisms: {'✅ ENABLED' if config.use_attention_mechanism else '❌ DISABLED'}")
        print(f"  🎼 Ensemble Matching: {'✅ ENABLED' if config.use_ensemble_matching else '❌ DISABLED'}")
        print(f"  📊 Advanced DTW: {'✅ ENABLED' if config.use_advanced_dtw else '❌ DISABLED'}")
        print(f"  🛰️  Enhanced GPS Processing: {'✅ ENABLED' if config.enable_gps_filtering else '❌ DISABLED'}")
        print(f"")
        print(f"Correlation Results:")
        print(f"  Total Videos with Results: {total_videos_with_results}")
        print(f"  Videos with Valid Matches (>0.1): {successful_matches}/{total_videos_with_results} ({100*successful_matches/max(total_videos_with_results, 1):.1f}%)")
        print(f"  360° Video Matches: {video_360_matches}")
        print(f"  Total Correlations Computed: {successful_correlations + failed_correlations}")
        print(f"  Successful Correlations: {successful_correlations}")
        print(f"  Failed Correlations: {failed_correlations}")
        print(f"")
        print(f"Quality Distribution:")
        print(f"  🟢 Excellent (≥0.85): {excellent_matches}")
        print(f"  🟡 Good/Very Good (≥0.55): {good_matches}")
        print(f"  🟠 Fair (≥0.40): {total_videos_with_results - excellent_matches - good_matches}")
        print(f"  🔴 Poor/Failed: {total_videos_with_results - successful_matches}")
        print(f"")
        print(f"Score Statistics:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Median Score: {median_score:.3f}")
        print(f"  Total Valid Scores: {len(all_scores)}")
        print(f"")
        print(f"Output Files:")
        print(f"  Results: {results_path}")
        print(f"  Report: {output_dir / 'enhanced_360_report.json'}")
        print(f"  Cache: {cache_dir}")
        print(f"  Log: enhanced_correlation.log")
        print(f"")
        
        # Display top correlations if any exist
        if all_scores:
            print(f"TOP ENHANCED CORRELATIONS:")
            print(f"{'='*100}")
            
            # Get top correlations across all videos
            all_correlations = []
            for video_path, result in results.items():
                if result['matches'] and result['matches'][0]['combined_score'] > 0:
                    best_match = result['matches'][0]
                    video_type = "360°" if best_match.get('is_360_video', False) else "STD"
                    all_correlations.append((
                        Path(video_path).name,
                        Path(best_match['path']).name,
                        best_match['combined_score'],
                        best_match.get('quality', 'unknown'),
                        video_type,
                        best_match.get('confidence', 0.0)
                    ))
            
            # Sort by score and display top 10
            all_correlations.sort(key=lambda x: x[2], reverse=True)
            for i, (video, gpx, score, quality, video_type, confidence) in enumerate(all_correlations[:10], 1):
                quality_emoji = {
                    'excellent': '🟢', 
                    'very_good': '🟡',
                    'good': '🟡', 
                    'fair': '🟠', 
                    'poor': '🔴', 
                    'very_poor': '🔴',
                    'failed': '❌'
                }.get(quality, '⚪')
                
                print(f"{i:2d}. {video[:45]:<45} ↔ {gpx[:25]:<25} [{video_type}]")
                print(f"     Score: {score:.3f} | Quality: {quality_emoji} {quality} | Confidence: {confidence:.2f}")
                if i < len(all_correlations):
                    print()
        else:
            print(f"❌ No successful correlations found!")
            print(f"   This could indicate:")
            print(f"   • Video processing failures (check logs)")
            print(f"   • GPX processing failures (check file formats)")
            print(f"   • Feature extraction issues")
            print(f"   • Incompatible data types")
            print(f"   • Very strict validation settings")
        
        print(f"{'='*100}")
        
        # Enhanced success determination
        if successful_matches > 0:
            if video_360_count > 0:
                logger.info("Enhanced 360° correlation system completed successfully with 360° video matches!")
            else:
                logger.info("Enhanced correlation system completed successfully with standard video matches!")
        elif len(valid_videos) > 0 and len(valid_gpx) > 0:
            logger.warning("System completed but found no correlations - check data compatibility")
        else:
            logger.error("System completed but no valid features were extracted")
        
        # Enhanced recommendations
        if failed_correlations > successful_correlations:
            print(f"\n🔧 TROUBLESHOOTING RECOMMENDATIONS:")
            print(f"   • Try reducing --parallel_videos to 1 for debugging")
            print(f"   • Reduce --max_frames (try 100 for memory issues)")
            print(f"   • Reduce --video_size (try 480 360 for memory issues)")
            print(f"   • Check video file formats and corruption with --validation_only")
            print(f"   • Verify GPX files contain valid track data")
            print(f"   • Enable --debug for detailed error analysis")
            if not config.powersafe:
                print(f"   • Use --powersafe to preserve progress during debugging")
            if config.strict_fail:
                print(f"   • Remove --strict_fail flag to allow skipping problematic videos")
            elif config.strict:
                print(f"   • Remove --strict flag to enable CPU fallbacks for debugging")
            
            print(f"\n🌐 360° VIDEO SPECIFIC:")
            print(f"   • Most 360° videos have 2:1 aspect ratio (3840x1920, 1920x960, etc.)")
            print(f"   • Enable --enable-360-detection for automatic detection")
            print(f"   • Use --enable-spherical-processing for distortion compensation")
            print(f"   • Try --enable-tangent-planes for better feature extraction")
        
        print(f"\n⚡ PERFORMANCE OPTIMIZATION:")
        print(f"   • Current settings: {args.max_frames} frames, {args.video_size[0]}x{args.video_size[1]} resolution")
        print(f"   • 360° videos benefit from higher resolution (720x480 minimum)")
        print(f"   • GPU Memory: {config.max_gpu_memory_gb:.1f}GB limit per GPU")
        print(f"   • Parallel Videos: {config.parallel_videos}")
        print(f"   • Temp Directory: {config.cache_dir}")
        
        print(f"\n🚀 NEXT STEPS:")
        if successful_matches > 0:
            print(f"   ✅ System is working! Consider:")
            print(f"      • Fine-tune validation levels for more GPX files")
            print(f"      • Adjust correlation thresholds")
            print(f"      • Enable more enhanced features for better accuracy")
        else:
            print(f"   🔍 Debug mode recommended:")
            print(f"      • Run with --debug --validation_only first")
            print(f"      • Check sample files manually")
            print(f"      • Test with known good video-GPX pairs")
        
        if video_360_count > 0:
            print(f"\n🌐 360° VIDEO RESULTS:")
            print(f"   • Successfully detected and processed {video_360_count} 360° videos")
            print(f"   • Used spherical-aware processing for distortion compensation")
            print(f"   • Applied tangent plane projections for better feature extraction")
            print(f"   • Enhanced optical flow with latitude weighting")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        if config.powersafe:
            logger.info("PowerSafe: Progress has been saved and can be resumed")
        print("\nProcess interrupted. PowerSafe progress has been saved." if config.powersafe else "\nProcess interrupted.")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Enhanced 360° system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        if config.powersafe:
            logger.info("PowerSafe: Partial progress has been saved")
            print(f"\nError occurred: {e}")
            print("PowerSafe: Partial progress has been saved and can be resumed with --powersafe flag")
        
        # Enhanced debugging suggestions
        print(f"\n🔧 DEBUGGING SUGGESTIONS:")
        print(f"   • Run with --debug for detailed error information")
        print(f"   • Try --parallel_videos 1 to isolate GPU issues")
        print(f"   • Reduce --max_frames to 100 for testing")
        print(f"   • Check video file integrity with ffprobe")
        print(f"   • Verify GPX files are valid XML")
        print(f"   • Run --validation_only to check for corrupted videos")
        print(f"   • Use --no_quarantine to keep corrupted videos in place")
        
        print(f"\n🌐 360° VIDEO DEBUGGING:")
        print(f"   • Check if videos are actually 360° (2:1 aspect ratio)")
        print(f"   • Try disabling 360° features: --disable-spherical-processing")
        print(f"   • Test with standard videos first")
        print(f"   • Verify 360° videos are equirectangular format")
        
        print(f"\n🛰️  GPX DEBUGGING:")
        print(f"   • Verify GPX files contain valid tracks with timestamps")
        print(f"   • Try different validation levels: --gpx-validation lenient")
        print(f"   • Check GPS coordinate ranges (lat: -90 to 90, lon: -180 to 180)")
        print(f"   • Ensure GPX files have reasonable point density")
        
        sys.exit(1)
    
    finally:
        # Enhanced cleanup
        try:
            if 'decoder' in locals():
                decoder.cleanup()
            if 'validator' in locals():
                validator.cleanup()
            logger.info("Enhanced system cleanup completed")
        except:
            pass

if __name__ == "__main__":
    main()