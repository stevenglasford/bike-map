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
from scipy.signal import correlate, find_peaks, butter, filtfilt
from scipy.spatial.distance import cdist
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
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from skimage.feature import hog
import warnings
import logging
from tqdm import tqdm
import gc
from collections import defaultdict
import time

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

class DualGPUVideoAnalyzer:
    def __init__(self, gpu_ids=[0, 1], batch_size=4):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.devices = [torch.device(f'cuda:{i}') for i in gpu_ids if torch.cuda.is_available() and i < torch.cuda.device_count()]
        
        if not self.devices:
            self.devices = [torch.device('cpu')]
            logger.warning("No GPUs available, falling back to CPU")
        else:
            logger.info(f"Using GPUs: {self.devices}")
        
        # Initialize CUDA streams for parallel processing
        self.streams = [torch.cuda.Stream(device=device) for device in self.devices] if torch.cuda.is_available() else []
        
        # Initialize optical flow calculators for each GPU
        self.flow_calcs = []
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            for gpu_id in gpu_ids[:cv2.cuda.getCudaEnabledDeviceCount()]:
                cv2.cuda.setDevice(gpu_id)
                flow_calc = cv2.cuda_FarnebackOpticalFlow.create(
                    numLevels=5, pyrScale=0.5, fastPyramids=True,
                    winSize=13, numIters=3, polyN=5, polySigma=1.1, flags=0
                )
                self.flow_calcs.append(flow_calc)
            logger.info(f"Initialized {len(self.flow_calcs)} CUDA optical flow calculators")
        
        # Feature extraction CNN (lightweight MobileNetV2)
        self.feature_extractors = []
        for device in self.devices:
            try:
                model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
                # Keep only the feature extraction part
                model = nn.Sequential(*list(model.features.children())[:-1])
                model.eval().to(device)
                self.feature_extractors.append(model)
            except Exception as e:
                logger.warning(f"Could not load MobileNetV2, using simple CNN: {e}")
                # Fallback to simple CNN
                model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                ).to(device)
                model.eval()
                self.feature_extractors.append(model)
    
    def extract_motion_features_batch(self, video_paths, max_workers=4):
        """Extract features from multiple videos in parallel using both GPUs"""
        results = {}
        
        # Split videos between GPUs
        gpu_assignments = defaultdict(list)
        for i, video_path in enumerate(video_paths):
            gpu_idx = i % len(self.devices)
            gpu_assignments[gpu_idx].append(video_path)
        
        # Process videos on each GPU in parallel
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for gpu_idx, videos in gpu_assignments.items():
                future = executor.submit(
                    self._process_videos_on_gpu, 
                    videos, 
                    gpu_idx
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                gpu_results = future.result()
                results.update(gpu_results)
        
        return results
    
    def _process_videos_on_gpu(self, video_paths, gpu_idx):
        """Process videos on a specific GPU"""
        device = self.devices[gpu_idx]
        flow_calc = self.flow_calcs[gpu_idx] if gpu_idx < len(self.flow_calcs) else None
        feature_extractor = self.feature_extractors[gpu_idx]
        
        results = {}
        for video_path in tqdm(video_paths, desc=f"GPU {gpu_idx}"):
            try:
                features = self._extract_single_video_features(
                    video_path, device, flow_calc, feature_extractor
                )
                if features and any(len(v) > 0 for k, v in features.items() if isinstance(v, list)):
                    results[video_path] = features
                else:
                    logger.warning(f"No valid features extracted from {video_path}")
                    results[video_path] = None
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()
                results[video_path] = None
            
            # Clear GPU memory periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _extract_single_video_features(self, video_path, device, flow_calc, feature_extractor):
        """Extract comprehensive features from a single video"""
        # Try different backends for better codec support
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                break
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Adaptive sampling based on video length
        sample_rate = min(2, max(0.5, 30 / duration))  # Sample more for shorter videos
        
        features = {
            'motion_magnitude': [],
            'motion_direction': [],
            'acceleration': [],
            'jerk': [],
            'rotation': [],
            'scene_features': [],
            'color_histogram': [],
            'edge_density': [],
            'optical_flow_complexity': [],
            'temporal_gradient': []
        }
        
        prev_gray = None
        prev_features = None
        prev_motion = None
        prev_accel = None
        
        # Process frames
        frame_interval = max(1, int(fps / sample_rate))
        frame_indices = range(0, frame_count, frame_interval)
        
        # Batch processing for GPU efficiency
        batch_frames = []
        batch_indices = []
        failed_frames = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                failed_frames += 1
                continue
            
            try:
                # Resize for processing
                frame = cv2.resize(frame, (640, 360))
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                
                # Process batch when full
                if len(batch_frames) >= self.batch_size:
                    self._process_frame_batch(
                        batch_frames, batch_indices, features, 
                        device, flow_calc, feature_extractor,
                        prev_gray, prev_features, prev_motion, prev_accel
                    )
                    
                    # Update previous values
                    prev_gray = cv2.cvtColor(batch_frames[-1], cv2.COLOR_BGR2GRAY)
                    prev_features = features['scene_features'][-1] if features['scene_features'] else None
                    prev_motion = features['motion_magnitude'][-1] if features['motion_magnitude'] else None
                    prev_accel = features['acceleration'][-1] if features['acceleration'] else None
                    
                    batch_frames = []
                    batch_indices = []
            except Exception as e:
                logger.debug(f"Frame processing error: {e}")
                failed_frames += 1
        
        # Process remaining frames
        if batch_frames:
            try:
                self._process_frame_batch(
                    batch_frames, batch_indices, features,
                    device, flow_calc, feature_extractor,
                    prev_gray, prev_features, prev_motion, prev_accel
                )
            except Exception as e:
                logger.debug(f"Final batch processing error: {e}")
        
        cap.release()
        
        logger.info(f"Processed {video_path}: {len(features['motion_magnitude'])} samples, {failed_frames} failed frames")
        
        # Post-process features
        features = self._post_process_features(features)
        features['duration'] = duration
        features['fps'] = fps
        
        return features
    
    def _process_frame_batch(self, frames, indices, features, device, flow_calc, 
                           feature_extractor, prev_gray, prev_features, prev_motion, prev_accel):
        """Process a batch of frames efficiently on GPU"""
        # Convert to tensor batch
        frame_tensor = torch.stack([
            transforms.ToTensor()(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]).to(device)
        
        # Extract deep features
        with torch.no_grad():
            try:
                if torch.cuda.is_available():
                    with torch.cuda.stream(self.streams[self.devices.index(device)]):
                        deep_features = feature_extractor(frame_tensor)
                        # Handle different output shapes
                        if len(deep_features.shape) == 4:
                            scene_features = F.adaptive_avg_pool2d(deep_features, (1, 1)).squeeze()
                        else:
                            scene_features = deep_features.squeeze()
                        
                        # Ensure 2D output
                        if len(scene_features.shape) == 1:
                            scene_features = scene_features.unsqueeze(0)
                        scene_features = scene_features.cpu().numpy()
                else:
                    deep_features = feature_extractor(frame_tensor)
                    if len(deep_features.shape) == 4:
                        scene_features = F.adaptive_avg_pool2d(deep_features, (1, 1)).squeeze()
                    else:
                        scene_features = deep_features.squeeze()
                    
                    if len(scene_features.shape) == 1:
                        scene_features = scene_features.unsqueeze(0)
                    scene_features = scene_features.cpu().numpy()
            except Exception as e:
                logger.warning(f"Error extracting deep features: {e}")
                # Fallback to simple features
                scene_features = np.zeros((len(frames), 64))  # Default feature size
        
        # Process each frame
        for i, (frame, scene_feat) in enumerate(zip(frames, scene_features)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Color histogram
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features['color_histogram'].append(hist)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features['edge_density'].append(edge_density)
            
            # Scene features
            features['scene_features'].append(scene_feat)
            
            # Optical flow features
            if prev_gray is not None:
                if flow_calc and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    # GPU optical flow
                    gpu_prev = cv2.cuda_GpuMat()
                    gpu_curr = cv2.cuda_GpuMat()
                    gpu_prev.upload(prev_gray)
                    gpu_curr.upload(gray)
                    flow = flow_calc.calc(gpu_prev, gpu_curr, None).download()
                else:
                    # CPU fallback
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                
                # Extract motion features
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                direction = np.arctan2(flow[..., 1], flow[..., 0])
                
                # Motion statistics
                motion_mag = np.mean(magnitude)
                motion_dir = np.mean(np.exp(1j * direction))
                flow_complexity = np.std(magnitude)
                
                # Rotation estimation
                rotation = self._estimate_rotation_gpu(flow)
                
                # Temporal gradient
                if i > 0:
                    temporal_grad = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                else:
                    temporal_grad = 0
                
                features['motion_magnitude'].append(motion_mag)
                features['motion_direction'].append(np.abs(motion_dir))
                features['optical_flow_complexity'].append(flow_complexity)
                features['rotation'].append(rotation)
                features['temporal_gradient'].append(temporal_grad)
                
                # Higher-order motion features
                if prev_motion is not None:
                    accel = motion_mag - prev_motion
                    features['acceleration'].append(accel)
                    
                    if prev_accel is not None:
                        jerk = accel - prev_accel
                        features['jerk'].append(jerk)
                    else:
                        features['jerk'].append(0)
                else:
                    features['acceleration'].append(0)
                    features['jerk'].append(0)
            else:
                # Initialize for first frame
                for key in ['motion_magnitude', 'motion_direction', 'optical_flow_complexity', 
                           'rotation', 'temporal_gradient', 'acceleration', 'jerk']:
                    features[key].append(0)
            
            prev_gray = gray
    
    def _estimate_rotation_gpu(self, flow):
        """Estimate rotation using GPU-accelerated computation"""
        h, w = flow.shape[:2]
        cx, cy = w//2, h//2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        x = x - cx
        y = y - cy
        
        # Calculate rotation using cross product
        rotation = np.mean(x * flow[..., 1] - y * flow[..., 0]) / (np.mean(x**2 + y**2) + 1e-8)
        
        return rotation
    
    def _post_process_features(self, features):
        """Post-process and normalize features"""
        processed = {}
        
        for key, values in features.items():
            if values and isinstance(values[0], (int, float, np.number)):
                # Convert to numpy array
                arr = np.array(values)
                
                # Apply smoothing
                if len(arr) > 5:
                    arr = self._adaptive_smooth(arr)
                
                # Normalize
                if np.std(arr) > 1e-8:
                    arr = (arr - np.mean(arr)) / np.std(arr)
                
                processed[key] = arr
            else:
                processed[key] = values
        
        return processed
    
    def _adaptive_smooth(self, signal, max_window=7):
        """Apply adaptive smoothing based on signal characteristics"""
        # Estimate noise level
        noise_level = np.median(np.abs(np.diff(signal)))
        
        # Adaptive window size
        window_size = min(max_window, max(3, int(len(signal) * 0.05)))
        
        # Apply Savitzky-Golay filter for better feature preservation
        if len(signal) > window_size:
            from scipy.signal import savgol_filter
            return savgol_filter(signal, window_size, min(3, window_size-1))
        
        return signal

class AdvancedGPXProcessor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
    
    def parse_gpx_batch(self, gpx_paths, max_workers=None):
        """Parse multiple GPX files with enhanced feature extraction"""
        if max_workers is None:
            max_workers = min(32, mp.cpu_count())
        
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.parse_single_gpx, path): path 
                      for path in gpx_paths}
            
            for future in tqdm(as_completed(futures), total=len(gpx_paths), desc="Parsing GPX files"):
                path = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[path] = result
                except Exception as e:
                    logger.error(f"Error parsing {path}: {e}")
        
        return results
    
    def parse_single_gpx(self, gpx_path):
        """Parse GPX with enhanced feature extraction"""
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
            
            # Calculate enhanced features
            features = self._calculate_enhanced_features(df)
            
            # Calculate metadata
            duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
            distance = self._calculate_total_distance(df)
            
            return {
                'df': df,
                'features': features,
                'duration': duration,
                'distance': distance,
                'start_time': df['timestamp'].iloc[0],
                'end_time': df['timestamp'].iloc[-1],
                'point_count': len(df)
            }
        
        except Exception as e:
            logger.error(f"Error parsing {gpx_path}: {e}")
            return None
    
    def _calculate_enhanced_features(self, df):
        """Calculate comprehensive motion features with smoothing"""
        # Add time-based column
        df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
        features = {
            'speed': [],
            'acceleration': [],
            'jerk': [],
            'bearing': [],
            'bearing_change': [],
            'elevation_change': [],
            'curvature': [],
            'stop_probability': [],
            'speed_variance': [],
            'path_complexity': []
        }
        
        # Pre-calculate distances and bearings
        lats = df['lat'].values
        lons = df['lon'].values
        times = df['seconds'].values
        elevations = df['elevation'].values
        
        # Vectorized distance calculation
        distances = self._haversine_vectorized(
            lats[:-1], lons[:-1], lats[1:], lons[1:]
        )
        
        # Vectorized bearing calculation
        bearings = self._bearing_vectorized(
            lats[:-1], lons[:-1], lats[1:], lons[1:]
        )
        
        # Calculate features
        for i in range(len(df)):
            if i == 0:
                # Initialize first values
                for key in features:
                    features[key].append(0.0)
                continue
            
            # Time difference
            dt = times[i] - times[i-1]
            if dt <= 0:
                dt = 1.0
            
            # Speed
            dist = distances[i-1]
            speed = (dist * 3600) / dt  # mph
            features['speed'].append(speed)
            
            # Stop probability (low speed indicator)
            stop_prob = np.exp(-speed / 2)  # Exponential decay
            features['stop_probability'].append(stop_prob)
            
            # Bearing
            bearing = bearings[i-1]
            features['bearing'].append(bearing)
            
            # Elevation change
            elev_change = elevations[i] - elevations[i-1]
            features['elevation_change'].append(elev_change)
            
            # Higher-order features
            if i >= 2:
                # Acceleration
                prev_speed = features['speed'][-2]
                accel = (speed - prev_speed) / dt
                features['acceleration'].append(accel)
                
                # Bearing change
                prev_bearing = features['bearing'][-2]
                bearing_change = self._angle_difference(bearing, prev_bearing)
                features['bearing_change'].append(abs(bearing_change))
                
                # Curvature
                curvature = abs(bearing_change) / max(dist, 0.001)
                features['curvature'].append(curvature)
                
                # Path complexity (rolling measurement)
                if i >= 10:
                    recent_bearings = features['bearing_change'][-10:]
                    path_complexity = np.std(recent_bearings)
                else:
                    path_complexity = 0
                features['path_complexity'].append(path_complexity)
                
                if i >= 3:
                    # Jerk
                    prev_accel = features['acceleration'][-2]
                    jerk = (accel - prev_accel) / dt
                    features['jerk'].append(jerk)
                    
                    # Speed variance (rolling)
                    if i >= 10:
                        recent_speeds = features['speed'][-10:]
                        speed_var = np.var(recent_speeds)
                    else:
                        speed_var = 0
                    features['speed_variance'].append(speed_var)
                else:
                    features['jerk'].append(0)
                    features['speed_variance'].append(0)
            else:
                # Fill with zeros for consistency
                features['acceleration'].append(0)
                features['bearing_change'].append(0)
                features['curvature'].append(0)
                features['path_complexity'].append(0)
                features['jerk'].append(0)
                features['speed_variance'].append(0)
        
        # Apply filtering and normalization
        for key in features:
            arr = np.array(features[key])
            
            # Remove outliers using IQR
            if len(arr) > 10 and key in ['speed', 'acceleration', 'jerk']:
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                arr = np.clip(arr, lower, upper)
            
            # Apply smoothing
            if len(arr) > 5:
                # Use Butterworth filter for smoother results
                try:
                    b, a = butter(3, 0.3)
                    arr = filtfilt(b, a, arr)
                except Exception as e:
                    # Fallback to simple moving average if filter fails
                    logger.debug(f"Butterworth filter failed, using moving average: {e}")
                    window = min(5, len(arr) // 3)
                    if window > 1:
                        arr = np.convolve(arr, np.ones(window)/window, mode='same')
            
            features[key] = arr
        
        return features
    
    def _haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """Vectorized haversine distance calculation"""
        R = 3958.8  # Earth radius in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _bearing_vectorized(self, lat1, lon1, lat2, lon2):
        """Vectorized bearing calculation"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        return np.degrees(np.arctan2(y, x))
    
    def _angle_difference(self, a1, a2):
        """Calculate smallest angle difference"""
        diff = (a2 - a1 + 180) % 360 - 180
        return diff
    
    def _calculate_total_distance(self, df):
        """Calculate total distance efficiently"""
        lats = df['lat'].values
        lons = df['lon'].values
        
        if len(lats) < 2:
            return 0
        
        distances = self._haversine_vectorized(
            lats[:-1], lons[:-1], lats[1:], lons[1:]
        )
        
        return np.sum(distances)

class EnhancedGPUCorrelator:
    def __init__(self, gpu_ids=[0, 1]):
        self.devices = [torch.device(f'cuda:{i}') for i in gpu_ids if torch.cuda.is_available() and i < torch.cuda.device_count()]
        if not self.devices:
            self.devices = [torch.device('cpu')]
        
        logger.info(f"Correlator using devices: {self.devices}")
        
        # Initialize neural network for learned similarity
        self.similarity_networks = []
        for device in self.devices:
            net = self._create_similarity_network().to(device)
            self.similarity_networks.append(net)
    
    def _create_similarity_network(self):
        """Create a neural network for learned similarity matching"""
        class SimilarityNetwork(nn.Module):
            def __init__(self, input_dim=256):
                super().__init__()
                self.fc1 = nn.Linear(input_dim * 2, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, 1)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x1, x2):
                x = torch.cat([x1, x2], dim=-1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.sigmoid(self.fc4(x))
                return x
        
        return SimilarityNetwork()
    
    def correlate_all(self, video_features_dict, gpx_database, output_dir, top_k=5):
        """Correlate all videos with all GPX files and generate accuracy report"""
        results = {}
        
        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process each video
        video_paths = list(video_features_dict.keys())
        logger.info(f"Correlating {len(video_paths)} videos with {len(gpx_database)} GPX files")
        
        # Split videos among GPUs
        gpu_assignments = defaultdict(list)
        for i, video_path in enumerate(video_paths):
            gpu_idx = i % len(self.devices)
            gpu_assignments[gpu_idx].append(video_path)
        
        # Process in parallel on multiple GPUs
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = []
            for gpu_idx, videos in gpu_assignments.items():
                future = executor.submit(
                    self._correlate_on_gpu,
                    videos, video_features_dict, gpx_database, 
                    gpu_idx, top_k
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                gpu_results = future.result()
                results.update(gpu_results)
        
        # Generate accuracy report
        self._generate_accuracy_report(results, output_path)
        
        return results
    
    def _correlate_on_gpu(self, video_paths, video_features_dict, gpx_database, gpu_idx, top_k):
        """Correlate videos on specific GPU"""
        device = self.devices[gpu_idx]
        similarity_net = self.similarity_networks[gpu_idx]
        
        results = {}
        
        for video_path in tqdm(video_paths, desc=f"GPU {gpu_idx} correlation"):
            video_features = video_features_dict[video_path]
            if video_features is None:
                results[video_path] = None
                continue
            
            # Find best matches
            matches = self._find_matches_enhanced(
                video_features, gpx_database, device, similarity_net, top_k
            )
            
            results[video_path] = {
                'matches': matches,
                'video_duration': video_features.get('duration', 0)
            }
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _find_matches_enhanced(self, video_features, gpx_database, device, similarity_net, top_k):
        """Enhanced matching using multiple correlation methods"""
        candidates = []
        
        # Create comprehensive video signature
        video_sig = self._create_enhanced_signature(video_features, device)
        
        for gpx_path, gpx_data in gpx_database.items():
            if gpx_data is None:
                continue
            
            # Create GPX signature
            gpx_sig = self._create_enhanced_signature(gpx_data['features'], device)
            
            # Multi-method scoring
            scores = {}
            
            # 1. FFT-based correlation
            fft_score = self._fft_correlation(video_sig, gpx_sig, device)
            if fft_score > 0:
                scores['fft'] = fft_score
            
            # 2. DTW-based alignment
            dtw_score = self._dtw_similarity(video_sig, gpx_sig, device)
            if dtw_score > 0:
                scores['dtw'] = dtw_score
            
            # 3. Statistical similarity
            stats_score = self._statistical_similarity(video_sig, gpx_sig)
            if stats_score > 0:
                scores['stats'] = stats_score
            
            # 4. Neural network similarity (if enough features)
            if 'embedding' in video_sig and 'embedding' in gpx_sig:
                try:
                    neural_score = self._neural_similarity(
                        video_sig['embedding'], gpx_sig['embedding'], 
                        device, similarity_net
                    )
                    if neural_score > 0:
                        scores['neural'] = neural_score
                except Exception as e:
                    logger.debug(f"Neural similarity failed: {e}")
            
            # 5. Temporal pattern matching
            temporal_score = self._temporal_pattern_score(video_sig, gpx_sig)
            if temporal_score > 0:
                scores['temporal'] = temporal_score
            
            # Only proceed if we have some valid scores
            if not scores:
                continue
            
            # Combine scores with adaptive weights
            if len(scores) >= 3:
                # Full weight distribution when we have enough methods
                weights = {
                    'fft': 0.25,
                    'dtw': 0.25,
                    'stats': 0.15,
                    'neural': 0.20,
                    'temporal': 0.15
                }
            else:
                # Equal weights when we have fewer methods
                weight_per_method = 1.0 / len(scores)
                weights = {k: weight_per_method for k in scores}
            
            combined_score = sum(weights.get(k, 0) * scores.get(k, 0) 
                               for k in scores)
            
            # Duration compatibility check (more lenient)
            duration_ratio = min(video_features.get('duration', 1), gpx_data['duration']) / \
                           max(video_features.get('duration', 1), gpx_data['duration'])
            
            # Accept wider duration range
            if duration_ratio < 0.3:  # Very lenient - 30% duration match
                combined_score *= 0.5  # Penalize but don't eliminate
            
            # Always add candidates with any positive score
            if combined_score > 0:
                candidates.append({
                    'path': gpx_path,
                    'combined_score': combined_score,
                    'scores': scores,
                    'duration_ratio': duration_ratio,
                    'distance': gpx_data.get('distance', 0),
                    'point_count': gpx_data.get('point_count', 0)
                })
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Perform fine alignment on top candidates (only if we have candidates)
        if candidates:
            top_candidates = candidates[:min(top_k * 2, len(candidates))]
            
            for candidate in top_candidates:
                try:
                    gpx_data = gpx_database[candidate['path']]
                    offset, alignment_score = self._fine_align_gpu(
                        video_features, gpx_data, device
                    )
                    candidate['alignment_score'] = alignment_score
                    candidate['time_offset'] = offset
                    # More balanced final score calculation
                    candidate['final_score'] = candidate['combined_score'] * (0.5 + 0.5 * alignment_score)
                except Exception as e:
                    logger.debug(f"Fine alignment failed for {candidate['path']}: {e}")
                    candidate['alignment_score'] = 0
                    candidate['time_offset'] = 0
                    candidate['final_score'] = candidate['combined_score'] * 0.5
            
            # Re-sort by final score
            top_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            return top_candidates[:top_k]
        else:
            return []
    
    def _create_enhanced_signature(self, features, device):
        """Create comprehensive multi-modal signature"""
        signature = {}
        
        # Process each feature type
        for key, values in features.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                # Skip if values are multidimensional (like scene features)
                if len(values.shape) > 1:
                    continue
                    
                # Normalize
                if np.std(values) > 1e-8:
                    normalized = (values - np.mean(values)) / np.std(values)
                else:
                    normalized = values
                
                # Multiple representations
                # 1. Frequency domain
                if len(normalized) > 1:
                    fft = np.fft.fft(normalized)
                    fft_mag = np.abs(fft)[:len(fft)//2]
                    signature[f'{key}_fft'] = fft_mag
                
                # 2. Statistical features
                stats = np.array([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    np.percentile(values, 25),
                    np.percentile(values, 50),
                    np.percentile(values, 75),
                    np.sum(np.abs(np.diff(values))) if len(values) > 1 else 0,  # Total variation
                    len(values)  # Signal length
                ])
                signature[f'{key}_stats'] = stats
                
                # 3. Temporal patterns
                if len(values) > 10:
                    # Find peaks and valleys
                    try:
                        peaks, _ = find_peaks(normalized, prominence=0.5)
                        valleys, _ = find_peaks(-normalized, prominence=0.5)
                    except:
                        peaks = valleys = np.array([])
                    
                    pattern = np.zeros(20)
                    if len(peaks) > 0:
                        pattern[:min(10, len(peaks))] = normalized[peaks[:10]]
                    if len(valleys) > 0:
                        pattern[10:10+min(10, len(valleys))] = normalized[valleys[:10]]
                    signature[f'{key}_pattern'] = pattern
                
                # 4. Raw signal (downsampled if needed)
                if len(normalized) > 100:
                    downsampled = np.interp(
                        np.linspace(0, len(normalized)-1, 100),
                        np.arange(len(normalized)),
                        normalized
                    )
                else:
                    downsampled = normalized
                signature[f'{key}_signal'] = downsampled
        
        # Create embedding vector for neural similarity
        embedding_parts = []
        for key in ['motion_magnitude_stats', 'speed_stats', 'acceleration_stats']:
            if key in signature:
                embedding_parts.append(signature[key])
        
        if embedding_parts:
            embedding = np.concatenate(embedding_parts)
            # Pad or truncate to fixed size
            if len(embedding) < 256:
                embedding = np.pad(embedding, (0, 256 - len(embedding)))
            else:
                embedding = embedding[:256]
            signature['embedding'] = embedding
        
        return signature
    
    def _fft_correlation(self, sig1, sig2, device):
        """FFT-based correlation score"""
        scores = []
        
        for key in sig1:
            if key.endswith('_fft') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                # Skip if empty
                if len(s1) == 0 or len(s2) == 0:
                    continue
                
                # Match lengths
                min_len = min(len(s1), len(s2))
                s1_norm = s1[:min_len]
                s2_norm = s2[:min_len]
                
                if len(s1_norm) > 0:
                    # Compute correlation in frequency domain
                    try:
                        if device.type == 'cuda':
                            t1 = torch.tensor(s1_norm, device=device, dtype=torch.float32)
                            t2 = torch.tensor(s2_norm, device=device, dtype=torch.float32)
                            
                            # Normalized correlation
                            corr = torch.sum(t1 * t2) / (torch.norm(t1) * torch.norm(t2) + 1e-8)
                            scores.append(abs(corr.item()))
                        else:
                            corr = np.dot(s1_norm, s2_norm) / (np.linalg.norm(s1_norm) * np.linalg.norm(s2_norm) + 1e-8)
                            scores.append(abs(corr))
                    except Exception as e:
                        logger.debug(f"FFT correlation error: {e}")
        
        return np.mean(scores) if scores else 0.0
    
    def _dtw_similarity(self, sig1, sig2, device, max_samples=100):
        """GPU-accelerated DTW similarity"""
        scores = []
        
        for key in sig1:
            if key.endswith('_signal') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                # Downsample for efficiency
                if len(s1) > max_samples:
                    s1 = s1[::len(s1)//max_samples]
                if len(s2) > max_samples:
                    s2 = s2[::len(s2)//max_samples]
                
                if device.type == 'cuda':
                    # GPU DTW using distance matrix
                    t1 = torch.tensor(s1, device=device, dtype=torch.float32)
                    t2 = torch.tensor(s2, device=device, dtype=torch.float32)
                    
                    # Compute distance matrix
                    dist_matrix = torch.cdist(t1.unsqueeze(1), t2.unsqueeze(1), p=2).squeeze()
                    
                    # DTW with CUDA kernels
                    dtw_dist = self._cuda_dtw(dist_matrix)
                    
                    # Convert to similarity score
                    similarity = 1.0 / (1.0 + dtw_dist / len(s1))
                    scores.append(similarity)
                else:
                    # CPU fallback
                    dtw_dist = self._cpu_dtw(s1, s2)
                    similarity = 1.0 / (1.0 + dtw_dist / len(s1))
                    scores.append(similarity)
        
        return np.mean(scores) if scores else 0.0
    
    def _cuda_dtw(self, dist_matrix):
        """CUDA-accelerated DTW computation"""
        n, m = dist_matrix.shape
        
        # Initialize DTW matrix
        dtw = torch.full((n+1, m+1), float('inf'), device=dist_matrix.device)
        dtw[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = dist_matrix[i-1, j-1]
                dtw[i, j] = cost + torch.min(torch.stack([
                    dtw[i-1, j],      # insertion
                    dtw[i, j-1],      # deletion
                    dtw[i-1, j-1]     # match
                ]))
        
        return dtw[n, m].item()
    
    def _cpu_dtw(self, s1, s2):
        """CPU DTW implementation"""
        n, m = len(s1), len(s2)
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        
        return dtw[n, m]
    
    def _statistical_similarity(self, sig1, sig2):
        """Statistical similarity based on distribution matching"""
        scores = []
        
        for key in sig1:
            if key.endswith('_stats') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                # Euclidean distance in stats space
                dist = np.linalg.norm(s1 - s2)
                similarity = 1.0 / (1.0 + dist)
                scores.append(similarity)
        
        return np.mean(scores) if scores else 0.0
    
    def _neural_similarity(self, emb1, emb2, device, net):
        """Neural network-based similarity"""
        with torch.no_grad():
            t1 = torch.tensor(emb1, device=device, dtype=torch.float32).unsqueeze(0)
            t2 = torch.tensor(emb2, device=device, dtype=torch.float32).unsqueeze(0)
            
            similarity = net(t1, t2).item()
        
        return similarity
    
    def _temporal_pattern_score(self, sig1, sig2):
        """Score based on temporal pattern matching"""
        scores = []
        
        for key in sig1:
            if key.endswith('_pattern') and key in sig2:
                p1, p2 = sig1[key], sig2[key]
                
                # Correlation of pattern vectors
                if np.std(p1) > 0 and np.std(p2) > 0:
                    corr = np.corrcoef(p1, p2)[0, 1]
                    scores.append(abs(corr))
        
        return np.mean(scores) if scores else 0.0
    
    def _fine_align_gpu(self, video_features, gpx_data, device):
        """GPU-accelerated fine alignment"""
        # Use primary motion signals
        video_signal = video_features.get('motion_magnitude', np.array([]))
        gpx_signal = gpx_data['features'].get('speed', np.array([]))
        
        if len(video_signal) == 0 or len(gpx_signal) == 0:
            return 0, 0.0
        
        # Normalize signals
        v_norm = (video_signal - np.mean(video_signal)) / (np.std(video_signal) + 1e-8)
        g_norm = (gpx_signal - np.mean(gpx_signal)) / (np.std(gpx_signal) + 1e-8)
        
        if device.type == 'cuda':
            # GPU cross-correlation
            tv = torch.tensor(v_norm, device=device, dtype=torch.float32)
            tg = torch.tensor(g_norm, device=device, dtype=torch.float32)
            
            # Use conv1d for cross-correlation
            correlation = F.conv1d(
                tv.unsqueeze(0).unsqueeze(0),
                tg.flip(0).unsqueeze(0).unsqueeze(0),
                padding=len(tg)-1
            ).squeeze()
            
            # Find peak
            max_idx = torch.argmax(correlation)
            offset = max_idx.item() - (len(tg) - 1)
            
            # Normalized score
            max_corr = correlation[max_idx].item()
            norm_factor = torch.sqrt(torch.sum(tv**2) * torch.sum(tg**2)).item()
            score = max_corr / norm_factor if norm_factor > 0 else 0.0
        else:
            # CPU fallback
            correlation = correlate(v_norm, g_norm, mode='full')
            max_idx = np.argmax(correlation)
            offset = max_idx - (len(g_norm) - 1)
            
            max_corr = correlation[max_idx]
            norm_factor = np.sqrt(np.sum(v_norm**2) * np.sum(g_norm**2))
            score = max_corr / norm_factor if norm_factor > 0 else 0.0
        
        return offset, score
    
    def _generate_accuracy_report(self, results, output_dir):
        """Generate comprehensive accuracy report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(results),
            'successful_correlations': 0,
            'failed_correlations': 0,
            'score_distribution': [],
            'confidence_levels': {
                'high': 0,      # score > 0.3
                'medium': 0,    # 0.15 < score <= 0.3
                'low': 0,       # 0.05 < score <= 0.15
                'very_low': 0   # score <= 0.05
            },
            'detailed_results': []
        }
        
        for video_path, result in results.items():
            if result is None or not result.get('matches'):
                report['failed_correlations'] += 1
                logger.warning(f"No matches for {video_path}")
                continue
            
            report['successful_correlations'] += 1
            
            best_match = result['matches'][0] if result['matches'] else None
            if best_match:
                score = best_match['final_score']
                report['score_distribution'].append(score)
                
                # Adjusted confidence thresholds based on diagnostic results
                if score > 0.3:
                    report['confidence_levels']['high'] += 1
                elif score > 0.15:
                    report['confidence_levels']['medium'] += 1
                elif score > 0.05:
                    report['confidence_levels']['low'] += 1
                else:
                    report['confidence_levels']['very_low'] += 1
                
                # Detailed result
                report['detailed_results'].append({
                    'video': str(video_path),
                    'best_match': str(best_match['path']),
                    'score': score,
                    'sub_scores': best_match.get('scores', {}),
                    'duration_ratio': best_match['duration_ratio'],
                    'time_offset': best_match.get('time_offset', 0),
                    'all_matches': [
                        {
                            'gpx': str(m['path']),
                            'score': m['final_score']
                        } for m in result['matches'][:5]
                    ]
                })
        
        # Calculate statistics
        if report['score_distribution']:
            scores = np.array(report['score_distribution'])
            report['statistics'] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'median_score': float(np.median(scores))
            }
        
        # Save report
        report_path = output_dir / 'accuracy_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary
        summary_path = output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Video-GPX Correlation Accuracy Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Total Videos: {report['total_videos']}\n")
            f.write(f"Successful: {report['successful_correlations']}\n")
            f.write(f"Failed: {report['failed_correlations']}\n\n")
            
            if 'statistics' in report:
                f.write("Score Statistics:\n")
                f.write(f"  Mean: {report['statistics']['mean_score']:.4f}\n")
                f.write(f"  Std: {report['statistics']['std_score']:.4f}\n")
                f.write(f"  Min: {report['statistics']['min_score']:.4f}\n")
                f.write(f"  Max: {report['statistics']['max_score']:.4f}\n")
                f.write(f"  Median: {report['statistics']['median_score']:.4f}\n\n")
            
            f.write("Confidence Distribution:\n")
            for level, count in report['confidence_levels'].items():
                percentage = (count / report['total_videos'] * 100) if report['total_videos'] > 0 else 0
                f.write(f"  {level}: {count} ({percentage:.1f}%)\n")
        
        logger.info(f"Accuracy report saved to {report_path}")
        logger.info(f"Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Video-GPX Correlation System for Accuracy Testing")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing videos and GPX files")
    parser.add_argument("-o", "--output", default="./correlation_results", help="Output directory for results")
    parser.add_argument("-c", "--cache", default="./cache", help="Cache directory")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top K matches per video")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], help="GPU IDs to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for GPU processing")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads")
    parser.add_argument("--force", action='store_true', help="Force reprocessing")
    
    args = parser.parse_args()
    
    # Setup directories
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)
    
    # Find all videos and GPX files (case-insensitive)
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext.lower()}')))
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext.upper()}')))
    
    # Remove duplicates while preserving order
    video_files = list(dict.fromkeys(video_files))
    
    # Find GPX files (case-insensitive)
    gpx_files = []
    gpx_files.extend(glob.glob(os.path.join(args.directory, '*.gpx')))
    gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
    gpx_files.extend(glob.glob(os.path.join(args.directory, '*.Gpx')))
    
    # Remove duplicates
    gpx_files = list(dict.fromkeys(gpx_files))
    
    logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
    
    if not video_files or not gpx_files:
        logger.error("No videos or GPX files found!")
        return
    
    # Initialize components
    video_analyzer = DualGPUVideoAnalyzer(gpu_ids=args.gpu_ids, batch_size=args.batch_size)
    gpx_processor = AdvancedGPXProcessor()
    correlator = EnhancedGPUCorrelator(gpu_ids=args.gpu_ids)
    
    # Process videos
    logger.info("Processing videos...")
    video_cache_path = cache_dir / "video_features.pkl"
    
    if video_cache_path.exists() and not args.force:
        with open(video_cache_path, 'rb') as f:
            video_features = pickle.load(f)
        logger.info(f"Loaded cached video features for {len(video_features)} videos")
    else:
        video_features = video_analyzer.extract_motion_features_batch(
            video_files, max_workers=args.max_workers or len(args.gpu_ids)
        )
        
        with open(video_cache_path, 'wb') as f:
            pickle.dump(video_features, f)
        logger.info(f"Processed and cached {len(video_features)} videos")
    
    # Process GPX files
    logger.info("Processing GPX files...")
    gpx_cache_path = cache_dir / "gpx_features.pkl"
    
    if gpx_cache_path.exists() and not args.force:
        with open(gpx_cache_path, 'rb') as f:
            gpx_database = pickle.load(f)
        logger.info(f"Loaded cached GPX features for {len(gpx_database)} files")
    else:
        gpx_database = gpx_processor.parse_gpx_batch(
            gpx_files, max_workers=args.max_workers
        )
        
        with open(gpx_cache_path, 'wb') as f:
            pickle.dump(gpx_database, f)
        logger.info(f"Processed and cached {len(gpx_database)} GPX files")
    
    # Perform correlation analysis
    logger.info("Performing correlation analysis...")
    results = correlator.correlate_all(
        video_features, gpx_database, output_dir, top_k=args.top_k
    )
    
    # Save detailed results
    results_path = output_dir / "all_correlations.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    # Print summary
    with open(output_dir / 'summary.txt', 'r') as f:
        print("\n" + f.read())

if __name__ == "__main__":
    main()