#!/usr/bin/env python3
"""
NVIDIA Hardware-Accelerated Video-GPX Matcher
Uses NVDEC for video decoding and keeps all data on GPU
"""

import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gpxpy
import pandas as pd
from datetime import datetime
import argparse
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle
import json
import warnings
from tqdm import tqdm
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional, Any
import traceback

# For hardware video decoding
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    from PyNvVideoCodec import *
    NVDEC_AVAILABLE = True
except ImportError:
    NVDEC_AVAILABLE = False
    print("Warning: PyNvCodec not available, falling back to CPU decoding")

# Try VPF (Video Processing Framework) as alternative
try:
    import PyNvVideoCodec as nvc
    VPF_AVAILABLE = True
except ImportError:
    VPF_AVAILABLE = False

# Fallback to decord for GPU decoding
try:
    from decord import VideoReader, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

import cv2  # Fallback option

warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU memory pools
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class GPUVideoDecoder:
    """Hardware-accelerated video decoder using available GPU backends"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.backend = None
        
        # Try different GPU decoding backends
        if NVDEC_AVAILABLE:
            self.backend = 'nvdec'
            logger.info("Using NVDEC hardware decoder")
        elif VPF_AVAILABLE:
            self.backend = 'vpf'
            logger.info("Using VPF hardware decoder")
        elif DECORD_AVAILABLE:
            self.backend = 'decord'
            logger.info("Using Decord GPU decoder")
        else:
            logger.error("No GPU decoder available. Install Decord or VPF with GPU support.")
            sys.exit(1)    
    def decode_video_gpu(self, video_path, sample_interval=15):
        """Decode video directly to GPU memory"""
        if self.backend == 'decord':
            return self._decode_with_decord(video_path, sample_interval)
        elif self.backend == 'vpf':
            return self._decode_with_vpf(video_path, sample_interval)
        elif self.backend == 'nvdec':
            return self._decode_with_nvdec(video_path, sample_interval)
        else:
            return self._decode_with_opencv(video_path, sample_interval)
    
    def _decode_with_decord(self, video_path, sample_interval):
        """Use Decord for GPU decoding"""
        try:
            # Set GPU context
            ctx = gpu(self.gpu_id)
            
            # Open video with GPU decoder
            vr = VideoReader(video_path, ctx=ctx)
            
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            
            # Sample frames
            frame_indices = list(range(0, total_frames, sample_interval))
            
            # Batch decode frames directly to GPU
            frames_gpu = []
            batch_size = 32
            
            for i in range(0, len(frame_indices), batch_size):
                batch_indices = frame_indices[i:i+batch_size]
                batch = vr.get_batch(batch_indices)  # Returns GPU tensor
                
                # Convert to CuPy array
                frames_np = batch.asnumpy()  # Unfortunately need CPU transfer
                frames_cp = cp.asarray(frames_np)
                frames_gpu.append(frames_cp)
            
            if frames_gpu:
                frames_gpu = cp.concatenate(frames_gpu, axis=0)
            
            return {
                'frames': frames_gpu,
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps
            }
            
        except Exception as e:
            logger.error(f"Decord decoding failed: {e}")
            return self._decode_with_opencv(video_path, sample_interval)
    
    def _decode_with_opencv(self, video_path, sample_interval):
        """Fallback CPU decoding with immediate GPU upload"""
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize and upload to GPU immediately
                frame = cv2.resize(frame, (640, 360))
                frames.append(frame)
        
        cap.release()
        
        # Batch upload to GPU
        if frames:
            frames_np = np.array(frames)
            frames_gpu = cp.asarray(frames_np)
        else:
            frames_gpu = cp.array([])
        
        return {
            'frames': frames_gpu,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }


class GPUFeatureExtractor:
    """Pure GPU feature extraction using CuPy kernels"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        
        # Define custom CUDA kernels for motion analysis
        self.frame_diff_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void frame_diff(const unsigned char* frame1, const unsigned char* frame2, 
                       float* diff, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                int idx = y * width + x;
                float d = abs((float)frame1[idx] - (float)frame2[idx]);
                diff[idx] = d;
            }
        }
        ''', 'frame_diff')
        
        self.motion_stats_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void motion_stats(const float* diff, float* stats, int size) {
            extern __shared__ float sdata[];
            
            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Load and compute local sum
            sdata[tid] = (i < size) ? diff[i] : 0;
            __syncthreads();
            
            // Reduction
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                atomicAdd(&stats[0], sdata[0]);  // sum
                atomicAdd(&stats[1], 1.0f);      // count
            }
        }
        ''', 'motion_stats')
        
        # Initialize PyTorch model for deep features
        self._init_torch_model()
    
    def _init_torch_model(self):
        """Initialize efficient CNN for feature extraction"""
        self.device = torch.device(f'cuda:{self.gpu_id}')
        
        # MobileNetV3 for efficiency
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
        self.model.classifier = nn.Identity()  # Remove classifier
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Compile with TorchScript for speed
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.model = torch.jit.trace(self.model, dummy_input)
    
    def extract_features_batch_gpu(self, frames_gpu):
        """Extract features keeping everything on GPU"""
        if len(frames_gpu) == 0:
            return None
        
        num_frames = len(frames_gpu)
        
        # Convert to grayscale on GPU
        # Use matrix multiplication for RGB to gray conversion
        rgb_weights = cp.array([0.299, 0.587, 0.114], dtype=cp.float32)
        
        # Reshape for batch processing
        if frames_gpu.ndim == 4:  # (N, H, W, C)
            gray_frames = cp.dot(frames_gpu.astype(cp.float32), rgb_weights)
        else:
            gray_frames = frames_gpu
        
        # Motion features using custom kernels
        motion_magnitudes = cp.zeros(num_frames - 1, dtype=cp.float32)
        motion_complexity = cp.zeros(num_frames - 1, dtype=cp.float32)
        
        # Compute frame differences
        for i in range(num_frames - 1):
            diff = cp.abs(gray_frames[i+1] - gray_frames[i])
            motion_magnitudes[i] = cp.mean(diff)
            motion_complexity[i] = cp.std(diff)
        
        # Deep features extraction
        deep_features = self._extract_deep_features_gpu(frames_gpu)
        
        # Optical flow approximation using phase correlation
        flow_magnitudes = self._compute_flow_gpu(gray_frames)
        
        # Keep everything on GPU
        features = {
            'motion_magnitude': motion_magnitudes,
            'motion_complexity': motion_complexity,
            'flow_magnitude': flow_magnitudes,
            'deep_features': deep_features,
            'num_frames': num_frames
        }
        
        return features
    
    def _extract_deep_features_gpu(self, frames_gpu):
        """Extract deep features using PyTorch, keeping on GPU"""
        batch_size = 32
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(frames_gpu), batch_size):
                batch = frames_gpu[i:i+batch_size]
                
                # Convert CuPy to PyTorch without CPU transfer
                # Use __cuda_array_interface__ for zero-copy
                batch_torch = torch.as_tensor(batch, device=self.device)
                
                # Normalize and reshape
                if batch_torch.dim() == 4:  # (N, H, W, C)
                    batch_torch = batch_torch.permute(0, 3, 1, 2).float() / 255.0
                
                # Resize if needed
                if batch_torch.shape[-1] != 224:
                    batch_torch = F.interpolate(batch_torch, size=(224, 224), mode='bilinear')
                
                # Extract features
                features = self.model(batch_torch)
                
                # Convert back to CuPy
                features_cp = cp.asarray(features.detach())
                all_features.append(features_cp)
        
        if all_features:
            return cp.concatenate(all_features, axis=0)
        return cp.array([])
    
    def _compute_flow_gpu(self, gray_frames):
        """Fast optical flow approximation on GPU"""
        if len(gray_frames) < 2:
            return cp.array([])
        
        flow_mags = cp.zeros(len(gray_frames) - 1, dtype=cp.float32)
        
        for i in range(len(gray_frames) - 1):
            # Simple but fast flow approximation
            frame1 = gray_frames[i].astype(cp.float32)
            frame2 = gray_frames[i + 1].astype(cp.float32)
            
            # Compute gradients
            dx = cp.gradient(frame2, axis=1) - cp.gradient(frame1, axis=1)
            dy = cp.gradient(frame2, axis=0) - cp.gradient(frame1, axis=0)
            
            # Flow magnitude
            mag = cp.sqrt(dx**2 + dy**2)
            flow_mags[i] = cp.mean(mag)
        
        return flow_mags


class GPUCorrelator:
    """GPU-accelerated correlation using batch operations"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
    
    def correlate_batch_gpu(self, video_features, gpx_features_list, top_k=5):
        """Correlate video with all GPX files in batch on GPU"""
        if not video_features or not gpx_features_list:
            return []
        
        # Create signatures
        video_sig = self._create_signature_gpu(video_features)
        
        # Batch process all GPX signatures
        num_gpx = len(gpx_features_list)
        scores = cp.zeros(num_gpx, dtype=cp.float32)
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        
        for chunk_start in range(0, num_gpx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_gpx)
            chunk_features = gpx_features_list[chunk_start:chunk_end]
            
            # Compute similarities for chunk
            chunk_scores = self._compute_chunk_similarities_gpu(video_sig, chunk_features)
            scores[chunk_start:chunk_end] = chunk_scores
        
        # Find top-k matches
        if len(scores) > 0:
            # Use argpartition for efficiency
            k = min(top_k, len(scores))
            top_indices = cp.argpartition(-scores, k-1)[:k]
            top_indices = top_indices[cp.argsort(-scores[top_indices])]
            
            # Get scores
            top_scores = scores[top_indices]
            
            return cp.asnumpy(top_indices), cp.asnumpy(top_scores)
        
        return np.array([]), np.array([])
    
    def _create_signature_gpu(self, features):
        """Create normalized signature on GPU"""
        signature = {}
        
        for key, values in features.items():
            if isinstance(values, cp.ndarray) and len(values) > 0:
                # Normalize
                mean = cp.mean(values)
                std = cp.std(values) + 1e-8
                normalized = (values - mean) / std
                
                # FFT
                fft = cp.fft.fft(normalized)
                fft_mag = cp.abs(fft[:len(fft)//2])
                
                signature[f'{key}_fft'] = fft_mag
                signature[f'{key}_stats'] = cp.array([mean, std, cp.min(values), cp.max(values)])
                
                # Downsample for correlation
                if len(normalized) > 100:
                    indices = cp.linspace(0, len(normalized)-1, 100).astype(cp.int32)
                    signature[f'{key}_signal'] = normalized[indices]
                else:
                    signature[f'{key}_signal'] = normalized
        
        return signature
    
    def _compute_chunk_similarities_gpu(self, video_sig, gpx_features_chunk):
        """Compute similarities for a chunk of GPX files"""
        chunk_size = len(gpx_features_chunk)
        chunk_scores = cp.zeros(chunk_size, dtype=cp.float32)
        
        for i, gpx_features in enumerate(gpx_features_chunk):
            gpx_sig = self._create_signature_gpu(gpx_features)
            
            # Multi-modal similarity
            fft_score = self._fft_similarity_gpu(video_sig, gpx_sig)
            signal_score = self._signal_similarity_gpu(video_sig, gpx_sig)
            stats_score = self._stats_similarity_gpu(video_sig, gpx_sig)
            
            # Weighted combination
            chunk_scores[i] = 0.4 * fft_score + 0.4 * signal_score + 0.2 * stats_score
        
        return chunk_scores
    
    def _fft_similarity_gpu(self, sig1, sig2):
        """FFT-based similarity"""
        scores = []
        
        for key in sig1:
            if key.endswith('_fft') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                # Match lengths
                min_len = min(len(s1), len(s2))
                if min_len > 0:
                    s1 = s1[:min_len]
                    s2 = s2[:min_len]
                    
                    # Normalized correlation
                    corr = cp.sum(s1 * s2) / (cp.linalg.norm(s1) * cp.linalg.norm(s2) + 1e-8)
                    scores.append(float(cp.abs(corr)))
        
        return np.mean(scores) if scores else 0.0
    
    def _signal_similarity_gpu(self, sig1, sig2):
        """Direct signal correlation"""
        scores = []
        
        for key in sig1:
            if key.endswith('_signal') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                if len(s1) > 0 and len(s2) > 0:
                    # Cross-correlation
                    corr = cp.correlate(s1, s2, mode='valid')
                    if len(corr) > 0:
                        max_corr = cp.max(cp.abs(corr))
                        norm = cp.sqrt(cp.sum(s1**2) * cp.sum(s2**2)) + 1e-8
                        scores.append(float(max_corr / norm))
        
        return np.mean(scores) if scores else 0.0
    
    def _stats_similarity_gpu(self, sig1, sig2):
        """Statistical similarity"""
        scores = []
        
        for key in sig1:
            if key.endswith('_stats') and key in sig2:
                s1, s2 = sig1[key], sig2[key]
                
                if len(s1) == len(s2):
                    # Cosine similarity
                    dot = cp.sum(s1 * s2)
                    norm = cp.linalg.norm(s1) * cp.linalg.norm(s2) + 1e-8
                    scores.append(float(cp.abs(dot / norm)))
        
        return np.mean(scores) if scores else 0.0


def process_videos_gpu(video_files, gpu_id=0):
    """Process videos with full GPU acceleration"""
    decoder = GPUVideoDecoder(gpu_id)
    extractor = GPUFeatureExtractor(gpu_id)
    
    results = {}
    
    for video_path in tqdm(video_files, desc=f"GPU {gpu_id} videos"):
        try:
            # Decode video to GPU
            video_data = decoder.decode_video_gpu(video_path, sample_interval=15)
            
            if video_data['frames'] is not None and len(video_data['frames']) > 0:
                # Extract features on GPU
                features = extractor.extract_features_batch_gpu(video_data['frames'])
                
                if features:
                    # Add metadata
                    features['duration'] = video_data['duration']
                    features['fps'] = video_data['fps']
                    
                    # Convert to CPU only what's needed
                    results[video_path] = {
                        'duration': video_data['duration'],
                        'fps': video_data['fps'],
                        'features_gpu': features  # Keep on GPU
                    }
            
            # Clear GPU memory periodically
            if len(results) % 5 == 0:
                mempool.free_all_blocks()
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            results[video_path] = None
    
    return results


def process_gpx_files_gpu(gpx_files, gpu_id=0):
    """Process GPX files with GPU acceleration"""
    results = {}
    
    with cp.cuda.Device(gpu_id):
        for gpx_path in tqdm(gpx_files, desc="GPX files"):
            try:
                # Parse GPX
                with open(gpx_path, 'r', encoding='utf-8') as f:
                    gpx = gpxpy.parse(f)
                
                points = []
                for track in gpx.tracks:
                    for segment in track.segments:
                        for pt in segment.points:
                            if pt.time:
                                points.append([
                                    pt.time.timestamp(),
                                    pt.latitude,
                                    pt.longitude,
                                    pt.elevation or 0
                                ])
                
                if len(points) < 10:
                    continue
                
                # Process on GPU
                data = cp.asarray(points, dtype=cp.float32)
                
                # Sort by time
                data = data[cp.argsort(data[:, 0])]
                
                # Extract features
                times = data[:, 0]
                lats = data[:, 1]
                lons = data[:, 2]
                
                # Vectorized calculations
                dt = cp.diff(times)
                dt = cp.maximum(dt, 0.1)
                
                # Haversine distance
                lat1, lat2 = lats[:-1], lats[1:]
                lon1, lon2 = lons[:-1], lons[1:]
                
                dlat = cp.radians(lat2 - lat1)
                dlon = cp.radians(lon2 - lon1)
                
                a = cp.sin(dlat/2)**2 + cp.cos(cp.radians(lat1)) * cp.cos(cp.radians(lat2)) * cp.sin(dlon/2)**2
                c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))
                distances = 3958.8 * c
                
                # Speed
                speeds = (distances * 3600) / dt
                speeds = cp.clip(speeds, 0, 200)
                
                # Pad arrays
                speeds_padded = cp.zeros(len(points), dtype=cp.float32)
                speeds_padded[1:] = speeds
                
                # Keep on GPU
                results[gpx_path] = {
                    'duration': float(times[-1] - times[0]),
                    'distance': float(cp.sum(distances)),
                    'features_gpu': {
                        'speed': speeds_padded,
                        'distances': distances
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to process {gpx_path}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Video-GPX Matcher")
    parser.add_argument("-d", "--directory", required=True, help="Directory with videos and GPX files")
    parser.add_argument("-o", "--output", default="./gpu_results", help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--top-k", type=int, default=5, help="Top K matches")
    
    args = parser.parse_args()
    
    # Find files
    video_files = []
    for ext in ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']:
        video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
    
    gpx_files = []
    for ext in ['gpx', 'GPX']:
        gpx_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
    
    video_files = list(set(video_files))
    gpx_files = list(set(gpx_files))
    
    logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
    
    if not video_files or not gpx_files:
        logger.error("No files found!")
        return
    
    # Process videos on GPU
    logger.info("Processing videos on GPU...")
    video_features = process_videos_gpu(video_files, args.gpu)
    
    # Process GPX files on GPU
    logger.info("Processing GPX files on GPU...")
    gpx_features = process_gpx_files_gpu(gpx_files, args.gpu)
    
    # Perform correlation on GPU
    logger.info("Correlating on GPU...")
    correlator = GPUCorrelator(args.gpu)
    
    results = {}
    for video_path, video_data in tqdm(video_features.items(), desc="Correlating"):
        if video_data is None:
            continue
        
        # Extract GPU features list for GPX files
        gpx_features_list = [gdata['features_gpu'] for gdata in gpx_features.values() if gdata]
        gpx_paths = [gpath for gpath, gdata in gpx_features.items() if gdata]
        
        # Correlate on GPU
        top_indices, top_scores = correlator.correlate_batch_gpu(
            video_data['features_gpu'],
            gpx_features_list,
            args.top_k
        )
        
        # Build results
        matches = []
        for idx, score in zip(top_indices, top_scores):
            if score > 0.01:
                matches.append({
                    'gpx': gpx_paths[idx],
                    'score': float(score)
                })
        
        results[video_path] = {
            'matches': matches,
            'duration': video_data['duration']
        }
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in results.values() if r.get('matches'))
    logger.info(f"\nResults: {successful}/{len(results)} videos matched")
    
    # Clean up
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()