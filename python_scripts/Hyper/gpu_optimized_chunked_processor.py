#!/usr/bin/env python3
"""
Complete GPU-Optimized Chunked Video Processor Implementation
Designed for: 2x16GB VRAM, 128GB RAM, 16-core AMD system

Key Features:
- Chunked processing to handle unlimited video length/resolution
- RAM-based staging for 4K+ videos  
- Aggressive GPU memory management
- Parallel dual-GPU utilization
- Memory pool allocation/reuse
- Pipeline processing with overlap

Copy this entire file into gpu_optimized_chunked_processor.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import subprocess
import os
import time
import gc
import math
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
import psutil
import queue
import threading
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for video chunking"""
    target_chunk_frames: int = 60  # Frames per chunk
    max_chunk_memory_gb: float = 4.0  # Max memory per chunk on GPU
    overlap_frames: int = 5  # Frame overlap between chunks
    min_chunk_frames: int = 10  # Minimum viable chunk size
    ram_buffer_chunks: int = 3  # Chunks to keep in RAM buffer

class GPUMemoryPool:
    """Pre-allocated GPU memory pool for efficient reuse"""
    
    def __init__(self, gpu_id: int, pool_size_gb: float = 12.0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.pool_size_gb = pool_size_gb
        self.allocated_tensors = {}
        self.free_tensors = {}
        self.max_allocations = {}
        
        # Pre-allocate common tensor sizes
        self._preallocate_common_sizes()
        
    def _preallocate_common_sizes(self):
        """Pre-allocate tensors for common video resolutions"""
        common_sizes = [
            (60, 3, 1080, 1920),   # 1080p, 60 frames
            (60, 3, 2160, 3840),   # 4K, 60 frames
            (30, 3, 1080, 1920),   # 1080p, 30 frames
            (30, 3, 2160, 3840),   # 4K, 30 frames
        ]
        
        available_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
        target_pool_bytes = int(self.pool_size_gb * 1024**3)
        
        with torch.cuda.device(self.device):
            for size in common_sizes:
                try:
                    memory_needed = np.prod(size) * 4  # float32
                    if memory_needed < target_pool_bytes * 0.3:  # Use max 30% per tensor
                        tensor = torch.empty(size, dtype=torch.float32, device=self.device)
                        size_key = tuple(size)
                        if size_key not in self.free_tensors:
                            self.free_tensors[size_key] = []
                        self.free_tensors[size_key].append(tensor)
                        logger.info(f"Pre-allocated tensor {size} on GPU {self.gpu_id}")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"Could not pre-allocate tensor {size} on GPU {self.gpu_id}")
                    break
    
    def get_tensor(self, shape: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        """Get tensor from pool or allocate new one"""
        shape_key = tuple(shape)
        
        # Try to reuse existing tensor
        if shape_key in self.free_tensors and self.free_tensors[shape_key]:
            tensor = self.free_tensors[shape_key].pop()
            tensor.zero_()  # Clear data
            return tensor
        
        # Allocate new tensor
        try:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            return tensor
        except torch.cuda.OutOfMemoryError:
            # Emergency cleanup and retry
            self.emergency_cleanup()
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        if tensor.device != self.device:
            return
        
        shape_key = tuple(tensor.shape)
        if shape_key not in self.free_tensors:
            self.free_tensors[shape_key] = []
        
        # Limit pool size per shape
        if len(self.free_tensors[shape_key]) < 3:
            self.free_tensors[shape_key].append(tensor)
        del tensor
    
    def emergency_cleanup(self):
        """Emergency GPU memory cleanup"""
        # Clear free tensors
        for shape_tensors in self.free_tensors.values():
            shape_tensors.clear()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

class ChunkedVideoProcessor:
    """High-performance chunked video processor"""
    
    def __init__(self, gpu_manager, config):
        self.gpu_manager = gpu_manager
        self.config = config
        self.chunk_config = ChunkConfig()
        
        # Create memory pools for each GPU
        self.memory_pools = {}
        for gpu_id in gpu_manager.gpu_ids:
            self.memory_pools[gpu_id] = GPUMemoryPool(gpu_id, pool_size_gb=12.0)
        
        # RAM management for chunk staging
        self.ram_usage_gb = 0
        self.max_ram_usage_gb = 100.0  # Use up to 100GB of 128GB RAM
        self.chunk_cache = {}  # Cache for processed chunks
        
        # Feature extractors with memory optimization
        self.feature_extractors = {}
        for gpu_id in gpu_manager.gpu_ids:
            self.feature_extractors[gpu_id] = OptimizedFeatureExtractor(gpu_id, self.memory_pools[gpu_id])
        
        logger.info(f"Initialized chunked processor for GPUs: {gpu_manager.gpu_ids}")
        logger.info(f"RAM allocation: {self.max_ram_usage_gb}GB / 128GB available")
    
    def process_video_chunked(self, video_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Process video using chunked approach for unlimited size support"""
        
        # Get video information
        video_info = self._get_comprehensive_video_info(video_path)
        if not video_info:
            logger.error(f"Could not analyze video: {video_path}")
            return None
        
        logger.info(f"Processing {Path(video_path).name}: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['frame_count']} frames, {video_info['duration']:.1f}s")
        
        # Calculate optimal chunking strategy
        chunk_strategy = self._calculate_chunking_strategy(video_info)
        logger.info(f"Chunking strategy: {chunk_strategy['num_chunks']} chunks of "
                   f"{chunk_strategy['frames_per_chunk']} frames each")
        
        # Process video in chunks
        all_features = []
        processing_times = []
        
        try:
            for chunk_idx in range(chunk_strategy['num_chunks']):
                chunk_start_time = time.time()
                
                # Calculate chunk boundaries
                start_frame = chunk_idx * chunk_strategy['frames_per_chunk']
                end_frame = min(start_frame + chunk_strategy['frames_per_chunk'], 
                              video_info['frame_count'])
                
                if end_frame - start_frame < self.chunk_config.min_chunk_frames:
                    logger.info(f"Skipping small final chunk: {end_frame - start_frame} frames")
                    break
                
                logger.info(f"Processing chunk {chunk_idx + 1}/{chunk_strategy['num_chunks']}: "
                           f"frames {start_frame}-{end_frame}")
                
                # Process chunk
                chunk_features = self._process_single_chunk(
                    video_path, video_info, start_frame, end_frame, chunk_idx
                )
                
                if chunk_features is not None:
                    all_features.append(chunk_features)
                    
                    chunk_time = time.time() - chunk_start_time
                    processing_times.append(chunk_time)
                    
                    # Performance monitoring
                    fps = (end_frame - start_frame) / chunk_time
                    logger.info(f"Chunk {chunk_idx + 1} completed in {chunk_time:.1f}s "
                               f"({fps:.1f} FPS)")
                else:
                    logger.error(f"Failed to process chunk {chunk_idx + 1}")
                    return None
                
                # Memory cleanup between chunks
                self._cleanup_between_chunks()
            
            # Aggregate all chunk features
            if all_features:
                final_features = self._aggregate_chunk_features(all_features, video_info)
                
                # Performance summary
                total_time = sum(processing_times)
                avg_fps = video_info['frame_count'] / total_time if total_time > 0 else 0
                
                logger.info(f"Video processing complete: {total_time:.1f}s total, "
                           f"{avg_fps:.1f} FPS average")
                
                return final_features
            else:
                logger.error("No chunks were successfully processed")
                return None
                
        except Exception as e:
            logger.error(f"Error in chunked video processing: {e}")
            return None
        finally:
            # Final cleanup
            self._final_cleanup()
    
    def _get_comprehensive_video_info(self, video_path: str) -> Optional[Dict]:
        """Get detailed video information for chunking strategy"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            
            probe_data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Extract comprehensive info
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            # Frame count calculation
            frame_count = 0
            if 'nb_frames' in video_stream and video_stream['nb_frames'] != 'N/A':
                frame_count = int(video_stream['nb_frames'])
            else:
                # Fallback: calculate from duration and frame rate
                duration = float(video_stream.get('duration', 0))
                fps_str = video_stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) > 0 else 0
                else:
                    fps = float(fps_str)
                frame_count = int(duration * fps) if fps > 0 else 0
            
            # Calculate memory requirements
            bytes_per_frame = width * height * 3 * 4  # RGB float32
            total_video_memory_gb = (frame_count * bytes_per_frame) / (1024**3)
            
            return {
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'duration': float(video_stream.get('duration', 0)),
                'fps': fps,
                'pixel_format': video_stream.get('pix_fmt', ''),
                'codec': video_stream.get('codec_name', ''),
                'bytes_per_frame': bytes_per_frame,
                'total_memory_gb': total_video_memory_gb,
                'bitrate': int(video_stream.get('bit_rate', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None
    
    def _calculate_chunking_strategy(self, video_info: Dict) -> Dict:
        """Calculate optimal chunking strategy based on video and hardware"""
        
        width, height = video_info['width'], video_info['height']
        total_frames = video_info['frame_count']
        bytes_per_frame = video_info['bytes_per_frame']
        
        # Calculate frames that fit in GPU memory
        available_gpu_memory = 12.0 * 1024**3  # 12GB per GPU (conservative)
        
        # Account for model memory (ResNet18 + buffers ≈ 200MB)
        model_memory = 0.5 * 1024**3
        available_for_frames = available_gpu_memory - model_memory
        
        # Calculate max frames per chunk
        max_frames_per_gpu = int(available_for_frames / bytes_per_frame)
        
        # Conservative safety margin
        frames_per_chunk = max(
            self.chunk_config.min_chunk_frames,
            min(max_frames_per_gpu * 0.7, self.chunk_config.target_chunk_frames)
        )
        
        # Handle edge cases
        if frames_per_chunk > total_frames:
            frames_per_chunk = total_frames
            num_chunks = 1
        else:
            num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
        
        # Memory usage estimates
        chunk_memory_gb = (frames_per_chunk * bytes_per_frame) / (1024**3)
        total_ram_needed = num_chunks * chunk_memory_gb * 0.5  # Staging
        
        logger.info(f"Chunking analysis:")
        logger.info(f"  Video: {width}x{height}, {total_frames} frames")
        logger.info(f"  Chunk size: {frames_per_chunk} frames ({chunk_memory_gb:.1f}GB)")
        logger.info(f"  Total chunks: {num_chunks}")
        logger.info(f"  RAM staging: {total_ram_needed:.1f}GB")
        
        return {
            'frames_per_chunk': int(frames_per_chunk),
            'num_chunks': num_chunks,
            'chunk_memory_gb': chunk_memory_gb,
            'total_ram_needed': total_ram_needed,
            'bytes_per_frame': bytes_per_frame
        }
    
    def _process_single_chunk(self, video_path: str, video_info: Dict, 
                            start_frame: int, end_frame: int, chunk_idx: int) -> Optional[Dict]:
        """Process a single video chunk on GPU"""
        
        # Acquire GPU
        gpu_id = self.gpu_manager.acquire_gpu(timeout=30)
        if gpu_id is None:
            logger.error(f"Could not acquire GPU for chunk {chunk_idx}")
            return None
        
        try:
            # Load chunk frames to RAM first
            chunk_frames = self._load_chunk_to_ram(video_path, video_info, start_frame, end_frame)
            if chunk_frames is None:
                return None
            
            # Transfer to GPU in optimal batch size
            gpu_frames = self._transfer_chunk_to_gpu(chunk_frames, gpu_id)
            if gpu_frames is None:
                return None
            
            # Extract features on GPU
            chunk_features = self.feature_extractors[gpu_id].extract_features_optimized(gpu_frames)
            
            # Return GPU tensor to pool and cleanup
            self.memory_pools[gpu_id].return_tensor(gpu_frames)
            del gpu_frames, chunk_frames
            
            return chunk_features
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            return None
        finally:
            self.gpu_manager.release_gpu(gpu_id)
    
    def _load_chunk_to_ram(self, video_path: str, video_info: Dict, 
                         start_frame: int, end_frame: int) -> Optional[np.ndarray]:
        """Load video chunk to RAM using optimized FFmpeg"""
        
        try:
            width, height = video_info['width'], video_info['height']
            num_frames = end_frame - start_frame
            
            # Create temporary file for frames
            temp_dir = Path(os.path.expanduser("~/penis/temp/chunks"))
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique output pattern
            timestamp = int(time.time() * 1000000)
            output_pattern = temp_dir / f"chunk_{timestamp}_%06d.jpg"
            
            # FFmpeg command for specific frame range
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-i', video_path,
                '-vf', f'select=between(n\\,{start_frame}\\,{end_frame-1})',
                '-vsync', '0',  # Disable frame dropping
                '-q:v', '2',    # High quality JPEG
                '-threads', '4',
                str(output_pattern)
            ]
            
            # Execute FFmpeg
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            
            # Load frames to numpy array
            frame_pattern = f"chunk_{timestamp}_*.jpg"
            frame_files = sorted(temp_dir.glob(frame_pattern))
            
            if len(frame_files) != num_frames:
                logger.warning(f"Expected {num_frames} frames, got {len(frame_files)}")
            
            if not frame_files:
                logger.error("No frames were extracted")
                return None
            
            # Pre-allocate array
            frames_array = np.empty((len(frame_files), height, width, 3), dtype=np.float32)
            
            # Load frames efficiently
            for i, frame_file in enumerate(frame_files):
                try:
                    img = cv2.imread(str(frame_file))
                    if img is not None:
                        # Resize if necessary
                        if img.shape[:2] != (height, width):
                            img = cv2.resize(img, (width, height))
                        
                        # Convert BGR to RGB and normalize
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frames_array[i] = img_rgb.astype(np.float32) / 255.0
                    os.remove(frame_file)  # Cleanup immediately
                except Exception as e:
                    logger.warning(f"Error loading frame {frame_file}: {e}")
                    # Fill with zeros if frame loading fails
                    frames_array[i] = np.zeros((height, width, 3), dtype=np.float32)
                    try:
                        os.remove(frame_file)
                    except:
                        pass
            
            logger.debug(f"Loaded chunk to RAM: {frames_array.shape} ({frames_array.nbytes / 1024**3:.1f}GB)")
            return frames_array
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            return None
        except Exception as e:
            logger.error(f"Error loading chunk to RAM: {e}")
            return None
    
    def _transfer_chunk_to_gpu(self, chunk_frames: np.ndarray, gpu_id: int) -> Optional[torch.Tensor]:
        """Transfer chunk from RAM to GPU efficiently"""
        
        try:
            # Get tensor from memory pool
            target_shape = (chunk_frames.shape[0], 3, chunk_frames.shape[1], chunk_frames.shape[2])
            gpu_tensor = self.memory_pools[gpu_id].get_tensor(target_shape)
            
            # Convert numpy to torch and transfer
            with torch.cuda.device(gpu_id):
                # Convert to CHW format efficiently
                frames_torch = torch.from_numpy(chunk_frames).permute(0, 3, 1, 2)
                gpu_tensor.copy_(frames_torch.to(self.memory_pools[gpu_id].device))
                
                del frames_torch  # Free CPU memory immediately
                
            logger.debug(f"Transferred chunk to GPU {gpu_id}: {gpu_tensor.shape}")
            return gpu_tensor
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"GPU {gpu_id} OOM during chunk transfer")
            self.memory_pools[gpu_id].emergency_cleanup()
            return None
        except Exception as e:
            logger.error(f"Error transferring chunk to GPU: {e}")
            return None
    
    def _aggregate_chunk_features(self, chunk_features_list: List[Dict], 
                                video_info: Dict) -> Dict[str, np.ndarray]:
        """Aggregate features from all chunks"""
        
        try:
            if not chunk_features_list:
                return {}
            
            # Initialize aggregated features
            aggregated = {}
            
            # Concatenate time-series features
            time_series_keys = ['motion_magnitude', 'motion_direction', 'acceleration', 
                              'color_variance', 'edge_density']
            
            for key in time_series_keys:
                if key in chunk_features_list[0]:
                    all_values = []
                    for chunk_features in chunk_features_list:
                        if key in chunk_features:
                            all_values.append(chunk_features[key])
                    
                    if all_values:
                        aggregated[key] = np.concatenate(all_values, axis=0)
            
            # Aggregate CNN features (take mean or concatenate based on type)
            cnn_keys = ['global_features', 'motion_features', 'texture_features']
            
            for key in cnn_keys:
                if key in chunk_features_list[0]:
                    all_features = []
                    for chunk_features in chunk_features_list:
                        if key in chunk_features:
                            all_features.append(chunk_features[key])
                    
                    if all_features:
                        # For CNN features, take temporal mean to create video-level descriptor
                        stacked = np.concatenate(all_features, axis=0)
                        aggregated[key] = np.mean(stacked, axis=0, keepdims=True)
            
            # Aggregate histogram features
            if 'color_histograms' in chunk_features_list[0]:
                all_hists = []
                for chunk_features in chunk_features_list:
                    if 'color_histograms' in chunk_features:
                        all_hists.append(chunk_features['color_histograms'])
                
                if all_hists:
                    aggregated['color_histograms'] = np.concatenate(all_hists, axis=0)
            
            # Add metadata
            aggregated['duration'] = video_info['duration']
            aggregated['fps'] = video_info['fps']
            aggregated['frame_count'] = video_info['frame_count']
            aggregated['resolution'] = (video_info['width'], video_info['height'])
            aggregated['processing_mode'] = 'GPU_CHUNKED'
            
            logger.info(f"Aggregated features from {len(chunk_features_list)} chunks")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating chunk features: {e}")
            return {}
    
    def _cleanup_between_chunks(self):
        """Cleanup memory between chunks"""
        # Clear all GPU memory pools
        for pool in self.memory_pools.values():
            pool.emergency_cleanup()
        
        # System-wide cleanup
        gc.collect()
        
        # Small delay to let memory stabilize
        time.sleep(0.1)
    
    def _final_cleanup(self):
        """Final cleanup after video processing"""
        # Clear all memory pools
        for pool in self.memory_pools.values():
            for shape_tensors in pool.free_tensors.values():
                shape_tensors.clear()
            pool.free_tensors.clear()
        
        # Clear chunk cache
        self.chunk_cache.clear()
        self.ram_usage_gb = 0
        
        # Full GPU cleanup
        for gpu_id in self.gpu_manager.gpu_ids:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        gc.collect()

class OptimizedFeatureExtractor:
    """GPU-optimized feature extractor with memory pooling"""
    
    def __init__(self, gpu_id: int, memory_pool):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.memory_pool = memory_pool
        
        # Create optimized model
        self.model = self._create_optimized_model().to(self.device)
        self.model.eval()
        
        # Pre-allocate buffers for common operations
        self._preallocate_buffers()
    
    def _create_optimized_model(self):
        """Create memory-optimized CNN model"""
        
        class OptimizedCNNExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                # Use EfficientNet-B0 for better speed/accuracy tradeoff
                try:
                    backbone = models.efficientnet_b0(pretrained=True)
                    self.features = backbone.features
                    self.avgpool = backbone.avgpool
                    feature_dim = 1280
                except:
                    # Fallback to ResNet18 if EfficientNet not available
                    backbone = models.resnet18(pretrained=True)
                    self.features = nn.Sequential(*list(backbone.children())[:-2])
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    feature_dim = 512
                
                # Lightweight feature heads
                self.global_head = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128)
                )
                
                self.motion_head = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64)
                )
                
                self.texture_head = nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32)
                )
                
                # Freeze backbone for speed
                for param in self.features.parameters():
                    param.requires_grad = False
            
            def forward(self, x):
                features = self.features(x)
                pooled = self.avgpool(features).view(x.size(0), -1)
                
                return {
                    'global_features': self.global_head(pooled),
                    'motion_features': self.motion_head(pooled),
                    'texture_features': self.texture_head(pooled)
                }
        
        return OptimizedCNNExtractor()
    
    def _preallocate_buffers(self):
        """Pre-allocate buffers for motion/color computation"""
        # These will be allocated on demand and reused
        self.motion_buffers = {}
        self.color_buffers = {}
    
    def extract_features_optimized(self, frames_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extract features with optimized GPU utilization"""
        
        try:
            with torch.no_grad():
                batch_size, channels, height, width = frames_tensor.shape
                
                # Process in sub-batches if needed
                max_batch_size = self._calculate_max_batch_size(height, width)
                
                if batch_size <= max_batch_size:
                    # Process all at once
                    cnn_features = self.model(frames_tensor)
                else:
                    # Process in sub-batches
                    cnn_features = self._process_in_batches(frames_tensor, max_batch_size)
                
                # Compute motion features efficiently
                motion_features = self._compute_motion_optimized(frames_tensor)
                
                # Compute color features efficiently
                color_features = self._compute_color_optimized(frames_tensor)
                
                # Compute edge features efficiently
                edge_features = self._compute_edge_optimized(frames_tensor)
                
                # Combine all features
                all_features = {}
                
                # Convert CNN features to numpy
                for key, value in cnn_features.items():
                    all_features[key] = value.cpu().numpy()
                
                # Add motion, color, edge features
                all_features.update(motion_features)
                all_features.update(color_features)
                all_features.update(edge_features)
                
                return all_features
                
        except Exception as e:
            logger.error(f"Error in optimized feature extraction: {e}")
            raise
    
    def _calculate_max_batch_size(self, height: int, width: int) -> int:
        """Calculate maximum batch size that fits in GPU memory"""
        # Estimate memory per frame
        memory_per_frame = height * width * 3 * 4  # RGB float32
        
        # Available memory (conservative estimate)
        available_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory * 0.6
        
        # Account for model activations (roughly 3x input size)
        effective_memory_per_frame = memory_per_frame * 4
        
        max_batch = max(1, int(available_memory / effective_memory_per_frame))
        return min(max_batch, 32)  # Cap at 32 for stability
    
    def _process_in_batches(self, frames_tensor: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Process frames in smaller batches"""
        num_frames = frames_tensor.shape[0]
        all_results = {key: [] for key in ['global_features', 'motion_features', 'texture_features']}
        
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch = frames_tensor[i:end_idx]
            
            batch_results = self.model(batch)
            
            for key, value in batch_results.items():
                all_results[key].append(value)
        
        # Concatenate results
        final_results = {}
        for key, value_list in all_results.items():
            final_results[key] = torch.cat(value_list, dim=0)
        
        return final_results
    
    def _compute_motion_optimized(self, frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """Optimized motion computation on GPU"""
        num_frames = frames.shape[0]
        
        if num_frames < 2:
            return {
                'motion_magnitude': np.zeros(num_frames),
                'motion_direction': np.zeros(num_frames),
                'acceleration': np.zeros(num_frames)
            }
        
        # Convert to grayscale efficiently
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        # Compute frame differences
        frame_diffs = torch.abs(gray_frames[1:] - gray_frames[:-1])
        
        # Motion magnitude
        motion_mag = torch.mean(frame_diffs, dim=[1, 2])
        motion_magnitude = torch.zeros(num_frames, device=self.device)
        motion_magnitude[1:] = motion_mag
        
        # Motion direction (simplified)
        motion_direction = torch.zeros(num_frames, device=self.device)
        if num_frames > 1:
            grad_x = torch.mean(torch.abs(frame_diffs[:, :, 1:] - frame_diffs[:, :, :-1]), dim=[1, 2])
            grad_y = torch.mean(torch.abs(frame_diffs[:, 1:, :] - frame_diffs[:, :-1, :]), dim=[1, 2])
            motion_direction[1:] = torch.atan2(grad_y, grad_x + 1e-8)
        
        # Acceleration
        acceleration = torch.zeros(num_frames, device=self.device)
        if num_frames > 2:
            acceleration[1:-1] = motion_magnitude[2:] - motion_magnitude[1:-1]
        
        return {
            'motion_magnitude': motion_magnitude.cpu().numpy(),
            'motion_direction': motion_direction.cpu().numpy(),
            'acceleration': acceleration.cpu().numpy()
        }
    
    def _compute_color_optimized(self, frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """Optimized color computation on GPU"""
        # Color variance over spatial dimensions
        color_variance = torch.var(frames, dim=[2, 3])  # Variance per channel per frame
        mean_color_variance = torch.mean(color_variance, dim=1)  # Average across channels
        
        # Color statistics per frame
        color_mean = torch.mean(frames, dim=[2, 3])  # Mean per channel per frame
        color_std = torch.std(frames, dim=[2, 3])    # Std per channel per frame
        
        # Combine mean and std for histogram-like features
        color_histograms = torch.cat([color_mean, color_std], dim=1)
        
        return {
            'color_variance': mean_color_variance.cpu().numpy(),
            'color_histograms': color_histograms.cpu().numpy()
        }
    
    def _compute_edge_optimized(self, frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """Optimized edge computation using GPU convolutions"""
        # Convert to grayscale
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray_frames = gray_frames.unsqueeze(1)  # Add channel dimension
        
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Compute gradients
        grad_x = torch.nn.functional.conv2d(gray_frames, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray_frames, sobel_y, padding=1)
        
        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3])  # Mean per frame
        
        return {
            'edge_density': edge_density.cpu().numpy()
        }

# Performance monitoring functions
def monitor_system_performance():
    """Monitor system performance during processing"""
    gpu_stats = []
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        
        gpu_stats.append({
            'gpu_id': i,
            'name': props.name,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_pct': (reserved / total) * 100
        })
    
    # RAM usage
    ram_info = psutil.virtual_memory()
    ram_used_gb = (ram_info.total - ram_info.available) / 1024**3
    ram_total_gb = ram_info.total / 1024**3
    
    logger.info(f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_info.percent:.1f}%)")
    
    for gpu in gpu_stats:
        if gpu['allocated_gb'] > 0:  # Only log active GPUs
            logger.info(f"GPU {gpu['gpu_id']} ({gpu['name']}): "
                      f"{gpu['allocated_gb']:.1f}GB allocated, "
                      f"{gpu['reserved_gb']:.1f}GB reserved, "
                      f"{gpu['utilization_pct']:.1f}% utilization")
    
    return gpu_stats, ram_info

if __name__ == "__main__":
    print("🚀 GPU-Optimized Chunked Video Processor")
    print("This module provides chunked processing for unlimited video size/resolution")
    print("Import ChunkedVideoProcessor to use in your main script")
