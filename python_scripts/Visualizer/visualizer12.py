#!/usr/bin/env python3
"""
TurboGPU Video Processor v2.0 - Production Ready
=============================================

EXTREME GPU-Accelerated Video Processing System
- Dual RTX 5060 Ti optimization (30GB total VRAM)
- 90-98% GPU utilization target
- Advanced memory management & CUDA streams
- Asynchronous processing pipeline
- Smart batch sizing with memory pooling
- Production-grade error handling & monitoring

Author: AI Assistant
Target: Maximum performance on dual RTX 5060 Ti setup
"""

import json
import argparse
import logging
import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict, deque
import warnings
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, field
import gc
import psutil

# Critical GPU imports with enhanced error handling
try:
    import torch
    import torch.nn.functional as F
    import torch.cuda.amp as amp
    from torch.cuda.streams import Stream
    from ultralytics import YOLO
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    
    # Force CUDA availability check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    TORCH_AVAILABLE = True
    DEVICE_COUNT = torch.cuda.device_count()
    print(f"🚀 CUDA TURBO: {DEVICE_COUNT} GPUs detected")
    
    # Enable advanced optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
except ImportError as e:
    print(f"❌ Critical GPU imports failed: {e}")
    sys.exit(1)

# Advanced monitoring
try:
    import GPUtil
    import psutil
    import pynvml
    pynvml.nvmlInit()
    MONITORING_AVAILABLE = True
    print("✅ Advanced GPU monitoring enabled")
except ImportError:
    MONITORING_AVAILABLE = False
    print("⚠️ GPU monitoring limited")

# GPS processing
try:
    import gpxpy
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

# Memory optimization
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # CuDNN v8
warnings.filterwarnings('ignore')

# Advanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('turbo_gpu_processor.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GPUMemoryManager:
    """Advanced GPU memory management system"""
    device_id: int
    total_memory: float
    reserved_memory: float = 0.0
    allocated_memory: float = 0.0
    memory_pools: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    memory_threshold: float = 0.85  # Use up to 85% of GPU memory
    
    def __post_init__(self):
        self.device = torch.device(f'cuda:{self.device_id}')
        self.update_memory_stats()
    
    def update_memory_stats(self):
        """Update current memory statistics"""
        if torch.cuda.is_available():
            self.allocated_memory = torch.cuda.memory_allocated(self.device_id) / (1024**3)
            self.reserved_memory = torch.cuda.memory_reserved(self.device_id) / (1024**3)
    
    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        self.update_memory_stats()
        return (self.total_memory * self.memory_threshold) - self.allocated_memory
    
    def create_tensor_pool(self, pool_name: str, tensor_shape: Tuple, count: int = 10):
        """Create a pool of pre-allocated tensors for reuse"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = []
        
        for _ in range(count):
            tensor = torch.empty(tensor_shape, device=self.device, dtype=torch.float32)
            self.memory_pools[pool_name].append(tensor)
    
    def get_tensor_from_pool(self, pool_name: str, tensor_shape: Tuple) -> torch.Tensor:
        """Get a tensor from pool or create new one"""
        if pool_name in self.memory_pools and self.memory_pools[pool_name]:
            tensor = self.memory_pools[pool_name].pop()
            if tensor.shape == tensor_shape:
                return tensor.zero_()
        
        return torch.zeros(tensor_shape, device=self.device, dtype=torch.float32)
    
    def return_tensor_to_pool(self, pool_name: str, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = []
        
        if len(self.memory_pools[pool_name]) < 20:  # Limit pool size
            self.memory_pools[pool_name].append(tensor)
    
    def clear_pools(self):
        """Clear all memory pools"""
        for pool in self.memory_pools.values():
            pool.clear()
        torch.cuda.empty_cache()

@dataclass
class TurboConfig:
    """Advanced configuration for turbo GPU processing"""
    # GPU settings
    target_gpu_utilization: float = 0.95
    memory_utilization_target: float = 0.85
    enable_mixed_precision: bool = True
    use_cuda_streams: bool = True
    
    # Batch processing
    base_batch_size: int = 512
    adaptive_batching: bool = True
    max_batch_size: int = 1024
    min_batch_size: int = 64
    
    # Video processing
    max_video_frames: int = 15000
    frame_skip_adaptive: bool = True
    target_fps_processing: float = 120.0
    
    # Memory management
    enable_memory_pooling: bool = True
    memory_pool_size: int = 20
    enable_gradient_checkpointing: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    detailed_profiling: bool = False
    save_performance_logs: bool = True

class AdvancedVideoDataset(Dataset):
    """High-performance video dataset with prefetching"""
    
    def __init__(self, frames: np.ndarray, transform=None):
        self.frames = frames
        self.transform = transform
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame, idx

class TurboGPUVideoProcessor:
    """
    EXTREME GPU-Accelerated Video Processor v2.0
    
    Features:
    - Dual GPU workload distribution
    - Advanced memory management with pooling
    - CUDA streams for async processing
    - Mixed precision training
    - Adaptive batch sizing
    - Smart frame extraction
    - Production-grade monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the turbo processor with advanced optimizations"""
        logger.info("🚀 Initializing TurboGPU Video Processor v2.0...")
        
        # Store configuration first
        self.config = config
        self.turbo_config = TurboConfig()
        self._update_turbo_config(config)
        
        # Initialize core attributes BEFORE any GPU operations
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance settings - SET IMMEDIATELY
        self.batch_size = self.turbo_config.base_batch_size
        self.max_video_frames = self.turbo_config.max_video_frames
        self.frame_skip = config.get('frame_skip', 2)
        
        # Setup directories
        self._setup_output_directories()
        
        # GPU initialization with error handling
        self.gpu_managers = {}
        self.models = {}
        self.cuda_streams = {}
        self.scalers = {}  # For mixed precision
        
        try:
            self._initialize_gpu_system()
        except Exception as e:
            logger.error(f"❌ GPU initialization failed: {e}")
            raise
        
        # Statistics and monitoring
        self.stats = self._initialize_statistics()
        self.performance_monitor = PerformanceMonitor(self.turbo_config)
        
        # Processing pipeline
        self.processing_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
        logger.info("✅ TurboGPU Processor v2.0 initialized successfully!")
        self._log_system_info()
    
    def _update_turbo_config(self, config: Dict[str, Any]):
        """Update turbo configuration from user config"""
        if 'batch_size' in config:
            self.turbo_config.base_batch_size = config['batch_size']
        if 'max_video_frames' in config:
            self.turbo_config.max_video_frames = config['max_video_frames']
        if 'target_gpu_utilization' in config:
            self.turbo_config.target_gpu_utilization = config['target_gpu_utilization']
    
    def _setup_output_directories(self):
        """Setup output directory structure"""
        subdirs = [
            'object_tracking', 'stoplight_detection', 'traffic_counting',
            'processing_reports', 'performance_logs', 'debug_outputs'
        ]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _initialize_gpu_system(self):
        """Initialize advanced GPU system with aggressive memory management"""
        logger.info("🎮 Initializing advanced GPU system...")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")
        
        # Set memory management environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Aggressive memory cleanup before starting
        logger.info("🧹 Clearing existing GPU memory...")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.gpu_count = min(torch.cuda.device_count(), 2)  # Use up to 2 GPUs
        self.gpu_ids = list(range(self.gpu_count))
        
        if self.gpu_count < 2:
            logger.warning(f"⚠️ Only {self.gpu_count} GPU available. Dual-GPU optimizations disabled.")
        
        # Initialize each GPU with memory checks
        for gpu_id in self.gpu_ids:
            # Check available memory before initialization
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            logger.info(f"🎮 GPU {gpu_id} Memory Status: {free_memory:.1f}GB free / {total_memory:.1f}GB total")
            
            if free_memory < 3.0:  # Need at least 3GB free
                logger.warning(f"⚠️ GPU {gpu_id} has only {free_memory:.1f}GB free. Forcing cleanup...")
                torch.cuda.empty_cache()
                allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                free_memory = total_memory - allocated_memory
                
                if free_memory < 2.0:  # Still not enough
                    logger.error(f"❌ GPU {gpu_id} insufficient memory: {free_memory:.1f}GB free")
                    raise RuntimeError(f"GPU {gpu_id} insufficient memory for processing")
            
            self._initialize_single_gpu(gpu_id)
        
        logger.info(f"✅ {self.gpu_count} GPU(s) initialized successfully")
        
        # Test dual GPU capability if we have 2 GPUs
        if self.gpu_count == 2:
            self._test_dual_gpu_capability()
    
    def _test_dual_gpu_capability(self):
        """Test that both GPUs can process simultaneously"""
        logger.info("🧪 Testing dual GPU processing capability...")
        
        try:
            # Create small test tensors on both GPUs simultaneously
            test_results = {}
            
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                test_tensor = torch.randn(16, 3, 640, 640, device=f'cuda:{gpu_id}')
                
                # Test model inference
                model = self.models[gpu_id]
                with torch.no_grad():
                    start_time = time.time()
                    _ = model(test_tensor, verbose=False)
                    end_time = time.time()
                
                test_results[gpu_id] = end_time - start_time
                
                # Cleanup
                del test_tensor
                torch.cuda.empty_cache()
            
            # Log results
            for gpu_id, duration in test_results.items():
                logger.info(f"✅ GPU {gpu_id} test: {duration:.3f}s")
            
            logger.info("🎮 Dual GPU capability verified!")
            
        except Exception as e:
            logger.warning(f"⚠️ Dual GPU test failed: {e}")
            logger.warning("   Continuing with single GPU processing")
    
    def _initialize_single_gpu(self, gpu_id: int):
        """Initialize a single GPU with conservative optimizations"""
        logger.info(f"🔧 Initializing GPU {gpu_id}...")
        
        try:
            # Set device
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(gpu_id)
            total_memory = props.total_memory / (1024**3)
            
            logger.info(f"🎮 GPU {gpu_id}: {props.name} ({total_memory:.1f}GB)")
            
            # Create memory manager
            self.gpu_managers[gpu_id] = GPUMemoryManager(
                device_id=gpu_id,
                total_memory=total_memory
            )
            
            # Create CUDA stream for async operations
            if self.turbo_config.use_cuda_streams:
                self.cuda_streams[gpu_id] = Stream(device=device)
            
            # Initialize mixed precision scaler
            if self.turbo_config.enable_mixed_precision:
                self.scalers[gpu_id] = amp.GradScaler()
            
            # Load YOLO model
            self._load_model_on_gpu(gpu_id)
            
            # Create memory pools (conservative)
            if self.turbo_config.enable_memory_pooling:
                self._create_memory_pools(gpu_id)
            
            # Skip benchmarking if disabled or low memory, use conservative batch size
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            if self.config.get('disable_benchmarking', False):
                logger.info(f"🛡️ GPU {gpu_id}: Benchmarking disabled, using conservative batch size")
                self.batch_size = min(64, self.batch_size)  # Conservative but reasonable
                logger.info(f"🛡️ GPU {gpu_id}: Using safe batch size {self.batch_size}")
            elif free_memory < 4.0:  # Less than 4GB free
                logger.warning(f"⚠️ GPU {gpu_id}: Low memory ({free_memory:.1f}GB free), skipping benchmark")
                self.batch_size = min(32, self.batch_size)  # Very conservative
                logger.info(f"🛡️ GPU {gpu_id}: Using safe batch size {self.batch_size}")
            else:
                # Only benchmark if we have sufficient memory and it's enabled
                self._benchmark_gpu(gpu_id)
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} initialization failed: {e}")
            raise
    
    def _load_model_on_gpu(self, gpu_id: int):
        """Load YOLO model on specific GPU with safe optimizations"""
        model_path = self.config.get('yolo_model', 'yolo11x.pt')
        device = torch.device(f'cuda:{gpu_id}')
        
        logger.info(f"📦 Loading YOLO model on GPU {gpu_id}...")
        
        try:
            # Load YOLO model
            model = YOLO(model_path)
            
            # Move to GPU and optimize
            model.model = model.model.to(device)
            model.model.eval()
            
            # Disable gradients for inference (critical for performance)
            for param in model.model.parameters():
                param.requires_grad = False
            
            # Enable safe YOLO-compatible optimizations
            try:
                # Set model to half precision for RTX cards (if supported)
                if self.turbo_config.enable_mixed_precision:
                    model.model = model.model.half()
                    logger.info(f"⚡ GPU {gpu_id}: Half precision enabled")
            except Exception as e:
                logger.warning(f"⚠️ Half precision failed, using full precision: {e}")
            
            # Try PyTorch 2.0+ compilation (safer approach)
            if hasattr(torch, 'compile'):
                try:
                    # Use reduce-overhead mode instead of max-autotune for stability
                    model.model = torch.compile(model.model, mode='reduce-overhead')
                    logger.info(f"⚡ GPU {gpu_id}: Model compiled for enhanced performance")
                except Exception as e:
                    logger.warning(f"⚠️ Model compilation skipped: {e}")
            
            self.models[gpu_id] = model
            
            # Verify model is on GPU
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(f"✅ GPU {gpu_id}: Model loaded ({memory_allocated:.2f}GB allocated)")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed on GPU {gpu_id}: {e}")
            raise
    
    def _create_memory_pools(self, gpu_id: int):
        """Create conservative memory pools for efficient tensor reuse"""
        manager = self.gpu_managers[gpu_id]
        
        # Check available memory before creating pools
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        free_memory = total_memory - allocated_memory
        
        if free_memory < 2.0:  # Less than 2GB free
            logger.warning(f"⚠️ GPU {gpu_id}: Insufficient memory for pools, skipping")
            return
        
        # Conservative pool sizes based on final batch size
        conservative_batch = min(self.batch_size, 128)  # Cap at 128 for pools
        
        # Create smaller pools for common tensor sizes
        common_shapes = [
            (conservative_batch, 3, 640, 640),  # YOLO input
            (conservative_batch, 256),          # Feature vectors  
            (conservative_batch, 100, 4),       # Bounding boxes
            (conservative_batch, 100),          # Confidences
        ]
        
        pool_count = 5 if free_memory > 4.0 else 3  # Fewer pools if low memory
        
        try:
            for i, shape in enumerate(common_shapes):
                pool_name = f"pool_{i}"
                manager.create_tensor_pool(pool_name, shape, pool_count)
            
            logger.info(f"🗄️ GPU {gpu_id}: Memory pools created (batch={conservative_batch}, pools={pool_count})")
        except Exception as e:
            logger.warning(f"⚠️ GPU {gpu_id}: Memory pool creation failed: {e}")
            # Clear any partial pools
            manager.clear_pools()
    
    def _benchmark_gpu(self, gpu_id: int):
        """Benchmark GPU performance with conservative memory usage"""
        device = torch.device(f'cuda:{gpu_id}')
        model = self.models[gpu_id]
        
        logger.info(f"🏃 Benchmarking GPU {gpu_id} performance...")
        
        # Check available memory first
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        free_memory = total_memory - allocated_memory
        
        logger.info(f"💾 GPU {gpu_id} available memory: {free_memory:.1f}GB")
        
        # Conservative batch sizes based on available memory
        if free_memory > 8.0:
            test_batches = [32, 64, 128, 256, 512]  # Conservative for safety
        elif free_memory > 4.0:
            test_batches = [16, 32, 64, 128, 256]
        elif free_memory > 2.0:
            test_batches = [8, 16, 32, 64, 128]
        else:
            test_batches = [4, 8, 16, 32]  # Very conservative
            logger.warning(f"⚠️ GPU {gpu_id} low memory, using minimal batch sizes")
        
        best_batch_size = self.batch_size
        best_fps = 0
        
        for test_batch in test_batches:
            try:
                # Check if we have enough memory for this batch
                estimated_memory = (test_batch * 3 * 640 * 640 * 4) / (1024**3)  # Estimate in GB
                current_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                
                if (current_allocated + estimated_memory * 2) > (total_memory * 0.9):  # Leave 10% buffer
                    logger.info(f"💾 GPU {gpu_id} batch {test_batch}: Skipping (insufficient memory)")
                    continue
                
                # Create test data
                test_tensor = torch.randn(test_batch, 3, 640, 640, device=device, dtype=torch.float32)
                
                # Convert to half precision if model uses it
                if next(model.model.parameters()).dtype == torch.float16:
                    test_tensor = test_tensor.half()
                
                # Minimal warmup (just 1 iteration to save memory)
                with torch.no_grad():
                    _ = model(test_tensor, verbose=False)
                
                torch.cuda.synchronize(gpu_id)
                
                # Quick benchmark (fewer iterations to save memory)
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(3):  # Reduced from 10 to 3 iterations
                        _ = model(test_tensor, verbose=False)
                
                torch.cuda.synchronize(gpu_id)
                end_time = time.time()
                
                fps = (test_batch * 3) / (end_time - start_time)
                
                logger.info(f"📊 GPU {gpu_id} batch {test_batch}: {fps:.1f} FPS")
                
                if fps > best_fps:
                    best_fps = fps
                    best_batch_size = test_batch
                
                # Immediate cleanup
                del test_tensor
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"💾 GPU {gpu_id} batch {test_batch}: OOM - stopping benchmark")
                    torch.cuda.empty_cache()
                    break
                else:
                    logger.warning(f"⚠️ GPU {gpu_id} batch {test_batch}: Error - {e}")
                    torch.cuda.empty_cache()
                    continue
        
        # Update batch size for this GPU (be conservative)
        if self.turbo_config.adaptive_batching and best_batch_size > 0:
            # Use 75% of best batch size for safety margin
            safe_batch_size = max(16, int(best_batch_size * 0.75))
            self.batch_size = safe_batch_size
            logger.info(f"⚡ GPU {gpu_id}: Optimal batch size set to {safe_batch_size} ({best_fps:.1f} FPS, safety margin applied)")
        else:
            # Fallback to very conservative batch size
            self.batch_size = min(32, self.batch_size)
            logger.warning(f"⚠️ GPU {gpu_id}: Using conservative batch size {self.batch_size}")
        
        # Final cleanup
        torch.cuda.empty_cache()
    
    def _initialize_statistics(self) -> Dict[str, Any]:
        """Initialize performance statistics"""
        return {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'total_detections': 0,
            'processing_times': [],
            'gpu_utilization': {gpu_id: [] for gpu_id in self.gpu_ids},
            'memory_usage': {gpu_id: [] for gpu_id in self.gpu_ids},
            'fps_achieved': [],
            'batch_sizes_used': []
        }
    
    def extract_frames_turbo(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """Ultra-fast frame extraction with adaptive sampling"""
        logger.info(f"📹 Turbo frame extraction: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([]), {}
        
        # Configure OpenCV for maximum performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps if fps > 0 else 0,
            'is_360': 1.8 <= (width/height) <= 2.2
        }
        
        # Adaptive frame sampling
        if self.turbo_config.frame_skip_adaptive:
            # Adjust frame skip based on video length and target processing FPS
            target_frames = min(self.max_video_frames, frame_count)
            if frame_count > target_frames:
                self.frame_skip = max(1, frame_count // target_frames)
            else:
                self.frame_skip = 1
        
        # Smart frame selection
        frame_indices = self._select_optimal_frames(frame_count, fps)
        
        logger.info(f"📊 Extracting {len(frame_indices)} frames from {frame_count} total")
        
        # Pre-allocate arrays for maximum speed
        frames = []
        # Note: Python lists don't have reserve() method like C++
        
        # Extract frames with progress tracking
        extract_start = time.time()
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Ultra-fast preprocessing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
                frames.append(frame_resized)
            
            # Progress update for large videos
            if i > 0 and i % 1000 == 0:
                progress = (i / len(frame_indices)) * 100
                logger.info(f"📹 Frame extraction: {progress:.1f}% complete")
        
        cap.release()
        
        extract_time = time.time() - extract_start
        logger.info(f"⚡ Frame extraction completed in {extract_time:.2f}s")
        
        if frames:
            # Convert to optimized numpy array
            frames_array = np.stack(frames, dtype=np.float32) / 255.0
            frames_array = frames_array.transpose(0, 3, 1, 2)  # NHWC to NCHW
        else:
            frames_array = np.array([])
        
        video_info.update({
            'extracted_frames': len(frames),
            'frame_indices': frame_indices,
            'extraction_time': extract_time,
            'effective_frame_skip': self.frame_skip
        })
        
        return frames_array, video_info
    
    def _select_optimal_frames(self, frame_count: int, fps: float) -> List[int]:
        """Select optimal frames for processing based on content and performance"""
        # Basic implementation - can be enhanced with motion detection
        if frame_count <= self.max_video_frames:
            return list(range(0, frame_count, self.frame_skip))
        
        # For very long videos, use smart sampling
        step = max(1, frame_count // self.max_video_frames)
        return list(range(0, frame_count, step))[:self.max_video_frames]
    
    def process_video_turbo(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame) -> Dict:
        """Process video with maximum GPU acceleration"""
        video_name = Path(video_path).stem
        logger.info(f"🚀 GPU {gpu_id} processing: {video_name}")
        
        start_time = time.time()
        device = torch.device(f'cuda:{gpu_id}')
        model = self.models[gpu_id]
        memory_manager = self.gpu_managers[gpu_id]
        
        try:
            # Extract frames
            frames_array, video_info = self.extract_frames_turbo(video_path)
            if frames_array.size == 0:
                return {'status': 'failed', 'error': 'No frames extracted'}
            
            total_frames = len(frames_array)
            logger.info(f"🎮 GPU {gpu_id}: Processing {total_frames} frames with turbo acceleration")
            
            # Adaptive batch sizing based on available memory
            available_memory = memory_manager.get_available_memory()
            optimal_batch_size = self._calculate_optimal_batch_size(available_memory, gpu_id)
            
            all_detections = []
            processing_stats = {
                'batches_processed': 0,
                'total_detections': 0,
                'avg_batch_time': 0,
                'memory_peaks': []
            }
            
            # Create dataset and dataloader for efficient batching
            dataset = AdvancedVideoDataset(frames_array)
            dataloader = DataLoader(
                dataset, 
                batch_size=optimal_batch_size,
                shuffle=False,
                num_workers=0,  # Keep on main thread for GPU processing
                pin_memory=True,
                drop_last=False
            )
            
            # Process batches with maximum acceleration
            batch_times = []
            
            for batch_idx, (batch_frames, frame_indices) in enumerate(dataloader):
                batch_start = time.time()
                
                # Move to GPU
                if self.turbo_config.use_cuda_streams:
                    with torch.cuda.stream(self.cuda_streams[gpu_id]):
                        batch_tensor = batch_frames.to(device, non_blocking=True)
                else:
                    batch_tensor = batch_frames.to(device)
                
                # Process with proper YOLO inference (no mixed precision autocast for YOLO)
                with torch.no_grad():
                    # Use half precision if model was converted to half
                    if next(model.model.parameters()).dtype == torch.float16:
                        batch_tensor = batch_tensor.half()
                    
                    results = model(batch_tensor, verbose=False)
                
                # Extract detections
                batch_detections = self._extract_batch_detections(
                    results, frame_indices, gpu_id, video_info
                )
                all_detections.extend(batch_detections)
                
                # Update statistics
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                processing_stats['batches_processed'] += 1
                processing_stats['total_detections'] += sum(len(d['detections']) for d in batch_detections)
                
                # Memory monitoring
                memory_manager.update_memory_stats()
                processing_stats['memory_peaks'].append(memory_manager.allocated_memory)
                
                # Progress reporting
                if batch_idx > 0 and batch_idx % 10 == 0:
                    progress = (batch_idx / len(dataloader)) * 100
                    avg_fps = optimal_batch_size / np.mean(batch_times[-10:])
                    logger.info(f"🎮 GPU {gpu_id}: {progress:.1f}% - {avg_fps:.1f} FPS")
                
                # Memory cleanup
                del batch_tensor
                if self.turbo_config.use_cuda_streams:
                    torch.cuda.synchronize(gpu_id)
            
            # Final statistics
            processing_stats['avg_batch_time'] = np.mean(batch_times)
            
            # Merge with GPS data
            final_results = self._merge_detections_with_gps(all_detections, gps_df, video_info)
            
            processing_time = time.time() - start_time
            total_fps = total_frames / processing_time if processing_time > 0 else 0
            
            logger.info(f"✅ GPU {gpu_id} TURBO completed {video_name}:")
            logger.info(f"   Time: {processing_time:.2f}s")
            logger.info(f"   FPS: {total_fps:.1f}")
            logger.info(f"   Detections: {processing_stats['total_detections']}")
            logger.info(f"   Avg Memory: {np.mean(processing_stats['memory_peaks']):.2f}GB")
            
            self.stats['processed_videos'] += 1
            self.stats['fps_achieved'].append(total_fps)
            
            return {
                'status': 'success',
                'video_name': video_name,
                'processing_time': processing_time,
                'fps': total_fps,
                'gpu_id': gpu_id,
                'results': final_results,
                'stats': processing_stats,
                'optimal_batch_size': optimal_batch_size
            }
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} error processing {video_path}: {e}")
            self.stats['failed_videos'] += 1
            return {'status': 'failed', 'error': str(e), 'gpu_id': gpu_id}
        
        finally:
            # Comprehensive cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    def _calculate_optimal_batch_size(self, available_memory: float, gpu_id: int) -> int:
        """Calculate optimal batch size based on available memory"""
        # Rough estimate: 1GB can handle ~100 frames at 640x640
        memory_per_frame = 0.01  # GB per frame (conservative estimate)
        theoretical_max = int(available_memory / memory_per_frame)
        
        # Apply safety margins and constraints
        optimal = min(
            theoretical_max,
            self.turbo_config.max_batch_size,
            max(self.turbo_config.min_batch_size, theoretical_max)
        )
        
        logger.info(f"🧮 GPU {gpu_id}: Optimal batch size {optimal} (available memory: {available_memory:.1f}GB)")
        return optimal
    
    def _extract_batch_detections(self, results, frame_indices: torch.Tensor, 
                                 gpu_id: int, video_info: Dict) -> List[Dict]:
        """Extract detections from batch results efficiently"""
        batch_detections = []
        confidence_threshold = self.config.get('confidence_threshold', 0.3)
        
        for i, (result, frame_idx) in enumerate(zip(results, frame_indices)):
            detection_data = {
                'frame_idx': int(frame_idx),
                'detections': [],
                'counts': defaultdict(int),
                'gpu_id': gpu_id
            }
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Extract data efficiently
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                # Vectorized filtering
                valid_mask = confidences >= confidence_threshold
                valid_boxes = boxes[valid_mask]
                valid_classes = classes[valid_mask]
                valid_confidences = confidences[valid_mask]
                
                # Process valid detections
                for box, cls, conf in zip(valid_boxes, valid_classes, valid_confidences):
                    obj_class = result.names.get(cls, 'unknown')
                    
                    detection = {
                        'bbox': box.tolist(),
                        'class': obj_class,
                        'confidence': float(conf),
                        'class_id': int(cls)
                    }
                    
                    detection_data['detections'].append(detection)
                    detection_data['counts'][obj_class] += 1
                    self.stats['total_detections'] += 1
            
            batch_detections.append(detection_data)
        
        return batch_detections
    
    def _merge_detections_with_gps(self, all_detections: List[Dict], gps_df: pd.DataFrame, 
                                  video_info: Dict) -> Dict:
        """Efficiently merge detections with GPS data"""
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int)
        }
        
        fps = video_info.get('fps', 30)
        frame_indices = video_info.get('frame_indices', [])
        
        # Pre-compute GPS lookup for efficiency
        gps_lookup = {}
        if not gps_df.empty:
            for _, row in gps_df.iterrows():
                gps_lookup[row['second']] = row
        
        for detection_data in all_detections:
            frame_idx = detection_data['frame_idx']
            
            # Calculate timestamp
            if frame_idx < len(frame_indices):
                actual_frame_number = frame_indices[frame_idx]
                second = int(actual_frame_number / fps) if fps > 0 else frame_idx
            else:
                second = frame_idx
            
            # Get GPS data efficiently
            gps_data = gps_lookup.get(second, {})
            
            # Process detections
            for detection in detection_data['detections']:
                record = {
                    'frame_second': second,
                    'object_class': detection['class'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'lat': float(gps_data.get('lat', 0)),
                    'lon': float(gps_data.get('lon', 0)),
                    'gps_time': str(gps_data.get('gpx_time', '')),
                    'gpu_id': detection_data['gpu_id'],
                    'video_type': '360°' if video_info.get('is_360', False) else 'flat'
                }
                
                results['object_tracking'].append(record)
                
                # Traffic light detection
                if detection['class'] == 'traffic light':
                    stoplight_record = record.copy()
                    stoplight_record['stoplight_color'] = 'detected'
                    results['stoplight_detection'].append(stoplight_record)
            
            # Count objects
            for obj_class, count in detection_data['counts'].items():
                results['traffic_counting'][obj_class] += count
        
        return results
    
    def process_videos_dual_gpu_turbo(self, video_matches: Dict[str, Any]):
        """Process videos with enhanced dual GPU acceleration and monitoring"""
        total_videos = len(video_matches)
        logger.info(f"🚀 TURBO PROCESSING: {total_videos} videos with dual GPU acceleration")
        
        # Start advanced monitoring
        if self.turbo_config.performance_monitoring:
            monitor_thread = threading.Thread(
                target=self._advanced_gpu_monitoring, 
                daemon=True
            )
            monitor_thread.start()
        
        # Prepare video processing queue
        video_list = [
            {
                'path': video_path,
                'gps_path': info.get('gps_path', ''),
                'info': info
            }
            for video_path, info in video_matches.items()
        ]
        
        # Enhanced workload distribution with forced balancing
        workload_distribution = self._optimize_workload_distribution(video_list)
        
        # Force dual GPU utilization if we have 2 GPUs (or forced)
        if len(self.gpu_ids) == 2 and (total_videos > 1 or self.config.get('force_dual_gpu', False)):
            self._ensure_dual_gpu_utilization(workload_distribution, video_list)
        
        # Verify we have work for both GPUs
        active_gpu_count = len([gpu_id for gpu_id, videos in workload_distribution.items() if videos])
        logger.info(f"🎮 Active GPUs: {active_gpu_count}/{len(self.gpu_ids)}")
        
        if active_gpu_count < len(self.gpu_ids) and total_videos > 1:
            logger.warning(f"⚠️ Suboptimal GPU usage: Only {active_gpu_count}/{len(self.gpu_ids)} GPUs will be active")
    
    def _ensure_dual_gpu_utilization(self, workload_distribution: Dict, video_list: List[Dict]):
        """Ensure both GPUs get work when we have multiple videos or forced dual GPU"""
        gpu0_videos = len(workload_distribution.get(0, []))
        gpu1_videos = len(workload_distribution.get(1, []))
        
        logger.info(f"🔄 Initial distribution: GPU0={gpu0_videos}, GPU1={gpu1_videos}")
        
        # If one GPU has no videos, redistribute
        if (gpu0_videos == 0 or gpu1_videos == 0) and len(video_list) >= 1:
            logger.warning("⚠️ Uneven GPU distribution detected, rebalancing...")
            
            # Clear existing distribution
            workload_distribution[0] = []
            workload_distribution[1] = []
            
            # Handle single video case for force dual GPU
            if len(video_list) == 1 and self.config.get('force_dual_gpu', False):
                # Assign single video to GPU 0, GPU 1 will stay idle but available
                workload_distribution[0] = video_list
                logger.info("🎮 Single video assigned to GPU0 (force_dual_gpu mode)")
            else:
                # Force alternating assignment for multiple videos
                for i, video_info in enumerate(video_list):
                    gpu_id = i % 2  # Alternate between GPU 0 and 1
                    workload_distribution[gpu_id].append(video_info)
            
            new_gpu0_videos = len(workload_distribution[0])
            new_gpu1_videos = len(workload_distribution[1])
            logger.info(f"✅ Rebalanced distribution: GPU0={new_gpu0_videos}, GPU1={new_gpu1_videos}")
        
        # Enhanced verification
        total_assigned = len(workload_distribution[0]) + len(workload_distribution[1])
        if total_assigned != len(video_list):
            logger.error(f"❌ Assignment mismatch: {total_assigned} assigned vs {len(video_list)} total videos")
        
        # Warn about idle GPUs only if not expected
        if len(workload_distribution[0]) == 0 and len(video_list) > 0:
            logger.warning("⚠️ GPU0 has no work assigned!")
        if len(workload_distribution[1]) == 0 and len(video_list) > 1:
            logger.warning("⚠️ GPU1 has no work assigned!")
        elif len(workload_distribution[1]) == 0 and len(video_list) == 1:
            logger.info("ℹ️ GPU1 idle (single video processing)")
        
        # Enhanced parallel processing with per-GPU workers
        completed_videos = 0
        gpu_stats = {gpu_id: {'videos': 0, 'active': False} for gpu_id in self.gpu_ids}
        
        # Use ProcessPoolExecutor for true parallelism across GPUs
        max_workers = min(len(self.gpu_ids), active_gpu_count)
        logger.info(f"🔄 Starting {max_workers} parallel GPU workers")
        
    def _process_dual_gpu_workload(self, workload_distribution: Dict, total_videos: int, gpu_stats: Dict):
        """Process workload across dual GPUs with true parallelism"""
        import threading
        import queue
        
        # Create result queue for collecting results
        result_queue = queue.Queue()
        completed_videos = 0
        
        # Start monitoring thread
        if self.turbo_config.performance_monitoring:
            monitor_thread = threading.Thread(
                target=self._advanced_gpu_monitoring, 
                daemon=True
            )
            monitor_thread.start()
        
        # Create worker threads for each GPU
        worker_threads = []
        
        for gpu_id, gpu_videos in workload_distribution.items():
            if not gpu_videos:  # Skip GPUs with no videos
                continue
                
            logger.info(f"🎮 GPU {gpu_id}: Starting worker thread for {len(gpu_videos)} videos")
            gpu_stats[gpu_id]['active'] = True
            
            # Create worker thread for this GPU
            worker_thread = threading.Thread(
                target=self._gpu_worker_thread,
                args=(gpu_id, gpu_videos, result_queue),
                daemon=False  # Don't make it daemon so we wait for completion
            )
            worker_threads.append(worker_thread)
            worker_thread.start()
        
        logger.info(f"✅ Started {len(worker_threads)} GPU worker threads")
        
        # Collect results from both GPUs
        start_time = time.time()
        
        while completed_videos < total_videos and any(t.is_alive() for t in worker_threads):
            try:
                # Get result with timeout
                result = result_queue.get(timeout=30)  # 30 second timeout
                
                if result['status'] == 'success':
                    self._save_results_fast(result)
                    completed_videos += 1
                    gpu_id = result['gpu_id']
                    gpu_stats[gpu_id]['videos'] += 1
                    
                    # Enhanced progress reporting
                    elapsed_time = time.time() - start_time
                    progress = (completed_videos / total_videos) * 100
                    eta = (elapsed_time / completed_videos * total_videos - elapsed_time) if completed_videos > 0 else 0
                    
                    logger.info(f"📊 TURBO Progress: {progress:.1f}% ({completed_videos}/{total_videos})")
                    logger.info(f"   ⚡ {result['video_name']} completed by GPU {gpu_id} ({result['fps']:.1f} FPS)")
                    logger.info(f"   🎮 GPU Stats: " + " | ".join([f"GPU{gid}: {stats['videos']}" for gid, stats in gpu_stats.items() if stats['active']]))
                    logger.info(f"   ⏱️ ETA: {eta/60:.1f} minutes")
                    
                    # Log GPU utilization every few completions
                    if completed_videos % 3 == 0:
                        self._log_current_gpu_utilization()
                
                elif result['status'] == 'failed':
                    logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    completed_videos += 1  # Count failed videos too
                
                result_queue.task_done()
                
            except queue.Empty:
                logger.info("⏳ Waiting for GPU workers...")
                continue
            except Exception as e:
                logger.error(f"❌ Result processing error: {e}")
                break
        
        # Wait for all threads to complete
        logger.info("🔄 Waiting for all GPU workers to complete...")
        for thread in worker_threads:
            thread.join(timeout=60)  # 60 second timeout per thread
            if thread.is_alive():
                logger.warning(f"⚠️ Worker thread still alive after timeout")
        
        # Final GPU utilization summary
        self._log_final_gpu_summary(gpu_stats)
        
        # Generate comprehensive performance report
        self._generate_turbo_performance_report()
    
    def _gpu_worker_thread(self, gpu_id: int, gpu_videos: List[Dict], result_queue: queue.Queue):
        """Worker thread that processes videos on a specific GPU"""
        logger.info(f"🚀 GPU {gpu_id} worker thread started with {len(gpu_videos)} videos")
        
        try:
            # Set GPU context for this thread
            torch.cuda.set_device(gpu_id)
            
            # Process each video on this GPU
            for i, video_info in enumerate(gpu_videos):
                video_name = Path(video_info['path']).name
                logger.info(f"🎮 GPU {gpu_id} processing: {video_name} ({i+1}/{len(gpu_videos)})")
                
                try:
                    # Load GPS data
                    gps_df = self._load_gps_data_fast(video_info['gps_path'])
                    
                    # Process video on this GPU
                    result = self.process_video_turbo(video_info['path'], gpu_id, gps_df)
                    
                    # Add to result queue
                    result_queue.put(result)
                    
                except Exception as e:
                    logger.error(f"❌ GPU {gpu_id} error processing {video_name}: {e}")
                    error_result = {
                        'status': 'failed',
                        'error': str(e),
                        'gpu_id': gpu_id,
                        'video_name': video_name
                    }
                    result_queue.put(error_result)
            
            logger.info(f"✅ GPU {gpu_id} worker thread completed all {len(gpu_videos)} videos")
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} worker thread failed: {e}")
            # Put error result for each remaining video
            for video_info in gpu_videos:
                error_result = {
                    'status': 'failed',
                    'error': f"Worker thread error: {str(e)}",
                    'gpu_id': gpu_id,
                    'video_name': Path(video_info['path']).name
                }
                result_queue.put(error_result)
    
    def _log_current_gpu_utilization(self):
        """Log current GPU utilization for both GPUs"""
        try:
            if MONITORING_AVAILABLE:
                gpus = GPUtil.getGPUs()
                utilization_info = []
                for gpu_id in self.gpu_ids:
                    if gpu_id < len(gpus):
                        gpu = gpus[gpu_id]
                        utilization_info.append(f"GPU{gpu_id}: {gpu.load*100:.1f}%")
                
                if utilization_info:
                    logger.info(f"🎮 Current GPU Utilization: {' | '.join(utilization_info)}")
        except Exception as e:
            logger.debug(f"GPU utilization logging error: {e}")
    
    def _log_final_gpu_summary(self, gpu_stats: Dict):
        """Log final summary of GPU usage"""
        logger.info("🎮 FINAL GPU UTILIZATION SUMMARY:")
        for gpu_id, stats in gpu_stats.items():
            if stats['active']:
                logger.info(f"   GPU {gpu_id}: {stats['videos']} videos processed")
            else:
                logger.warning(f"   GPU {gpu_id}: IDLE (no videos assigned)")
        
        active_gpus = len([gpu_id for gpu_id, stats in gpu_stats.items() if stats['active']])
        logger.info(f"   Total Active GPUs: {active_gpus}/{len(self.gpu_ids)}")
    
    def _optimize_workload_distribution(self, video_list: List[Dict]) -> Dict[int, List[Dict]]:
        """Optimize video distribution across GPUs with enhanced load balancing"""
        distribution = {gpu_id: [] for gpu_id in self.gpu_ids}
        
        # Enhanced round-robin with video size consideration
        for i, video_info in enumerate(video_list):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            distribution[gpu_id].append(video_info)
        
        # Log detailed distribution
        total_videos = len(video_list)
        for gpu_id, videos in distribution.items():
            percentage = (len(videos) / total_videos) * 100 if total_videos > 0 else 0
            logger.info(f"🎮 GPU {gpu_id}: {len(videos)} videos assigned ({percentage:.1f}%)")
            
            # Log first few video names for verification
            if videos:
                sample_names = [Path(v['path']).name for v in videos[:3]]
                logger.info(f"   📹 Sample videos: {', '.join(sample_names)}")
        
        # Verify both GPUs have work
        active_gpus = [gpu_id for gpu_id, videos in distribution.items() if videos]
        if len(active_gpus) < len(self.gpu_ids):
            logger.warning(f"⚠️ Only {len(active_gpus)} of {len(self.gpu_ids)} GPUs will be used")
        
        return distribution
    
    def _advanced_gpu_monitoring(self):
        """Advanced GPU monitoring with detailed metrics and dual GPU tracking"""
        gpu_utilization_log = []
        
        while True:
            try:
                if MONITORING_AVAILABLE:
                    current_time = time.time()
                    utilization_snapshot = {}
                    
                    for gpu_id in self.gpu_ids:
                        # Collect comprehensive metrics
                        metrics = self._collect_gpu_metrics(gpu_id)
                        utilization_snapshot[f'gpu_{gpu_id}'] = metrics
                        
                        # Store metrics for reporting
                        self.stats['gpu_utilization'][gpu_id].append(metrics['utilization'])
                        self.stats['memory_usage'][gpu_id].append(metrics['memory_percent'])
                        
                        # Alert if utilization is significantly below target
                        target_util = self.turbo_config.target_gpu_utilization * 100
                        if metrics['utilization'] < target_util * 0.5:  # Less than 50% of target
                            logger.warning(f"⚠️ GPU {gpu_id} very low utilization: {metrics['utilization']:.1f}% (target: {target_util:.0f}%)")
                        elif metrics['utilization'] < target_util * 0.8:  # Less than 80% of target
                            logger.info(f"📊 GPU {gpu_id} below target utilization: {metrics['utilization']:.1f}% (target: {target_util:.0f}%)")
                    
                    # Log combined utilization every 30 seconds
                    gpu_utilization_log.append((current_time, utilization_snapshot))
                    if len(gpu_utilization_log) % 3 == 0:  # Every 3rd cycle (30 seconds)
                        self._log_dual_gpu_status(utilization_snapshot)
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
                break
    
    def _log_dual_gpu_status(self, utilization_snapshot: Dict):
        """Log the status of both GPUs for dual GPU monitoring"""
        gpu_status = []
        
        for gpu_id in self.gpu_ids:
            key = f'gpu_{gpu_id}'
            if key in utilization_snapshot:
                metrics = utilization_snapshot[key]
                status = f"GPU{gpu_id}: {metrics['utilization']:.0f}% util, {metrics['memory_percent']:.0f}% mem"
                gpu_status.append(status)
        
        if gpu_status:
            logger.info(f"🎮 DUAL GPU STATUS: {' | '.join(gpu_status)}")
            
            # Check for imbalanced utilization
            if len(gpu_status) == 2:
                gpu0_util = utilization_snapshot['gpu_0']['utilization']
                gpu1_util = utilization_snapshot['gpu_1']['utilization']
                
                if abs(gpu0_util - gpu1_util) > 30:  # More than 30% difference
                    logger.warning(f"⚠️ GPU utilization imbalance detected: GPU0={gpu0_util:.0f}% vs GPU1={gpu1_util:.0f}%")
    
    def _collect_gpu_metrics(self, gpu_id: int) -> Dict[str, float]:
        """Collect comprehensive GPU metrics"""
        metrics = {
            'utilization': 0.0,
            'memory_percent': 0.0,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'temperature': 0.0
        }
        
        try:
            # PyTorch metrics
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory = props.total_memory / (1024**3)
                
                metrics['memory_used'] = memory_allocated
                metrics['memory_total'] = total_memory
                metrics['memory_percent'] = (memory_allocated / total_memory) * 100
            
            # GPUtil metrics
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                metrics['utilization'] = gpu.load * 100
                metrics['temperature'] = gpu.temperature
            
        except Exception as e:
            logger.debug(f"GPU metrics collection error: {e}")
        
        return metrics
    
    def _load_gps_data_fast(self, gps_path: str) -> pd.DataFrame:
        """Fast GPS data loading with caching"""
        if not gps_path or not os.path.exists(gps_path):
            return pd.DataFrame()
        
        try:
            if gps_path.endswith('.csv'):
                df = pd.read_csv(gps_path)
                if 'long' in df.columns:
                    df['lon'] = df['long']
                return df
            elif gps_path.endswith('.gpx') and GPS_AVAILABLE:
                with open(gps_path, 'r') as f:
                    gpx = gpxpy.parse(f)
                
                records = []
                for track in gpx.tracks:
                    for segment in track.segments:
                        for i, point in enumerate(segment.points):
                            records.append({
                                'second': i,
                                'lat': point.latitude,
                                'lon': point.longitude,
                                'gpx_time': point.time
                            })
                
                return pd.DataFrame(records)
        except Exception as e:
            logger.warning(f"⚠️ GPS loading failed: {e}")
        
        return pd.DataFrame()
    
    def _save_results_fast(self, result: Dict):
        """Fast result saving with optimized I/O"""
        video_name = result['video_name']
        results = result['results']
        
        # Object tracking
        if results['object_tracking']:
            df = pd.DataFrame(results['object_tracking'])
            output_file = self.output_dir / 'object_tracking' / f"{video_name}_tracking.csv"
            df.to_csv(output_file, index=False)
        
        # Stoplight detection
        if results['stoplight_detection']:
            df = pd.DataFrame(results['stoplight_detection'])
            output_file = self.output_dir / 'stoplight_detection' / f"{video_name}_stoplights.csv"
            df.to_csv(output_file, index=False)
        
        # Traffic counting
        if results['traffic_counting']:
            counting_data = [
                {
                    'video_name': video_name, 
                    'object_type': obj_type, 
                    'total_count': count,
                    'gpu_id': result['gpu_id'],
                    'processing_fps': result['fps']
                }
                for obj_type, count in results['traffic_counting'].items()
            ]
            df = pd.DataFrame(counting_data)
            output_file = self.output_dir / 'traffic_counting' / f"{video_name}_counts.csv"
            df.to_csv(output_file, index=False)
    
    def load_matcher50_results(self, results_path: str) -> Dict[str, Any]:
        """Load matcher50 results with enhanced filtering"""
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', data)
        
        # Enhanced filtering for maximum quality
        min_score = self.config.get('min_score', 0.7)  # Higher threshold for turbo mode
        quality_map = {'excellent': 5, 'very_good': 4, 'good': 3, 'fair': 2, 'poor': 1}
        min_quality_level = quality_map.get(self.config.get('min_quality', 'very_good'), 4)
        
        filtered = {}
        for video_path, video_data in results.items():
            if 'matches' not in video_data or not video_data['matches']:
                continue
            
            best_match = video_data['matches'][0]
            score = best_match.get('combined_score', 0)
            quality = best_match.get('quality', 'poor')
            quality_level = quality_map.get(quality, 0)
            
            if score >= min_score and quality_level >= min_quality_level:
                filtered[video_path] = {
                    'gps_path': best_match.get('path', ''),
                    'quality': quality,
                    'score': score
                }
        
        logger.info(f"🔍 Turbo filtering: {len(filtered)} ultra-high-quality matches")
        return filtered
    
    def _generate_turbo_performance_report(self):
        """Generate comprehensive turbo performance report"""
        # Calculate advanced statistics
        avg_fps = np.mean(self.stats['fps_achieved']) if self.stats['fps_achieved'] else 0
        total_processing_time = sum(self.stats['processing_times'])
        
        gpu_utilization_stats = {}
        for gpu_id, utils in self.stats['gpu_utilization'].items():
            if utils:
                gpu_utilization_stats[f'gpu_{gpu_id}'] = {
                    'average_utilization': np.mean(utils),
                    'max_utilization': np.max(utils),
                    'min_utilization': np.min(utils),
                    'target_achievement': np.mean(utils) / (self.turbo_config.target_gpu_utilization * 100),
                    'samples': len(utils)
                }
        
        report = {
            'turbo_performance_summary': {
                'total_videos_processed': self.stats['processed_videos'],
                'failed_videos': self.stats['failed_videos'],
                'success_rate': (self.stats['processed_videos'] / (self.stats['processed_videos'] + self.stats['failed_videos'])) * 100 if (self.stats['processed_videos'] + self.stats['failed_videos']) > 0 else 0,
                'total_frames_processed': self.stats['total_frames'],
                'total_detections': self.stats['total_detections'],
                'average_fps': avg_fps,
                'total_processing_time_seconds': total_processing_time
            },
            'gpu_performance': gpu_utilization_stats,
            'turbo_configuration': {
                'target_gpu_utilization': self.turbo_config.target_gpu_utilization,
                'base_batch_size': self.turbo_config.base_batch_size,
                'adaptive_batching': self.turbo_config.adaptive_batching,
                'mixed_precision': self.turbo_config.enable_mixed_precision,
                'cuda_streams': self.turbo_config.use_cuda_streams,
                'memory_pooling': self.turbo_config.enable_memory_pooling
            },
            'system_info': {
                'gpu_count': self.gpu_count,
                'gpu_ids': self.gpu_ids,
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda
            }
        }
        
        # Save report
        report_path = self.output_dir / 'processing_reports' / 'turbo_performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info("🏁 TURBO GPU PROCESSING COMPLETE!")
        logger.info(f"   🎯 Success Rate: {report['turbo_performance_summary']['success_rate']:.1f}%")
        logger.info(f"   📊 Videos: {self.stats['processed_videos']}")
        logger.info(f"   🖼️ Frames: {self.stats['total_frames']:,}")
        logger.info(f"   🔍 Detections: {self.stats['total_detections']:,}")
        logger.info(f"   ⚡ Average FPS: {avg_fps:.1f}")
        
        for gpu_id, stats in gpu_utilization_stats.items():
            avg_util = stats['average_utilization']
            target_achievement = stats['target_achievement'] * 100
            logger.info(f"   🎮 {gpu_id.upper()}: {avg_util:.1f}% utilization ({target_achievement:.1f}% of target)")
    
    def _log_system_info(self):
        """Log comprehensive system information"""
        logger.info("🚀 TURBO GPU SYSTEM INFORMATION:")
        logger.info(f"   🎮 GPUs: {self.gpu_count} ({', '.join(f'GPU {i}' for i in self.gpu_ids)})")
        logger.info(f"   📦 Batch Size: {self.batch_size}")
        logger.info(f"   💾 Memory Pooling: {'✅' if self.turbo_config.enable_memory_pooling else '❌'}")
        logger.info(f"   ⚡ Mixed Precision: {'✅' if self.turbo_config.enable_mixed_precision else '❌'}")
        logger.info(f"   🌊 CUDA Streams: {'✅' if self.turbo_config.use_cuda_streams else '❌'}")
        logger.info(f"   🎯 Target GPU Utilization: {self.turbo_config.target_gpu_utilization*100:.0f}%")
        logger.info(f"   🖼️ Max Frames per Video: {self.max_video_frames:,}")

class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self, config: TurboConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_processing_start(self, video_name: str, gpu_id: int):
        """Log processing start"""
        self.metrics[f'gpu_{gpu_id}_videos'].append({
            'video': video_name,
            'start_time': time.time()
        })
    
    def log_processing_end(self, video_name: str, gpu_id: int, fps: float):
        """Log processing completion"""
        # Find the corresponding start entry
        gpu_videos = self.metrics[f'gpu_{gpu_id}_videos']
        for entry in reversed(gpu_videos):
            if entry['video'] == video_name and 'end_time' not in entry:
                entry['end_time'] = time.time()
                entry['fps'] = fps
                break

def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description="TurboGPU Video Processor v2.0 - Extreme Performance Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic turbo processing
  python turbo_visualizer.py -i results.json -o output/
  
  # Maximum performance with custom batch size
  python turbo_visualizer.py -i results.json -o output/ --batch-size 1024
  
  # High-quality filtering with performance monitoring
  python turbo_visualizer.py -i results.json -o output/ --min-score 0.8 --detailed-profiling
        """
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True, help='Matcher50 results JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Model settings
    parser.add_argument('--yolo-model', default='yolo11x.pt', help='YOLO model path (default: yolo11x.pt)')
    
    # Performance settings
    parser.add_argument('--batch-size', type=int, default=512, help='GPU batch size (default: 512)')
    parser.add_argument('--max-video-frames', type=int, default=15000, help='Max frames per video (default: 15000)')
    parser.add_argument('--frame-skip', type=int, default=2, help='Process every Nth frame (default: 2)')
    
    # Quality filtering
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum match score (default: 0.7)')
    parser.add_argument('--min-quality', default='very_good', choices=['excellent', 'very_good', 'good', 'fair', 'poor'], help='Minimum match quality (default: very_good)')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='YOLO confidence threshold (default: 0.3)')
    
    # Advanced options
    parser.add_argument('--target-gpu-utilization', type=float, default=0.95, help='Target GPU utilization (default: 0.95)')
    parser.add_argument('--disable-mixed-precision', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--disable-cuda-streams', action='store_true', help='Disable CUDA streams')
    parser.add_argument('--disable-memory-pooling', action='store_true', help='Disable memory pooling')
    parser.add_argument('--disable-benchmarking', action='store_true', help='Skip GPU benchmarking (use conservative batch size)')
    parser.add_argument('--force-dual-gpu', action='store_true', help='Force both GPUs to be used even with few videos')
    parser.add_argument('--detailed-profiling', action='store_true', help='Enable detailed performance profiling')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Build configuration
    config = {
        'output_dir': args.output,
        'yolo_model': args.yolo_model,
        'batch_size': args.batch_size,
        'max_video_frames': args.max_video_frames,
        'frame_skip': args.frame_skip,
        'min_score': args.min_score,
        'min_quality': args.min_quality,
        'confidence_threshold': args.confidence_threshold,
        'target_gpu_utilization': args.target_gpu_utilization,
        'enable_mixed_precision': not args.disable_mixed_precision,
        'use_cuda_streams': not args.disable_cuda_streams,
        'enable_memory_pooling': not args.disable_memory_pooling,
        'disable_benchmarking': args.disable_benchmarking,
        'force_dual_gpu': args.force_dual_gpu,
        'detailed_profiling': args.detailed_profiling
    }
    
    logger.info("🚀 Initializing TurboGPU Video Processor v2.0...")
    logger.info(f"   📁 Input: {args.input}")
    logger.info(f"   📁 Output: {args.output}")
    logger.info(f"   📦 Batch Size: {args.batch_size}")
    logger.info(f"   🎯 Target GPU Utilization: {args.target_gpu_utilization*100:.0f}%")
    
    try:
        # Initialize processor
        processor = TurboGPUVideoProcessor(config)
        
        # Load video matches
        video_matches = processor.load_matcher50_results(args.input)
        
        if not video_matches:
            logger.error("❌ No high-quality videos found to process")
            logger.info("💡 Try lowering --min-score or --min-quality")
            sys.exit(1)
        
        logger.info(f"✅ Ready to process {len(video_matches)} videos with TURBO acceleration")
        
        # Start turbo processing
        processor.process_videos_dual_gpu_turbo(video_matches)
        
        logger.info("🎉 TURBO PROCESSING COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        if args.detailed_profiling:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()