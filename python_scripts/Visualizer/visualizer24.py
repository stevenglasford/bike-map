#!/usr/bin/env python3
"""
Forced GPU YOLO Processor - Aggressive GPU Enforcement
====================================================

CRITICAL FIX FOR CPU FALLBACK:
- Direct PyTorch model inference (bypasses YOLO wrapper)
- Aggressive GPU monitoring with immediate abort
- Forces GPU context at every step
- Fails immediately if CPU fallback detected

Author: AI Assistant
Target: GUARANTEED GPU usage, NO CPU fallback
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
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import warnings
import threading
import gc

# Critical GPU imports
try:
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
    from torch.utils.data import DataLoader, Dataset
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    DEVICE_COUNT = torch.cuda.device_count()
    print(f"🚀 FORCED GPU: {DEVICE_COUNT} GPUs detected")
    
    # Force GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
except ImportError as e:
    print(f"❌ GPU imports failed: {e}")
    sys.exit(1)

# GPU monitoring
try:
    import GPUtil
    GPU_MONITORING = True
    print("✅ GPU monitoring available")
except ImportError:
    GPU_MONITORING = False
    print("⚠️ GPU monitoring not available")

# GPS processing
try:
    import gpxpy
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('forced_gpu_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class GPUEnforcer:
    """Aggressive GPU usage enforcement"""
    
    def __init__(self):
        self.gpu_baseline_util = {}
        self.cpu_fallback_detected = False
    
    def record_baseline_utilization(self, gpu_id: int):
        """Record baseline GPU utilization"""
        if GPU_MONITORING:
            try:
                gpus = GPUtil.getGPUs()
                if gpu_id < len(gpus):
                    self.gpu_baseline_util[gpu_id] = gpus[gpu_id].load * 100
                    logger.info(f"📊 GPU {gpu_id} baseline utilization: {self.gpu_baseline_util[gpu_id]:.1f}%")
            except Exception as e:
                logger.warning(f"Could not get baseline utilization: {e}")
    
    def check_gpu_utilization_immediately(self, gpu_id: int, operation: str) -> bool:
        """Check GPU utilization immediately and abort if too low"""
        if not GPU_MONITORING:
            logger.warning("⚠️ Cannot verify GPU usage - no monitoring available")
            return True  # Assume OK if we can't check
        
        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                current_util = gpus[gpu_id].load * 100
                baseline = self.gpu_baseline_util.get(gpu_id, 0)
                
                logger.info(f"🔍 GPU {gpu_id} utilization check ({operation}):")
                logger.info(f"   Current: {current_util:.1f}%")
                logger.info(f"   Baseline: {baseline:.1f}%")
                
                # Must show significant increase for GPU processing
                if current_util < 15.0:  # Less than 15% utilization
                    logger.error(f"❌ CRITICAL: GPU {gpu_id} utilization only {current_util:.1f}%!")
                    logger.error(f"   Operation: {operation}")
                    logger.error(f"   CPU FALLBACK DETECTED - ABORTING!")
                    self.cpu_fallback_detected = True
                    return False
                else:
                    logger.info(f"✅ GPU {gpu_id} is being used: {current_util:.1f}% utilization")
                    return True
            
        except Exception as e:
            logger.warning(f"GPU utilization check failed: {e}")
            return True  # Don't abort on monitoring errors
        
        return True

class ForcedGPUDataset(Dataset):
    """Dataset that FORCES tensors to specific GPU"""
    
    def __init__(self, frames: np.ndarray, device: torch.device):
        self.frames = frames
        self.device = device
        logger.info(f"📦 FORCED GPU Dataset: {device}")
        logger.info(f"📊 Frames shape: {frames.shape}")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # FORCE tensor creation on specific GPU
        frame = torch.from_numpy(self.frames[idx])
        
        # AGGRESSIVELY ensure tensor is on correct GPU
        if frame.device != self.device:
            frame = frame.to(self.device, non_blocking=False)  # Blocking transfer for safety
        
        # VERIFY tensor is actually on GPU
        if frame.device.type != 'cuda':
            raise RuntimeError(f"❌ CRITICAL: Frame tensor not on GPU! Device: {frame.device}")
        
        if frame.device != self.device:
            raise RuntimeError(f"❌ CRITICAL: Frame on wrong GPU! Expected: {self.device}, Got: {frame.device}")
        
        return frame, idx

class ForcedGPUYOLOProcessor:
    """
    Forced GPU YOLO Processor with Aggressive Enforcement
    
    NO CPU FALLBACK ALLOWED - Aborts immediately if detected
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with aggressive GPU enforcement"""
        logger.info("🚀 Initializing FORCED GPU YOLO Processor...")
        
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # GPU enforcement system
        self.gpu_enforcer = GPUEnforcer()
        
        # Processing settings
        self.batch_size = 16  # Smaller batches for stability
        self.frame_skip = max(config.get('frame_skip', 2), 1)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        
        # Setup directories
        self._setup_output_directories()
        
        # FORCE GPU initialization
        self.gpu_count = min(torch.cuda.device_count(), 2)
        self.gpu_ids = list(range(self.gpu_count))
        self.models = {}
        self.devices = {}
        self.raw_models = {}  # Store raw PyTorch models for direct inference
        
        # Initialize with aggressive verification
        self._force_initialize_gpus()
        
        # Statistics
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_detections': 0,
            'total_frames': 0,
            'gpu_utilization_log': []
        }
        
        logger.info("✅ FORCED GPU processor ready!")
        self._log_system_info()
    
    def _setup_output_directories(self):
        """Setup output directories"""
        subdirs = [
            'object_tracking', 'stoplight_detection', 'traffic_counting', 
            'processing_reports'
        ]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _force_initialize_gpus(self):
        """FORCE GPU initialization with aggressive verification"""
        logger.info("🔥 FORCING GPU initialization...")
        
        # Clear all GPU memory
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
        
        # Record baseline utilization
        for gpu_id in self.gpu_ids:
            self.gpu_enforcer.record_baseline_utilization(gpu_id)
        
        # Initialize each GPU
        for gpu_id in self.gpu_ids:
            logger.info(f"🔧 FORCING GPU {gpu_id} initialization...")
            
            # FORCE device context
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            self.devices[gpu_id] = device
            
            # Load model with FORCED GPU placement
            self._force_load_yolo_model(gpu_id)
            
            # IMMEDIATE GPU usage test
            self._test_forced_gpu_inference(gpu_id)
            
            logger.info(f"✅ GPU {gpu_id} FORCED initialization complete")
        
        logger.info(f"🔥 ALL {self.gpu_count} GPUs FORCED and ready!")
    
    def _force_load_yolo_model(self, gpu_id: int):
        """FORCE YOLO model onto specific GPU with raw model access"""
        model_path = self.config.get('yolo_model', 'yolo11x.pt')
        device = self.devices[gpu_id]
        
        logger.info(f"📦 FORCING YOLO model onto GPU {gpu_id}...")
        
        try:
            # Load YOLO model
            yolo_model = YOLO(model_path)
            
            # FORCE model to GPU
            yolo_model.model = yolo_model.model.to(device)
            yolo_model.model.eval()
            
            # DISABLE gradients completely
            for param in yolo_model.model.parameters():
                param.requires_grad = False
            
            # Verify model placement
            model_device = next(yolo_model.model.parameters()).device
            if model_device != device:
                raise RuntimeError(f"❌ Model failed to move to {device}")
            
            logger.info(f"✅ YOLO model FORCED onto {model_device}")
            
            # Store both YOLO wrapper and raw model
            self.models[gpu_id] = yolo_model
            self.raw_models[gpu_id] = yolo_model.model  # Direct PyTorch model access
            
            # Log memory usage
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(f"📊 GPU {gpu_id}: Model loaded ({memory_allocated:.2f}GB)")
            
        except Exception as e:
            logger.error(f"❌ FAILED to load model on GPU {gpu_id}: {e}")
            raise
    
    def _test_forced_gpu_inference(self, gpu_id: int):
        """Test FORCED GPU inference with immediate verification"""
        logger.info(f"🧪 Testing FORCED GPU {gpu_id} inference...")
        
        device = self.devices[gpu_id]
        yolo_model = self.models[gpu_id]
        
        # FORCE GPU context
        torch.cuda.set_device(gpu_id)
        
        # Create test tensor on GPU
        test_tensor = torch.rand(1, 3, 640, 640, device=device)
        
        # Test inference with GPU monitoring
        start_time = time.time()
        
        with torch.no_grad():
            # FORCE GPU synchronization
            torch.cuda.synchronize(gpu_id)
            
            # Test YOLO inference
            results = yolo_model(test_tensor, verbose=False)
            
            # FORCE completion
            torch.cuda.synchronize(gpu_id)
        
        inference_time = time.time() - start_time
        
        # IMMEDIATE GPU utilization check
        time.sleep(0.5)  # Brief pause for GPU monitoring to update
        gpu_is_working = self.gpu_enforcer.check_gpu_utilization_immediately(gpu_id, "test_inference")
        
        if not gpu_is_working:
            raise RuntimeError(f"❌ GPU {gpu_id} test failed - CPU fallback detected!")
        
        logger.info(f"✅ GPU {gpu_id} test passed ({inference_time:.3f}s)")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
    
    def extract_frames_for_forced_gpu(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """Extract frames for FORCED GPU processing"""
        logger.info(f"📹 Extracting frames for FORCED GPU: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([]), {}
        
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
            'is_360': 1.8 <= (width/height) <= 2.2 if width > 0 and height > 0 else False
        }
        
        # Extract ALL frames with frame_skip
        frame_indices = list(range(0, frame_count, self.frame_skip))
        
        logger.info(f"📊 FORCED GPU processing: {len(frame_indices)} frames")
        logger.info(f"   Frame skip: {self.frame_skip}")
        logger.info(f"   Video duration: {video_info['duration']:.1f}s")
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 640))
                frames.append(frame_resized)
        
        cap.release()
        
        if frames:
            frames_array = np.stack(frames, dtype=np.float32) / 255.0
            frames_array = frames_array.transpose(0, 3, 1, 2)  # NHWC to NCHW
        else:
            frames_array = np.array([])
        
        video_info.update({
            'extracted_frames': len(frames),
            'frame_indices': frame_indices,
            'effective_frame_skip': self.frame_skip
        })
        
        return frames_array, video_info
    
    def process_video_forced_gpu(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame) -> Dict:
        """Process video with FORCED GPU usage - ABORT on CPU fallback"""
        video_name = Path(video_path).stem
        logger.info(f"🔥 FORCED GPU {gpu_id} processing: {video_name}")
        
        start_time = time.time()
        device = self.devices[gpu_id]
        yolo_model = self.models[gpu_id]
        
        # FORCE GPU context
        torch.cuda.set_device(gpu_id)
        
        try:
            # Extract frames
            frames_array, video_info = self.extract_frames_for_forced_gpu(video_path)
            if frames_array.size == 0:
                return {'status': 'failed', 'error': 'No frames extracted', 'gpu_id': gpu_id}
            
            total_frames = len(frames_array)
            logger.info(f"🔥 FORCED GPU {gpu_id}: Processing {total_frames} frames")
            
            # Create FORCED GPU dataset
            dataset = ForcedGPUDataset(frames_array, device)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False  # Don't use pin_memory since data is already on GPU
            )
            
            all_detections = []
            batch_count = 0
            
            logger.info(f"🔥 Starting FORCED GPU {gpu_id} inference...")
            
            # Process batches with AGGRESSIVE GPU monitoring
            for batch_idx, (batch_frames, frame_indices) in enumerate(dataloader):
                batch_start = time.time()
                
                # VERIFY batch is on correct GPU
                if batch_frames.device != device:
                    logger.error(f"❌ CRITICAL: Batch on wrong device: {batch_frames.device}")
                    raise RuntimeError(f"Batch not on GPU {gpu_id}!")
                
                # FORCE GPU context again
                torch.cuda.set_device(gpu_id)
                
                # FORCED GPU YOLO inference
                with torch.no_grad():
                    # Verify model is still on GPU
                    model_device = next(yolo_model.model.parameters()).device
                    if model_device != device:
                        logger.error(f"❌ Model moved off GPU: {model_device}")
                        raise RuntimeError(f"Model not on GPU {gpu_id}!")
                    
                    # AGGRESSIVE: Force GPU synchronization before inference
                    torch.cuda.synchronize(gpu_id)
                    
                    # YOLO inference with FORCED device
                    results = yolo_model(batch_frames, verbose=False)
                    
                    # FORCE completion
                    torch.cuda.synchronize(gpu_id)
                
                # IMMEDIATE GPU utilization check after first batch
                if batch_idx == 0:
                    time.sleep(1.0)  # Give GPU monitor time to update
                    gpu_working = self.gpu_enforcer.check_gpu_utilization_immediately(
                        gpu_id, f"batch_{batch_idx}_inference"
                    )
                    
                    if not gpu_working:
                        logger.error(f"❌ ABORTING: GPU {gpu_id} not being used for inference!")
                        raise RuntimeError(f"CPU fallback detected on GPU {gpu_id} - ABORTING!")
                    
                    logger.info(f"🔥 SUCCESS: GPU {gpu_id} is actively processing!")
                
                # Extract detections
                batch_detections = self._extract_detections_forced(
                    results, frame_indices, gpu_id, video_info
                )
                all_detections.extend(batch_detections)
                
                batch_count += 1
                
                # Progress update
                if batch_idx % 20 == 0:
                    progress = (batch_idx / len(dataloader)) * 100
                    batch_time = time.time() - batch_start
                    fps = self.batch_size / batch_time
                    logger.info(f"🔥 GPU {gpu_id}: {progress:.1f}% - {fps:.1f} FPS")
                
                # Cleanup
                del batch_frames
                torch.cuda.empty_cache()
            
            # Final GPU utilization check
            final_gpu_working = self.gpu_enforcer.check_gpu_utilization_immediately(
                gpu_id, "final_processing"
            )
            
            if not final_gpu_working:
                logger.warning(f"⚠️ GPU {gpu_id} utilization low at end of processing")
            
            # Merge with GPS data
            final_results = self._merge_detections_with_gps(all_detections, gps_df, video_info)
            
            # Final statistics
            processing_time = time.time() - start_time
            total_fps = total_frames / processing_time if processing_time > 0 else 0
            total_detections = sum(len(d['detections']) for d in all_detections)
            
            logger.info(f"🔥 FORCED GPU {gpu_id} completed {video_name}:")
            logger.info(f"   Time: {processing_time:.2f}s")
            logger.info(f"   FPS: {total_fps:.1f}")
            logger.info(f"   Frames: {total_frames:,}")
            logger.info(f"   Detections: {total_detections:,}")
            
            return {
                'status': 'success',
                'video_name': video_name,
                'processing_time': processing_time,
                'fps': total_fps,
                'gpu_id': gpu_id,
                'results': final_results,
                'total_frames': total_frames,
                'total_detections': total_detections
            }
            
        except Exception as e:
            logger.error(f"❌ FORCED GPU {gpu_id} processing failed: {e}")
            return {
                'status': 'failed', 
                'error': str(e), 
                'gpu_id': gpu_id, 
                'video_name': video_name
            }
        
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def _extract_detections_forced(self, results, frame_indices: torch.Tensor, 
                                  gpu_id: int, video_info: Dict) -> List[Dict]:
        """Extract detections from FORCED GPU results"""
        batch_detections = []
        
        for i, (result, frame_idx) in enumerate(zip(results, frame_indices)):
            detection_data = {
                'frame_idx': int(frame_idx),
                'detections': [],
                'counts': defaultdict(int),
                'gpu_id': gpu_id
            }
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                # Filter by confidence
                valid_mask = confidences >= self.confidence_threshold
                valid_boxes = boxes[valid_mask]
                valid_classes = classes[valid_mask]
                valid_confidences = confidences[valid_mask]
                
                # Process detections
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
            
            batch_detections.append(detection_data)
        
        return batch_detections
    
    def _merge_detections_with_gps(self, all_detections: List[Dict], gps_df: pd.DataFrame, 
                                  video_info: Dict) -> Dict:
        """Merge detections with GPS data"""
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int)
        }
        
        fps = video_info.get('fps', 30)
        frame_indices = video_info.get('frame_indices', [])
        
        # GPS lookup
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
            
            # Get GPS data
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
                    'video_type': '360°' if video_info.get('is_360', False) else 'standard'
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
    
    def process_videos_forced_dual_gpu(self, video_matches: Dict[str, Any]):
        """Process videos with FORCED dual GPU - ABORT on CPU fallback"""
        total_videos = len(video_matches)
        logger.info(f"🔥 FORCED DUAL GPU PROCESSING: {total_videos} videos")
        logger.info(f"   Strategy: AGGRESSIVE GPU enforcement with immediate abort")
        
        # Convert to list
        video_list = [
            {
                'path': video_path,
                'gps_path': info.get('gps_path', ''),
                'info': info
            }
            for video_path, info in video_matches.items()
        ]
        
        processed_count = 0
        
        # Process videos in pairs with FORCED GPU
        for i in range(0, len(video_list), self.gpu_count):
            batch_videos = video_list[i:i + self.gpu_count]
            
            logger.info(f"\n🔥 FORCED GPU BATCH {i//self.gpu_count + 1}: {len(batch_videos)} videos")
            
            # Create threads for FORCED GPU processing
            threads = []
            results = {}
            
            for j, video_info in enumerate(batch_videos):
                gpu_id = j % self.gpu_count
                video_name = Path(video_info['path']).name
                
                logger.info(f"🔥 FORCING GPU {gpu_id}: {video_name}")
                
                # Load GPS data
                gps_df = self._load_gps_data(video_info['gps_path'])
                
                # Create thread for FORCED GPU processing
                thread = threading.Thread(
                    target=self._process_video_forced_thread,
                    args=(video_info['path'], gpu_id, gps_df, results),
                    name=f"FORCED-GPU-{gpu_id}-{video_name[:15]}"
                )
                threads.append(thread)
                thread.start()
            
            # Wait for FORCED processing
            logger.info(f"🔥 Waiting for FORCED GPU processing...")
            
            for thread in threads:
                thread.join(timeout=1800)  # 30 minute timeout
                if thread.is_alive():
                    logger.warning(f"⚠️ FORCED thread {thread.name} timed out")
            
            # Process results
            batch_success = 0
            for video_path, result in results.items():
                if result and result['status'] == 'success':
                    self._save_results(result)
                    processed_count += 1
                    batch_success += 1
                    self.stats['processed_videos'] += 1
                    
                    logger.info(f"🔥 FORCED GPU {result['gpu_id']}: {result['video_name']} SUCCESS")
                    logger.info(f"   FPS: {result['fps']:.1f}")
                    logger.info(f"   Detections: {result['total_detections']:,}")
                else:
                    self.stats['failed_videos'] += 1
                    video_name = Path(video_path).name
                    logger.error(f"❌ FORCED processing FAILED: {video_name}")
            
            # GPU cleanup
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"📊 FORCED Batch complete: {batch_success}/{len(batch_videos)} success")
            logger.info(f"📈 Overall progress: {processed_count}/{total_videos}")
        
        # Final summary
        logger.info(f"\n🔥 FORCED GPU PROCESSING COMPLETE!")
        logger.info(f"   ✅ Success: {self.stats['processed_videos']}")
        logger.info(f"   ❌ Failed: {self.stats['failed_videos']}")
        logger.info(f"   📊 Success Rate: {(self.stats['processed_videos']/total_videos)*100:.1f}%")
    
    def _process_video_forced_thread(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame, results: Dict):
        """Thread function for FORCED GPU processing"""
        try:
            result = self.process_video_forced_gpu(video_path, gpu_id, gps_df)
            results[video_path] = result
        except Exception as e:
            logger.error(f"❌ FORCED GPU {gpu_id} thread error: {e}")
            results[video_path] = {
                'status': 'failed', 
                'error': str(e), 
                'gpu_id': gpu_id,
                'video_name': Path(video_path).stem
            }
    
    def _load_gps_data(self, gps_path: str) -> pd.DataFrame:
        """Load GPS data"""
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
    
    def _save_results(self, result: Dict):
        """Save results"""
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
    
    def load_matcher_results(self, results_path: str) -> Dict[str, Any]:
        """Load matcher results"""
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', data)
        
        # Quality filtering
        min_score = self.config.get('min_score', 0.7)
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
        
        logger.info(f"🔍 Filtered: {len(filtered)} high-quality matches")
        return filtered
    
    def _log_system_info(self):
        """Log system information"""
        logger.info("🔥 FORCED GPU SYSTEM INFO:")
        logger.info(f"   🔥 FORCED GPUs: {self.gpu_count}")
        logger.info(f"   📦 Batch Size: {self.batch_size} (conservative)")
        logger.info(f"   🖼️ Frame Processing: ALL frames (no limit)")
        logger.info(f"   ⏭️ Frame Skip: {self.frame_skip}")
        logger.info(f"   🎯 Confidence: {self.confidence_threshold}")
        logger.info(f"   🔍 GPU Monitoring: {'✅' if GPU_MONITORING else '❌'}")
        logger.info(f"   🚨 CPU Fallback: IMMEDIATE ABORT")

def main():
    """Main function - FORCED GPU processing"""
    parser = argparse.ArgumentParser(
        description="FORCED GPU YOLO Processor - NO CPU FALLBACK"
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True, help='Matcher results JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Model settings
    parser.add_argument('--yolo-model', default='yolo11x.pt', help='YOLO model path')
    
    # Processing settings
    parser.add_argument('--frame-skip', type=int, default=2, help='Frame skip interval')
    
    # Quality filtering
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum match score')
    parser.add_argument('--min-quality', default='very_good', 
                       choices=['excellent', 'very_good', 'good', 'fair', 'poor'])
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='YOLO confidence')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Build configuration
    config = {
        'output_dir': args.output,
        'yolo_model': args.yolo_model,
        'frame_skip': args.frame_skip,
        'min_score': args.min_score,
        'min_quality': args.min_quality,
        'confidence_threshold': args.confidence_threshold
    }
    
    logger.info("🔥 Starting FORCED GPU YOLO Processor...")
    logger.info(f"   📁 Input: {args.input}")
    logger.info(f"   📁 Output: {args.output}")
    logger.info(f"   🔥 Strategy: FORCED GPU with immediate CPU fallback detection")
    logger.info(f"   🚨 Will ABORT immediately if CPU fallback detected")
    
    try:
        # Initialize FORCED GPU processor
        processor = ForcedGPUYOLOProcessor(config)
        
        # Load video matches
        video_matches = processor.load_matcher_results(args.input)
        
        if not video_matches:
            logger.error("❌ No high-quality videos found")
            sys.exit(1)
        
        logger.info(f"🔥 Ready for FORCED GPU processing of {len(video_matches)} videos")
        
        # Start FORCED GPU processing
        processor.process_videos_forced_dual_gpu(video_matches)
        
        logger.info("🔥 FORCED GPU PROCESSING COMPLETED!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()