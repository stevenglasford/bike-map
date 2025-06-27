#!/usr/bin/env python3
"""
Simple Dual GPU Video Processor - One Video Per GPU
==================================================

SIMPLIFIED APPROACH:
- Process exactly ONE video per GPU simultaneously
- No complex threading or queue management
- Conservative memory settings
- Clear, straightforward logic
- Robust error handling

Author: AI Assistant
Target: Reliable dual RTX 5060 Ti processing
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
    print(f"🚀 CUDA: {DEVICE_COUNT} GPUs detected")
    
    # Conservative optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
except ImportError as e:
    print(f"❌ GPU imports failed: {e}")
    sys.exit(1)

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
        logging.FileHandler('simple_gpu_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleVideoDataset(Dataset):
    """Simple video dataset"""
    
    def __init__(self, frames: np.ndarray):
        self.frames = frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx], idx

class SimpleGPUVideoProcessor:
    """
    Simple Dual GPU Video Processor
    
    Processes exactly ONE video per GPU simultaneously
    No complex threading, queues, or monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simple processor"""
        logger.info("🚀 Initializing Simple Dual GPU Video Processor...")
        
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Simple settings
        self.batch_size = 16  # VERY conservative
        self.max_video_frames = 5000  # Conservative limit
        self.frame_skip = max(config.get('frame_skip', 2), 2)
        
        # Setup directories
        self._setup_output_directories()
        
        # Initialize GPUs
        self.gpu_count = min(torch.cuda.device_count(), 2)
        self.gpu_ids = list(range(self.gpu_count))
        self.models = {}
        
        self._initialize_gpus()
        
        # Statistics
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_detections': 0
        }
        
        logger.info("✅ Simple processor initialized successfully!")
        logger.info(f"   🎮 GPUs: {self.gpu_count}")
        logger.info(f"   📦 Batch Size: {self.batch_size}")
        logger.info(f"   🖼️ Max Frames: {self.max_video_frames}")
    
    def _setup_output_directories(self):
        """Setup output directories"""
        subdirs = ['object_tracking', 'stoplight_detection', 'traffic_counting', 'processing_reports']
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _initialize_gpus(self):
        """Initialize GPUs with conservative settings"""
        logger.info("🎮 Initializing GPUs...")
        
        for gpu_id in self.gpu_ids:
            logger.info(f"🔧 Initializing GPU {gpu_id}...")
            
            # Set device and clear memory
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            # Get GPU info
            props = torch.cuda.get_device_properties(gpu_id)
            total_memory = props.total_memory / (1024**3)
            logger.info(f"   GPU {gpu_id}: {props.name} ({total_memory:.1f}GB)")
            
            # Load model
            self._load_model_on_gpu(gpu_id)
        
        logger.info(f"✅ All {self.gpu_count} GPUs initialized")
    
    def _load_model_on_gpu(self, gpu_id: int):
        """Load YOLO model on specific GPU"""
        model_path = self.config.get('yolo_model', 'yolo11x.pt')
        device = torch.device(f'cuda:{gpu_id}')
        
        logger.info(f"📦 Loading YOLO model on GPU {gpu_id}...")
        
        try:
            # Load YOLO model
            model = YOLO(model_path)
            model.model = model.model.to(device)
            model.model.eval()
            
            # Disable gradients
            for param in model.model.parameters():
                param.requires_grad = False
            
            self.models[gpu_id] = model
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(f"   ✅ GPU {gpu_id}: Model loaded ({memory_allocated:.2f}GB used)")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed on GPU {gpu_id}: {e}")
            raise
    
    def extract_frames_simple(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """Simple frame extraction"""
        logger.info(f"📹 Extracting frames: {Path(video_path).name}")
        
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
            'duration': frame_count / fps if fps > 0 else 0
        }
        
        # Conservative frame selection
        max_frames = min(self.max_video_frames, frame_count)
        frame_indices = list(range(0, frame_count, self.frame_skip))[:max_frames]
        
        logger.info(f"📊 Extracting {len(frame_indices)} frames from {frame_count} total")
        
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
        
        video_info['extracted_frames'] = len(frames)
        video_info['frame_indices'] = frame_indices
        
        return frames_array, video_info
    
    def process_video_on_gpu(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame) -> Dict:
        """Process single video on specific GPU"""
        video_name = Path(video_path).stem
        logger.info(f"🚀 GPU {gpu_id} processing: {video_name}")
        
        start_time = time.time()
        device = torch.device(f'cuda:{gpu_id}')
        model = self.models[gpu_id]
        
        try:
            # Set GPU context
            torch.cuda.set_device(gpu_id)
            
            # Extract frames
            frames_array, video_info = self.extract_frames_simple(video_path)
            if frames_array.size == 0:
                return {'status': 'failed', 'error': 'No frames extracted', 'gpu_id': gpu_id}
            
            total_frames = len(frames_array)
            logger.info(f"🎮 GPU {gpu_id}: Processing {total_frames} frames")
            
            # Create dataset and dataloader
            dataset = SimpleVideoDataset(frames_array)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Process batches
            all_detections = []
            confidence_threshold = self.config.get('confidence_threshold', 0.3)
            
            for batch_idx, (batch_frames, frame_indices) in enumerate(dataloader):
                # Move to GPU
                batch_tensor = batch_frames.to(device)
                
                # Process with YOLO
                with torch.no_grad():
                    results = model(batch_tensor, verbose=False)
                
                # Extract detections
                for i, (result, frame_idx) in enumerate(zip(results, frame_indices)):
                    detection_data = {
                        'frame_idx': int(frame_idx),
                        'detections': [],
                        'counts': defaultdict(int)
                    }
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        # Filter by confidence
                        valid_mask = confidences >= confidence_threshold
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
                    
                    all_detections.append(detection_data)
                
                # Clean up after each batch
                del batch_tensor
                torch.cuda.empty_cache()
                
                # Progress update
                if batch_idx % 10 == 0:
                    progress = (batch_idx / len(dataloader)) * 100
                    logger.info(f"🎮 GPU {gpu_id}: {progress:.1f}% complete")
            
            # Merge with GPS data
            final_results = self._merge_detections_with_gps(all_detections, gps_df, video_info)
            
            processing_time = time.time() - start_time
            total_fps = total_frames / processing_time if processing_time > 0 else 0
            
            logger.info(f"✅ GPU {gpu_id} completed {video_name}:")
            logger.info(f"   Time: {processing_time:.2f}s")
            logger.info(f"   FPS: {total_fps:.1f}")
            logger.info(f"   Detections: {sum(len(d['detections']) for d in all_detections)}")
            
            return {
                'status': 'success',
                'video_name': video_name,
                'processing_time': processing_time,
                'fps': total_fps,
                'gpu_id': gpu_id,
                'results': final_results
            }
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} error processing {video_path}: {e}")
            return {'status': 'failed', 'error': str(e), 'gpu_id': gpu_id, 'video_name': video_name}
        
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
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
                    'gpu_id': detection_data.get('gpu_id', -1)
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
    
    def process_videos_simple_dual_gpu(self, video_matches: Dict[str, Any]):
        """Process videos using simple dual GPU approach - ONE video per GPU"""
        total_videos = len(video_matches)
        logger.info(f"🚀 SIMPLE DUAL GPU: Processing {total_videos} videos")
        logger.info(f"   Strategy: ONE video per GPU simultaneously")
        
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
        
        # Process videos in pairs (one per GPU)
        for i in range(0, len(video_list), self.gpu_count):
            batch_videos = video_list[i:i + self.gpu_count]
            
            logger.info(f"\n🔄 Processing batch {i//self.gpu_count + 1}: {len(batch_videos)} videos")
            
            # Create threads for simultaneous processing
            threads = []
            results = {}
            
            for j, video_info in enumerate(batch_videos):
                gpu_id = j % self.gpu_count
                video_name = Path(video_info['path']).name
                
                logger.info(f"🎮 GPU {gpu_id} will process: {video_name}")
                
                # Load GPS data
                gps_df = self._load_gps_data(video_info['gps_path'])
                
                # Create thread for this GPU
                thread = threading.Thread(
                    target=self._process_video_thread,
                    args=(video_info['path'], gpu_id, gps_df, results),
                    name=f"GPU-{gpu_id}-{video_name[:10]}"
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            logger.info(f"⏳ Waiting for {len(threads)} GPUs to complete...")
            
            for thread in threads:
                thread.join(timeout=600)  # 10 minute timeout per video
                if thread.is_alive():
                    logger.warning(f"⚠️ Thread {thread.name} timed out")
            
            # Process results
            for video_path, result in results.items():
                if result and result['status'] == 'success':
                    self._save_results(result)
                    processed_count += 1
                    self.stats['processed_videos'] += 1
                    logger.info(f"✅ Saved results for {result['video_name']}")
                else:
                    self.stats['failed_videos'] += 1
                    logger.error(f"❌ Failed to process {Path(video_path).name}")
            
            # Clean up between batches
            for gpu_id in self.gpu_ids:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            
            gc.collect()
            
            # Progress update
            logger.info(f"📊 Progress: {processed_count}/{total_videos} videos completed")
        
        # Final summary
        logger.info(f"\n🏁 PROCESSING COMPLETE!")
        logger.info(f"   ✅ Processed: {self.stats['processed_videos']}")
        logger.info(f"   ❌ Failed: {self.stats['failed_videos']}")
        logger.info(f"   📊 Success Rate: {(self.stats['processed_videos']/total_videos)*100:.1f}%")
    
    def _process_video_thread(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame, results: Dict):
        """Thread function to process a single video on a specific GPU"""
        try:
            result = self.process_video_on_gpu(video_path, gpu_id, gps_df)
            results[video_path] = result
        except Exception as e:
            logger.error(f"❌ Thread error for GPU {gpu_id}: {e}")
            results[video_path] = {'status': 'failed', 'error': str(e), 'gpu_id': gpu_id}
    
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
        """Save processing results"""
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
                    'gpu_id': result['gpu_id']
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
        
        # Simple filtering
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

def main():
    """Main function - Simple dual GPU processing"""
    parser = argparse.ArgumentParser(
        description="Simple Dual GPU Video Processor - One Video Per GPU"
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True, help='Matcher results JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Model settings
    parser.add_argument('--yolo-model', default='yolo11x.pt', help='YOLO model path')
    
    # Simple settings
    parser.add_argument('--frame-skip', type=int, default=2, help='Frame skip interval')
    
    # Quality filtering
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum match score')
    parser.add_argument('--min-quality', default='very_good', 
                       choices=['excellent', 'very_good', 'good', 'fair', 'poor'])
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='YOLO confidence threshold')
    
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
    
    logger.info("🚀 Starting Simple Dual GPU Video Processor...")
    logger.info(f"   📁 Input: {args.input}")
    logger.info(f"   📁 Output: {args.output}")
    logger.info(f"   🎯 Strategy: ONE video per GPU")
    
    try:
        # Initialize processor
        processor = SimpleGPUVideoProcessor(config)
        
        # Load video matches
        video_matches = processor.load_matcher_results(args.input)
        
        if not video_matches:
            logger.error("❌ No high-quality videos found")
            sys.exit(1)
        
        logger.info(f"✅ Ready to process {len(video_matches)} videos")
        
        # Process videos
        processor.process_videos_simple_dual_gpu(video_matches)
        
        logger.info("🎉 SIMPLE PROCESSING COMPLETED!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()