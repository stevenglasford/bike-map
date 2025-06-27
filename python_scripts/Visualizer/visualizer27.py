#!/usr/bin/env python3
"""
Simplified GPU YOLO Processor - Based on Working process*.py patterns
====================================================================

Uses the proven simple approach from your working process_groups_yolo.py files
- Simple YOLO model initialization
- Reliable GPU usage without aggressive monitoring
- Direct model inference
- No false CPU fallback detection

Author: AI Assistant
Target: Reliable GPU usage with proven patterns
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

# Simple GPU imports (like your working scripts)
try:
    import torch
    from ultralytics import YOLO
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"🚀 GPU available: {torch.cuda.device_count()} devices")
    
except ImportError as e:
    print(f"❌ GPU imports failed: {e}")
    sys.exit(1)

# GPS processing
try:
    import gpxpy
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpu_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class SimplifiedGPUYOLOProcessor:
    """
    Simplified GPU YOLO Processor using proven patterns from process*.py
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with simple GPU setup (like your working scripts)"""
        logger.info("🚀 Initializing Simplified GPU YOLO Processor...")
        
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing settings (like your working scripts)
        self.frame_skip = max(config.get('frame_skip', 2), 1)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        
        # Setup directories
        self._setup_output_directories()
        
        # Simple GPU initialization (like your working scripts)
        self.gpu_count = min(torch.cuda.device_count(), 2)
        self.models = {}
        self.devices = {}
        
        # Initialize with simple approach
        self._initialize_gpus_simple()
        
        # Statistics
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_detections': 0,
            'total_frames': 0
        }
        
        logger.info("✅ Simplified GPU processor ready!")
        self._log_system_info()
    
    def _setup_output_directories(self):
        """Setup output directories"""
        subdirs = [
            'object_tracking', 'stoplight_detection', 'traffic_counting', 
            'processing_reports'
        ]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _initialize_gpus_simple(self):
        """Simple GPU initialization (like your working process*.py scripts)"""
        logger.info("🔧 Simple GPU initialization...")
        
        # Initialize each GPU with simple approach
        for gpu_id in range(self.gpu_count):
            logger.info(f"🔧 Initializing GPU {gpu_id}...")
            
            # Simple model loading (exactly like your working scripts)
            self._load_yolo_model_simple(gpu_id)
            
            logger.info(f"✅ GPU {gpu_id} initialized successfully")
        
        logger.info(f"🔧 All {self.gpu_count} GPUs ready!")
    
    def _load_yolo_model_simple(self, gpu_id: int):
        """Simple YOLO model loading (exactly like your working process*.py scripts)"""
        model_path = self.config.get('yolo_model', 'yolo11x.pt')
        
        logger.info(f"📦 Loading YOLO model on GPU {gpu_id}...")
        
        try:
            # EXACTLY like your working scripts - use string device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(model_path).to(device)
            
            # Set the specific GPU for this model
            torch.cuda.set_device(gpu_id)
            
            logger.info(f"✅ YOLO model loaded on GPU {gpu_id}")
            
            # Store model and device
            self.models[gpu_id] = model
            self.devices[gpu_id] = device
            
            # Log memory usage
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(f"📊 GPU {gpu_id}: Model loaded ({memory_allocated:.2f}GB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model on GPU {gpu_id}: {e}")
            raise
    
    def extract_frames_simple(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """Simple frame extraction (like your working scripts)"""
        logger.info(f"📹 Extracting frames: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {}
        
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
        
        # Extract frames with skip (like your working scripts)
        frame_indices = list(range(0, frame_count, self.frame_skip))
        
        logger.info(f"📊 Processing: {len(frame_indices)} frames")
        logger.info(f"   Frame skip: {self.frame_skip}")
        logger.info(f"   Video duration: {video_info['duration']:.1f}s")
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append((frame, frame_idx))
        
        cap.release()
        
        video_info.update({
            'extracted_frames': len(frames),
            'frame_indices': frame_indices,
            'effective_frame_skip': self.frame_skip
        })
        
        return frames, video_info
    
    def process_video_simple_gpu(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame) -> Dict:
        """Process video with simple GPU usage (like your working process*.py scripts)"""
        video_name = Path(video_path).stem
        logger.info(f"🔥 GPU {gpu_id} processing: {video_name}")
        
        start_time = time.time()
        model = self.models[gpu_id]
        
        try:
            # Extract frames (simple approach)
            frames_data, video_info = self.extract_frames_simple(video_path)
            if not frames_data:
                return {'status': 'failed', 'error': 'No frames extracted', 'gpu_id': gpu_id}
            
            total_frames = len(frames_data)
            logger.info(f"🔥 GPU {gpu_id}: Processing {total_frames} frames")
            
            all_detections = []
            
            logger.info(f"🔥 Starting GPU {gpu_id} inference...")
            
            # Process frames one by one (EXACTLY like your working scripts)
            torch.cuda.set_device(gpu_id)  # Set GPU context
            
            for i, (frame, frame_idx) in enumerate(frames_data):
                frame_start = time.time()
                
                # EXACTLY like your working scripts - use model.track()
                results = model.track(source=frame, persist=True, verbose=False, conf=self.confidence_threshold)
                
                # Handle results (your working scripts access [0])
                if results and len(results) > 0:
                    result = results[0]
                else:
                    result = None
                
                # Extract detections (simple approach)
                detection_data = self._extract_detections_simple(
                    result, frame_idx, gpu_id, video_info
                )
                all_detections.append(detection_data)
                
                # Progress update every 100 frames
                if i % 100 == 0:
                    progress = (i / len(frames_data)) * 100
                    frame_time = time.time() - frame_start
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    logger.info(f"🔥 GPU {gpu_id}: {progress:.1f}% - {fps:.1f} FPS - Frame {i}/{len(frames_data)}")
            
            # Merge with GPS data
            final_results = self._merge_detections_with_gps(all_detections, gps_df, video_info)
            
            # Final statistics
            processing_time = time.time() - start_time
            total_fps = total_frames / processing_time if processing_time > 0 else 0
            total_detections = sum(len(d['detections']) for d in all_detections)
            
            logger.info(f"🔥 GPU {gpu_id} completed {video_name}:")
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
            logger.error(f"❌ GPU {gpu_id} processing failed: {e}")
            return {
                'status': 'failed', 
                'error': str(e), 
                'gpu_id': gpu_id, 
                'video_name': video_name
            }
        
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def _extract_detections_simple(self, result, frame_idx: int, 
                                  gpu_id: int, video_info: Dict) -> Dict:
        """Extract detections (exactly like your working scripts)"""
        detection_data = {
            'frame_idx': frame_idx,
            'detections': [],
            'counts': defaultdict(int),
            'gpu_id': gpu_id
        }
        
        # Handle YOLO results (exactly like your working scripts)
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
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
        
        return detection_data
    
    def _merge_detections_with_gps(self, all_detections: List[Dict], gps_df: pd.DataFrame, 
                                  video_info: Dict) -> Dict:
        """Merge detections with GPS data (like your working scripts)"""
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int)
        }
        
        fps = video_info.get('fps', 30)
        frame_indices = video_info.get('frame_indices', [])
        
        # GPS lookup (simple approach)
        gps_lookup = {}
        if not gps_df.empty:
            for _, row in gps_df.iterrows():
                gps_lookup[row['second']] = row
        
        for detection_data in all_detections:
            frame_idx = detection_data['frame_idx']
            
            # Calculate timestamp (simple approach)
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
    
    def process_videos_dual_gpu(self, video_matches: Dict[str, Any]):
        """Process videos with dual GPU (simple approach)"""
        total_videos = len(video_matches)
        logger.info(f"🔥 DUAL GPU PROCESSING: {total_videos} videos")
        
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
        
        # Process videos in pairs
        for i in range(0, len(video_list), self.gpu_count):
            batch_videos = video_list[i:i + self.gpu_count]
            
            logger.info(f"\n🔥 GPU BATCH {i//self.gpu_count + 1}: {len(batch_videos)} videos")
            
            # Create threads for GPU processing
            threads = []
            results = {}
            
            for j, video_info in enumerate(batch_videos):
                gpu_id = j % self.gpu_count
                video_name = Path(video_info['path']).name
                
                logger.info(f"🔥 GPU {gpu_id}: {video_name}")
                
                # Load GPS data
                gps_df = self._load_gps_data(video_info['gps_path'])
                
                # Create thread for GPU processing
                thread = threading.Thread(
                    target=self._process_video_thread,
                    args=(video_info['path'], gpu_id, gps_df, results),
                    name=f"GPU-{gpu_id}-{video_name[:15]}"
                )
                threads.append(thread)
                thread.start()
            
            # Wait for processing
            logger.info(f"🔥 Waiting for GPU processing...")
            
            for thread in threads:
                thread.join(timeout=1800)  # 30 minute timeout
                if thread.is_alive():
                    logger.warning(f"⚠️ Thread {thread.name} timed out")
            
            # Process results
            batch_success = 0
            for video_path, result in results.items():
                if result and result['status'] == 'success':
                    self._save_results(result)
                    processed_count += 1
                    batch_success += 1
                    self.stats['processed_videos'] += 1
                    
                    logger.info(f"🔥 GPU {result['gpu_id']}: {result['video_name']} SUCCESS")
                    logger.info(f"   FPS: {result['fps']:.1f}")
                    logger.info(f"   Detections: {result['total_detections']:,}")
                else:
                    self.stats['failed_videos'] += 1
                    video_name = Path(video_path).name
                    logger.error(f"❌ Processing FAILED: {video_name}")
            
            # GPU cleanup
            for gpu_id in range(self.gpu_count):
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"📊 Batch complete: {batch_success}/{len(batch_videos)} success")
            logger.info(f"📈 Overall progress: {processed_count}/{total_videos}")
        
        # Final summary
        logger.info(f"\n🔥 GPU PROCESSING COMPLETE!")
        logger.info(f"   ✅ Success: {self.stats['processed_videos']}")
        logger.info(f"   ❌ Failed: {self.stats['failed_videos']}")
        logger.info(f"   📊 Success Rate: {(self.stats['processed_videos']/total_videos)*100:.1f}%")
    
    def _process_video_thread(self, video_path: str, gpu_id: int, gps_df: pd.DataFrame, results: Dict):
        """Thread function for GPU processing"""
        try:
            result = self.process_video_simple_gpu(video_path, gpu_id, gps_df)
            results[video_path] = result
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} thread error: {e}")
            results[video_path] = {
                'status': 'failed', 
                'error': str(e), 
                'gpu_id': gpu_id,
                'video_name': Path(video_path).stem
            }
    
    def _load_gps_data(self, gps_path: str) -> pd.DataFrame:
        """Load GPS data (simple approach like your working scripts)"""
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
        """Save results (like your working scripts)"""
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
        """Load matcher results (like your working scripts)"""
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
        logger.info("🔥 SIMPLIFIED GPU SYSTEM INFO:")
        logger.info(f"   🔥 GPUs: {self.gpu_count}")
        logger.info(f"   🖼️ Frame Processing: One by one (like process*.py)")
        logger.info(f"   ⏭️ Frame Skip: {self.frame_skip}")
        logger.info(f"   🎯 Confidence: {self.confidence_threshold}")
        logger.info(f"   📋 Approach: Simple & Reliable (like process*.py)")

def main():
    """Main function - Simple GPU processing"""
    parser = argparse.ArgumentParser(
        description="Simplified GPU YOLO Processor - Based on working process*.py patterns"
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
    
    logger.info("🔥 Starting Simplified GPU YOLO Processor...")
    logger.info(f"   📁 Input: {args.input}")
    logger.info(f"   📁 Output: {args.output}")
    logger.info(f"   🔥 Strategy: Simple & Reliable (like your working process*.py)")
    
    try:
        # Initialize simplified processor
        processor = SimplifiedGPUYOLOProcessor(config)
        
        # Load video matches
        video_matches = processor.load_matcher_results(args.input)
        
        if not video_matches:
            logger.error("❌ No high-quality videos found")
            sys.exit(1)
        
        logger.info(f"🔥 Ready for processing {len(video_matches)} videos")
        
        # Start processing
        processor.process_videos_dual_gpu(video_matches)
        
        logger.info("🔥 PROCESSING COMPLETED!")
        
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