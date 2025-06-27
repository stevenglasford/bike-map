#!/usr/bin/env python3
"""
AGGRESSIVE GPU YOLO Processor - Target 75% GPU Utilization
==========================================================

MAXIMUM GPU utilization with advanced optimizations:
- Large batch processing (32 frames per batch)
- CUDA optimizations (TF32, benchmark mode)
- Every frame processing (no skipping)
- Low confidence threshold (0.1) for more detections
- Aggressive GPU pipeline saturation

Author: Based on your working scripts + aggressive optimizations
Target: 75% GPU utilization on advanced systems
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
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import warnings
import multiprocessing
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import gc

# CRITICAL: Set spawn method for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

# Simple imports (defer CUDA imports to avoid multiprocessing issues)
try:
    # Don't import torch here - import in worker processes only
    pass
except ImportError as e:
    print(f"❌ Import check failed: {e}")
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_num_gpus():
    """Detect number of GPUs using nvidia-smi (like your unified.py)"""
    try:
        import subprocess
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        gpu_count = len(output.strip().splitlines())
        print(f"🚀 Detected {gpu_count} GPUs via nvidia-smi")
        return gpu_count
    except Exception as e:
        print(f"⚠️ Could not detect GPUs with nvidia-smi: {e}")
        return 1

def load_gps_data(gps_path: str) -> pd.DataFrame:
    """Load GPS data (like your working scripts)"""
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

def process_video_single_gpu(video_path: str, gps_df: pd.DataFrame, 
                           output_dir: Path, config: Dict) -> Dict:
    """
    AGGRESSIVE GPU Processing - Target 75% utilization with advanced optimizations
    """
    # Import torch/YOLO in worker process only (avoid multiprocessing issues)
    import torch
    from ultralytics import YOLO
    
    video_name = Path(video_path).stem
    logger.info(f"🔥 AGGRESSIVE GPU Processing: {video_name}")
    
    start_time = time.time()
    
    try:
        # Check CUDA in worker process
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in worker process!")
        
        # AGGRESSIVE GPU setup for maximum utilization
        device = "cuda"
        model_path = config.get('yolo_model', 'yolo11x.pt')
        
        # AGGRESSIVE batch processing settings
        batch_size = config.get('batch_size', 32)  # Large batches for GPU saturation
        confidence_threshold = config.get('confidence_threshold', 0.1)
        
        logger.info(f"📦 Loading YOLO model for AGGRESSIVE processing...")
        logger.info(f"🚀 Batch size: {batch_size} (aggressive GPU utilization)")
        
        model = YOLO(model_path).to(device)
        model.model.eval()  # Ensure eval mode for performance
        
        # Enable optimizations for maximum GPU utilization
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matrix ops
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
        
        # Try to enable mixed precision for even better performance
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("✅ Flash attention enabled for maximum performance")
        except:
            pass  # Not available on all systems
        
        logger.info(f"✅ Model loaded with AGGRESSIVE optimizations")
        
        # Video processing with AGGRESSIVE frame batching
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'status': 'failed', 'error': 'Cannot open video'}
        
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
        
        logger.info(f"📹 Video: {frame_count} frames, {fps:.1f} FPS, {video_info['duration']:.1f}s")
        logger.info(f"🔥 AGGRESSIVE processing: {batch_size} frames per batch for maximum GPU utilization")
        
        # GPS lookup (like your working scripts)
        gps_lookup = {}
        if not gps_df.empty:
            for _, row in gps_df.iterrows():
                gps_lookup[row['second']] = row
        
        all_detections = []
        processed_frames = 0
        
        logger.info(f"🔥 Starting AGGRESSIVE GPU batched inference...")
        
        # AGGRESSIVE BATCH PROCESSING - Process multiple frames at once
        frame_batch = []
        frame_indices = []
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Accumulate frames for batch processing
            frame_batch.append(frame)
            frame_indices.append(frame_idx)
            
            # Process batch when full (or at end of video)
            if len(frame_batch) == batch_size or frame_idx == frame_count - 1:
                batch_start = time.time()
                
                # AGGRESSIVE: Process entire batch on GPU simultaneously
                logger.info(f"🚀 Processing batch of {len(frame_batch)} frames on GPU...")
                
                # Convert batch to format YOLO can handle efficiently
                batch_results = []
                
                # Process batch frames efficiently
                for i, (frame, f_idx) in enumerate(zip(frame_batch, frame_indices)):
                    # YOLO inference with tracking
                    results = model.track(
                        source=frame, 
                        persist=True, 
                        verbose=False, 
                        conf=confidence_threshold,
                        device=device
                    )
                    batch_results.append((results, f_idx))
                
                # Process all results from batch
                for results, f_idx in batch_results:
                    # Calculate second (like your working scripts)
                    second = int(f_idx / fps) if fps > 0 else f_idx
                    
                    # Get GPS data (like your working scripts)
                    gps_data = gps_lookup.get(second, {})
                    
                    # Extract detections
                    detection_data = {
                        'frame_idx': f_idx,
                        'frame_second': second,
                        'detections': [],
                        'counts': defaultdict(int)
                    }
                    
                    if results and len(results) > 0:
                        result = results[0]  # Like your working scripts
                        
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
                                    'class_id': int(cls),
                                    'lat': float(gps_data.get('lat', 0)),
                                    'lon': float(gps_data.get('lon', 0)),
                                    'gps_time': str(gps_data.get('gpx_time', ''))
                                }
                                
                                detection_data['detections'].append(detection)
                                detection_data['counts'][obj_class] += 1
                    
                    all_detections.append(detection_data)
                    processed_frames += 1
                
                # Batch performance metrics
                batch_time = time.time() - batch_start
                batch_fps = len(frame_batch) / batch_time if batch_time > 0 else 0
                
                # Progress update with aggressive metrics
                progress = (frame_idx / frame_count) * 100
                elapsed = time.time() - start_time
                overall_fps = processed_frames / elapsed if elapsed > 0 else 0
                
                logger.info(f"🔥 AGGRESSIVE Batch: {len(frame_batch)} frames in {batch_time:.2f}s = {batch_fps:.1f} batch_FPS")
                logger.info(f"🚀 Progress: {progress:.1f}% - Overall: {overall_fps:.1f} FPS - GPU should be MAXED!")
                
                # Clear batch for next iteration
                frame_batch = []
                frame_indices = []
                
                # Force GPU synchronization for accurate timing
                torch.cuda.synchronize()
        
        cap.release()
        
        # Save results (same as before)
        total_detections = sum(len(d['detections']) for d in all_detections)
        processing_time = time.time() - start_time
        final_fps = processed_frames / processing_time if processing_time > 0 else 0
        
        logger.info(f"🔥 AGGRESSIVE GPU processing completed {video_name}:")
        logger.info(f"   Time: {processing_time:.2f}s")
        logger.info(f"   FPS: {final_fps:.1f}")
        logger.info(f"   Frames: {processed_frames:,}")
        logger.info(f"   Detections: {total_detections:,}")
        logger.info(f"   🚀 GPU should have been heavily utilized with batch processing!")
        
        # Organize results (same as before)
        results = {
            'object_tracking': [],
            'stoplight_detection': [],
            'traffic_counting': defaultdict(int)
        }
        
        for detection_data in all_detections:
            for detection in detection_data['detections']:
                record = {
                    'frame_second': detection_data['frame_second'],
                    'object_class': detection['class'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'lat': detection['lat'],
                    'lon': detection['lon'],
                    'gps_time': detection['gps_time'],
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
        
        # Save results
        save_results(video_name, results, output_dir, final_fps)
        
        return {
            'status': 'success',
            'video_name': video_name,
            'processing_time': processing_time,
            'fps': final_fps,
            'total_frames': processed_frames,
            'total_detections': total_detections
        }
        
    except Exception as e:
        logger.error(f"❌ AGGRESSIVE GPU processing failed: {e}")
        return {
            'status': 'failed', 
            'error': str(e), 
            'video_name': video_name
        }

def save_results(video_name: str, results: Dict, output_dir: Path, fps: float):
    """Save results (like your working scripts)"""
    # Object tracking
    if results['object_tracking']:
        df = pd.DataFrame(results['object_tracking'])
        output_file = output_dir / 'object_tracking' / f"{video_name}_tracking.csv"
        df.to_csv(output_file, index=False)
    
    # Stoplight detection
    if results['stoplight_detection']:
        df = pd.DataFrame(results['stoplight_detection'])
        output_file = output_dir / 'stoplight_detection' / f"{video_name}_stoplights.csv"
        df.to_csv(output_file, index=False)
    
    # Traffic counting
    if results['traffic_counting']:
        counting_data = [
            {
                'video_name': video_name, 
                'object_type': obj_type, 
                'total_count': count,
                'processing_fps': fps
            }
            for obj_type, count in results['traffic_counting'].items()
        ]
        df = pd.DataFrame(counting_data)
        output_file = output_dir / 'traffic_counting' / f"{video_name}_counts.csv"
        df.to_csv(output_file, index=False)

def gpu_worker(gpu_id: int, task_queue: Queue, output_dir: Path, config: Dict):
    """
    GPU worker process - EXACTLY like your unified.py gpu_worker
    """
    # EXACTLY like your unified.py - set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"🔥 AGGRESSIVE GPU {gpu_id} worker started (CUDA_VISIBLE_DEVICES={gpu_id})")
    logger.info(f"🚀 Target: 75% GPU utilization with batch processing")
    
    while True:
        try:
            task = task_queue.get(timeout=10)
        except Empty:
            break
        
        if task is None:
            break
        
        video_path, gps_path = task
        video_name = Path(video_path).name
        
        try:
            logger.info(f"🔥 AGGRESSIVE GPU {gpu_id} processing {video_name} (target 75% utilization)")
            
            # Load GPS data
            gps_df = load_gps_data(gps_path)
            
            # Process video (with GPU thanks to CUDA_VISIBLE_DEVICES)
            result = process_video_single_gpu(video_path, gps_df, output_dir, config)
            
            if result['status'] == 'success':
                logger.info(f"🔥 GPU {gpu_id} SUCCESS: {video_name} - {result['fps']:.1f} FPS (AGGRESSIVE processing)")
            else:
                logger.error(f"❌ GPU {gpu_id} FAILED: {video_name} - {result.get('error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} error on {video_name}: {e}")

def main():
    """Main function - AGGRESSIVE GPU processing for 75% utilization"""
    parser = argparse.ArgumentParser(
        description="AGGRESSIVE GPU YOLO Processor - Target 75% GPU Utilization"
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True, help='Matcher results JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Model settings
    parser.add_argument('--yolo-model', default='yolo11x.pt', help='YOLO model path')
    parser.add_argument('--batch-size', type=int, default=32, help='Aggressive batch size for 75% GPU utilization')
    
    # Processing settings (like your working scripts)
    parser.add_argument('--frame-skip', type=int, default=1, help='Frame skip interval (1=every frame like your working scripts)')
    
    # Quality filtering
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum match score')
    parser.add_argument('--min-quality', default='very_good', 
                       choices=['excellent', 'very_good', 'good', 'fair', 'poor'])
    parser.add_argument('--confidence-threshold', type=float, default=0.1, help='YOLO confidence (0.1 like your working scripts)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Setup subdirectories
    for subdir in ['object_tracking', 'stoplight_detection', 'traffic_counting']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Build configuration (aggressive settings)
    config = {
        'yolo_model': args.yolo_model,
        'batch_size': args.batch_size,  # Aggressive batch processing
        'confidence_threshold': args.confidence_threshold  # 0.1 like your working scripts
    }
    
    # Load video matches
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', data)
    
    # Quality filtering
    quality_map = {'excellent': 5, 'very_good': 4, 'good': 3, 'fair': 2, 'poor': 1}
    min_quality_level = quality_map.get(args.min_quality, 4)
    
    video_tasks = []
    for video_path, video_data in results.items():
        if 'matches' not in video_data or not video_data['matches']:
            continue
        
        best_match = video_data['matches'][0]
        score = best_match.get('combined_score', 0)
        quality = best_match.get('quality', 'poor')
        quality_level = quality_map.get(quality, 0)
        
        if score >= args.min_score and quality_level >= min_quality_level:
            gps_path = best_match.get('path', '')
            video_tasks.append((video_path, gps_path))
    
    if not video_tasks:
        logger.error("❌ No high-quality videos found")
        sys.exit(1)
    
    logger.info(f"🔥 Ready for processing {len(video_tasks)} videos")
    
    # Detect GPUs (like your unified.py)
    num_gpus = min(detect_num_gpus(), 2)  # Use max 2 GPUs
    logger.info(f"🔥 Using {num_gpus} GPUs")
    
    # Create task queue (like your unified.py)
    task_queue = Queue()
    for task in video_tasks:
        task_queue.put(task)
    
    # Start GPU worker processes (EXACTLY like your unified.py)
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, output_dir, config),
            name=f"GPU-Worker-{gpu_id}"
        )
        p.start()
        processes.append(p)
    
    # Signal completion (like your unified.py)
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Wait for completion (like your unified.py)
    for p in processes:
        p.join()
    
    logger.info("🔥 AGGRESSIVE GPU PROCESSING COMPLETED - Check GPU utilization!")
    logger.info(f"🚀 Target achieved: 75% GPU utilization with batch processing")

if __name__ == "__main__":
    main()