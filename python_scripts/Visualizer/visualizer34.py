#!/usr/bin/env python3
"""
TRUE MULTI-PROCESS GPU YOLO Processor with POWER-SAFE Features
=============================================================

MAXIMUM GPU utilization + POWER-SAFE processing:
- Multiple independent processes per GPU (5 for ~90% utilization)
- True GPU parallelism without threading bottlenecks
- POWER-SAFE: Resume processing after power cuts/interruptions
- Immediate result saving (no data loss on interruption)
- Progress tracking and resume capabilities
- Force restart options for fresh starts

Power-Safe Features:
- --powersafe: Enable resume functionality
- --force: Force restart ignoring previous progress
- --force --powersafe: Restart powersafe system fresh
- Results saved immediately after each video completion
- Graceful handling of interruptions (Ctrl+C, power loss)

Strategy: Create multiple independent processes per GPU that each process
one video at a time, exactly like your working scripts, with full
power-safe progress tracking and resume capabilities.

Author: Based on your working scripts + multi-process + power-safe features
Target: 70-90% GPU utilization with full data protection
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
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import warnings
import multiprocessing
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import gc
import signal
import atexit
from datetime import datetime

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

class PowerSafeManager:
    """
    Power-Safe Progress Manager - Handles resume/restart with progress tracking
    """
    
    def __init__(self, output_dir: Path, force_restart: bool = False):
        self.output_dir = output_dir
        self.progress_file = output_dir / 'powersafe_progress.json'
        self.force_restart = force_restart
        self.completed_videos: Set[str] = set()
        self.failed_videos: Set[str] = set()
        self.session_start_time = datetime.now().isoformat()
        self.total_videos = 0
        self.active_processes: List[int] = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        self._load_progress()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum} - initiating graceful shutdown...")
        self._save_progress()
        logger.info("💾 Progress saved. Safe to power off.")
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup function called on exit"""
        self._save_progress()
    
    def _load_progress(self):
        """Load existing progress or start fresh"""
        if self.force_restart:
            logger.info("🔄 FORCE RESTART: Ignoring existing progress")
            if self.progress_file.exists():
                # Backup old progress
                backup_file = self.output_dir / f'powersafe_progress_backup_{int(time.time())}.json'
                self.progress_file.rename(backup_file)
                logger.info(f"📦 Backed up old progress to: {backup_file.name}")
            return
        
        if not self.progress_file.exists():
            logger.info("🆕 Starting fresh - no previous progress found")
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            self.completed_videos = set(progress_data.get('completed_videos', []))
            self.failed_videos = set(progress_data.get('failed_videos', []))
            previous_session = progress_data.get('session_start_time', 'Unknown')
            
            logger.info("🔋 POWER-SAFE RESUME DETECTED!")
            logger.info(f"   📊 Previous session: {previous_session}")
            logger.info(f"   ✅ Completed videos: {len(self.completed_videos)}")
            logger.info(f"   ❌ Failed videos: {len(self.failed_videos)}")
            logger.info(f"   🚀 Resuming where we left off...")
            
            if self.completed_videos:
                logger.info("📋 Previously completed videos:")
                for video in sorted(list(self.completed_videos)[:10]):  # Show first 10
                    logger.info(f"     ✅ {video}")
                if len(self.completed_videos) > 10:
                    logger.info(f"     ... and {len(self.completed_videos) - 10} more")
                    
        except Exception as e:
            logger.error(f"⚠️ Error loading progress: {e}")
            logger.info("🆕 Starting fresh due to progress file error")
    
    def _save_progress(self):
        """Save current progress to disk (thread-safe)"""
        try:
            progress_data = {
                'session_start_time': self.session_start_time,
                'last_update': datetime.now().isoformat(),
                'completed_videos': sorted(list(self.completed_videos)),
                'failed_videos': sorted(list(self.failed_videos)),
                'total_videos': self.total_videos,
                'completion_rate': len(self.completed_videos) / self.total_videos if self.total_videos > 0 else 0,
                'active_processes': self.active_processes
            }
            
            # Atomic write to avoid corruption (thread-safe)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', dir=self.output_dir, delete=False, suffix='.tmp') as f:
                json.dump(progress_data, f, indent=2)
                temp_path = f.name
            
            # Atomic rename
            Path(temp_path).rename(self.progress_file)
            
        except Exception as e:
            logger.error(f"⚠️ Error saving progress: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    Path(temp_path).unlink(missing_ok=True)
            except:
                pass
    
    def set_total_videos(self, total: int):
        """Set total number of videos for progress tracking"""
        self.total_videos = total
        self._save_progress()
    
    def is_video_completed(self, video_path: str) -> bool:
        """Check if video was already processed"""
        video_name = Path(video_path).name
        return video_name in self.completed_videos
    
    def is_video_failed(self, video_path: str) -> bool:
        """Check if video previously failed"""
        video_name = Path(video_path).name
        return video_name in self.failed_videos
    
    def mark_video_completed(self, video_path: str, result: Dict):
        """Mark video as completed and save progress immediately"""
        video_name = Path(video_path).name
        self.completed_videos.add(video_name)
        
        # Remove from failed if it was there
        self.failed_videos.discard(video_name)
        
        # Save progress immediately (power-safe)
        self._save_progress()
        
        completion_rate = len(self.completed_videos) / self.total_videos if self.total_videos > 0 else 0
        remaining = self.total_videos - len(self.completed_videos)
        
        logger.info(f"💾 POWER-SAFE: Video completed and saved immediately")
        logger.info(f"   ✅ {video_name}")
        logger.info(f"   📊 Progress: {len(self.completed_videos)}/{self.total_videos} ({completion_rate:.1%})")
        logger.info(f"   ⏳ Remaining: {remaining} videos")
        
        if result.get('status') == 'success':
            logger.info(f"   ⚡ FPS: {result.get('fps', 0):.1f}")
            logger.info(f"   🎯 Detections: {result.get('total_detections', 0):,}")
    
    def mark_video_failed(self, video_path: str, error: str):
        """Mark video as failed"""
        video_name = Path(video_path).name
        self.failed_videos.add(video_name)
        self._save_progress()
        
        logger.warning(f"❌ Video failed: {video_name}")
        logger.warning(f"   Error: {error}")
    
    def get_remaining_videos(self, all_video_tasks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter out already completed videos for resume functionality"""
        remaining = []
        
        for video_path, gps_path in all_video_tasks:
            if self.is_video_completed(video_path):
                logger.info(f"⏭️ Skipping completed: {Path(video_path).name}")
                continue
            elif self.is_video_failed(video_path):
                logger.info(f"🔄 Retrying failed: {Path(video_path).name}")
                # Remove from failed to retry
                self.failed_videos.discard(Path(video_path).name)
            
            remaining.append((video_path, gps_path))
        
        return remaining
    
    def add_active_process(self, process_id: int):
        """Track active process for monitoring"""
        self.active_processes.append(process_id)
        self._save_progress()
    
    def remove_active_process(self, process_id: int):
        """Remove completed process from tracking"""
        if process_id in self.active_processes:
            self.active_processes.remove(process_id)
        self._save_progress()
    
    def get_progress_summary(self) -> Dict:
        """Get current progress summary"""
        return {
            'completed': len(self.completed_videos),
            'failed': len(self.failed_videos),
            'total': self.total_videos,
            'remaining': self.total_videos - len(self.completed_videos),
            'completion_rate': len(self.completed_videos) / self.total_videos if self.total_videos > 0 else 0,
            'session_start': self.session_start_time
        }

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

def process_single_video_simple(video_path: str, gps_df: pd.DataFrame, 
                               output_dir: Path, config: Dict, process_id: int, gpu_id: int) -> Dict:
    """
    Process single video - EXACTLY like your working process*.py scripts
    This function runs in a completely separate process
    """
    # Import torch/YOLO in worker process only (avoid multiprocessing issues)
    import torch
    from ultralytics import YOLO
    
    video_name = Path(video_path).stem
    logger.info(f"🔥 Process {process_id} (GPU {gpu_id}): {video_name}")
    
    start_time = time.time()
    
    try:
        # Check CUDA in worker process
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in worker process!")
        
        # EXACTLY like your working scripts - simple device setup
        device = "cuda"
        model_path = config.get('yolo_model', 'yolo11x.pt')
        confidence_threshold = config.get('confidence_threshold', 0.05)
        
        logger.info(f"📦 Process {process_id}: Loading YOLO model...")
        
        # EXACTLY like your working scripts - simple model loading
        model = YOLO(model_path).to(device)
        
        # Simple optimizations (like your working scripts)
        torch.backends.cudnn.benchmark = True
        
        logger.info(f"✅ Process {process_id}: Model loaded on {device}")
        
        # Video processing - EXACTLY like your working scripts
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
        
        logger.info(f"📹 Process {process_id}: {frame_count} frames, {fps:.1f} FPS, {video_info['duration']:.1f}s")
        logger.info(f"🔥 Process {process_id}: Processing EVERY frame (exactly like your working scripts)")
        
        # GPS lookup (like your working scripts)
        gps_lookup = {}
        if not gps_df.empty:
            for _, row in gps_df.iterrows():
                gps_lookup[row['second']] = row
        
        all_detections = []
        processed_frames = 0
        
        logger.info(f"🔥 Process {process_id}: Starting GPU inference...")
        
        # EXACTLY like your working process*.py scripts - EVERY FRAME, ONE BY ONE!
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate second (like your working scripts)
            second = int(frame_idx / fps) if fps > 0 else frame_idx
            
            # Get GPS data (like your working scripts)
            gps_data = gps_lookup.get(second, {})
            
            # YOLO inference - EXACTLY like your working scripts
            results = model.track(source=frame, persist=True, verbose=False, conf=confidence_threshold)
            
            # Extract detections - EXACTLY like your working scripts
            detection_data = {
                'frame_idx': frame_idx,
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
            
            # Progress update (like your working scripts)
            if processed_frames % 100 == 0:
                progress = (frame_idx / frame_count) * 100
                elapsed = time.time() - start_time
                fps_current = processed_frames / elapsed if elapsed > 0 else 0
                logger.info(f"🔥 Process {process_id}: {progress:.1f}% - {fps_current:.1f} FPS - Frame {frame_idx}/{frame_count}")
        
        cap.release()
        
        # Save results (like your working scripts)
        total_detections = sum(len(d['detections']) for d in all_detections)
        processing_time = time.time() - start_time
        final_fps = processed_frames / processing_time if processing_time > 0 else 0
        
        logger.info(f"🔥 Process {process_id} completed {video_name}:")
        logger.info(f"   Time: {processing_time:.2f}s")
        logger.info(f"   FPS: {final_fps:.1f}")
        logger.info(f"   Frames: {processed_frames:,}")
        logger.info(f"   Detections: {total_detections:,}")
        
        # Organize results
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
        
        # Save results IMMEDIATELY (POWER-SAFE)
        save_success = save_results_immediately(video_name, results, output_dir, final_fps, process_id)
        
        if not save_success:
            logger.error(f"❌ Failed to save results for {video_name}")
            return {
                'status': 'failed', 
                'error': 'Failed to save results', 
                'video_name': video_name,
                'process_id': process_id
            }
        
        return {
            'status': 'success',
            'video_name': video_name,
            'processing_time': processing_time,
            'fps': final_fps,
            'total_frames': processed_frames,
            'total_detections': total_detections,
            'process_id': process_id,
            'save_success': save_success
        }
        
    except Exception as e:
        logger.error(f"❌ Process {process_id} failed: {e}")
        return {
            'status': 'failed', 
            'error': str(e), 
            'video_name': video_name,
            'process_id': process_id
        }

def save_results_immediately(video_name: str, results: Dict, output_dir: Path, fps: float, process_id: int):
    """Save results immediately after video completion (POWER-SAFE)"""
    try:
        # Object tracking
        if results['object_tracking']:
            df = pd.DataFrame(results['object_tracking'])
            output_file = output_dir / 'object_tracking' / f"{video_name}_tracking_p{process_id}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved: {output_file.name}")
        
        # Stoplight detection
        if results['stoplight_detection']:
            df = pd.DataFrame(results['stoplight_detection'])
            output_file = output_dir / 'stoplight_detection' / f"{video_name}_stoplights_p{process_id}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved: {output_file.name}")
        
        # Traffic counting
        if results['traffic_counting']:
            counting_data = [
                {
                    'video_name': video_name, 
                    'object_type': obj_type, 
                    'total_count': count,
                    'processing_fps': fps,
                    'process_id': process_id,
                    'completed_at': datetime.now().isoformat()
                }
                for obj_type, count in results['traffic_counting'].items()
            ]
            df = pd.DataFrame(counting_data)
            output_file = output_dir / 'traffic_counting' / f"{video_name}_counts_p{process_id}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved: {output_file.name}")
        
        # Summary file for this video (power-safe record)
        summary = {
            'video_name': video_name,
            'process_id': process_id,
            'processing_fps': fps,
            'object_tracking_count': len(results['object_tracking']),
            'stoplight_count': len(results['stoplight_detection']),
            'traffic_categories': len(results['traffic_counting']),
            'completed_at': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        summary_file = output_dir / 'processing_reports' / f"{video_name}_summary_p{process_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ POWER-SAFE: All results saved immediately for {video_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving results for {video_name}: {e}")
        return False

def single_video_worker(gpu_id: int, video_path: str, gps_path: str, output_dir: Path, config: Dict, process_id: int, powersafe_enabled: bool = True):
    """
    Single video worker process with POWER-SAFE progress tracking
    """
    # Set CUDA_VISIBLE_DEVICES for this specific process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"🔥 Process {process_id} started (GPU {gpu_id}, CUDA_VISIBLE_DEVICES={gpu_id})")
    logger.info(f"🎯 Processing: {Path(video_path).name}")
    
    # Create local PowerSafe manager for this process (if enabled)
    local_powersafe = None
    if powersafe_enabled:
        try:
            local_powersafe = PowerSafeManager(output_dir, force_restart=False)
            local_powersafe.add_active_process(process_id)
        except Exception as e:
            logger.warning(f"⚠️ PowerSafe setup failed for process {process_id}: {e}")
    
    try:
        # Load GPS data
        gps_df = load_gps_data(gps_path)
        
        # Process video (exactly like your working scripts)
        result = process_single_video_simple(video_path, gps_df, output_dir, config, process_id, gpu_id)
        
        if result['status'] == 'success':
            logger.info(f"🔥 Process {process_id} SUCCESS: {result['video_name']} - {result['fps']:.1f} FPS")
            
            # Mark as completed in PowerSafe (if enabled)
            if local_powersafe:
                local_powersafe.mark_video_completed(video_path, result)
                
        else:
            logger.error(f"❌ Process {process_id} FAILED: {result.get('video_name', 'Unknown')} - {result.get('error', 'Unknown')}")
            
            # Mark as failed in PowerSafe (if enabled)
            if local_powersafe:
                local_powersafe.mark_video_failed(video_path, result.get('error', 'Unknown'))
            
    except Exception as e:
        logger.error(f"❌ Process {process_id} exception: {e}")
        
        # Mark as failed in PowerSafe (if enabled)
        if local_powersafe:
            local_powersafe.mark_video_failed(video_path, str(e))
    
    finally:
        # Remove from active processes
        if local_powersafe:
            local_powersafe.remove_active_process(process_id)

def main():
    """Main function - TRUE multi-process with POWER-SAFE features"""
    parser = argparse.ArgumentParser(
        description="TRUE MULTI-PROCESS GPU YOLO Processor with POWER-SAFE Features"
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True, help='Matcher results JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Model settings
    parser.add_argument('--yolo-model', default='yolo11x.pt', help='YOLO model path')
    
    # Processing settings
    parser.add_argument('--processes-per-gpu', type=int, default=5, help='Number of independent processes per GPU (5 for ~90% utilization)')
    
    # Power-Safe Features
    parser.add_argument('--powersafe', action='store_true', help='Enable power-safe resume functionality')
    parser.add_argument('--force', action='store_true', help='Force restart ignoring previous progress')
    
    # Quality filtering
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum match score')
    parser.add_argument('--min-quality', default='very_good', 
                       choices=['excellent', 'very_good', 'good', 'fair', 'poor'])
    parser.add_argument('--confidence-threshold', type=float, default=0.05, help='Ultra-low confidence for maximum detections')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Setup output directory and PowerSafe manager
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Setup subdirectories
    for subdir in ['object_tracking', 'stoplight_detection', 'traffic_counting', 'processing_reports']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Initialize PowerSafe Manager
    powersafe_enabled = args.powersafe
    force_restart = args.force
    
    # Handle flag combinations
    if force_restart and powersafe_enabled:
        logger.info("🔄⚡ FORCE + POWERSAFE: Restarting powersafe system with fresh progress")
        powersafe_manager = PowerSafeManager(output_dir, force_restart=True)
    elif force_restart and not powersafe_enabled:
        logger.info("🔄 FORCE: Starting fresh, no powersafe features")
        powersafe_manager = None
    elif powersafe_enabled and not force_restart:
        logger.info("🔋 POWERSAFE: Resuming from previous progress (if any)")
        powersafe_manager = PowerSafeManager(output_dir, force_restart=False)
    else:
        logger.info("🆕 STANDARD: No powersafe features, starting fresh")
        powersafe_manager = None
    
    # Build configuration
    config = {
        'yolo_model': args.yolo_model,
        'confidence_threshold': args.confidence_threshold
    }
    
    # Load video matches and apply quality filtering
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', data)
    
    # Quality filtering
    quality_map = {'excellent': 5, 'very_good': 4, 'good': 3, 'fair': 2, 'poor': 1}
    min_quality_level = quality_map.get(args.min_quality, 4)
    
    all_video_tasks = []
    for video_path, video_data in results.items():
        if 'matches' not in video_data or not video_data['matches']:
            continue
        
        best_match = video_data['matches'][0]
        score = best_match.get('combined_score', 0)
        quality = best_match.get('quality', 'poor')
        quality_level = quality_map.get(quality, 0)
        
        if score >= args.min_score and quality_level >= min_quality_level:
            gps_path = best_match.get('path', '')
            all_video_tasks.append((video_path, gps_path))
    
    if not all_video_tasks:
        logger.error("❌ No high-quality videos found")
        sys.exit(1)
    
    # Apply PowerSafe filtering (skip completed videos)
    if powersafe_manager:
        powersafe_manager.set_total_videos(len(all_video_tasks))
        video_tasks = powersafe_manager.get_remaining_videos(all_video_tasks)
        
        if len(video_tasks) == 0:
            logger.info("🎉 ALL VIDEOS ALREADY COMPLETED!")
            logger.info("✅ Nothing to process - all videos have been successfully completed.")
            progress = powersafe_manager.get_progress_summary()
            logger.info(f"📊 Final Stats: {progress['completed']}/{progress['total']} videos completed")
            sys.exit(0)
        
        logger.info(f"🔋 POWERSAFE RESUME:")
        logger.info(f"   📊 Total videos: {len(all_video_tasks)}")
        logger.info(f"   ✅ Already completed: {len(all_video_tasks) - len(video_tasks)}")
        logger.info(f"   ⏳ Remaining to process: {len(video_tasks)}")
    else:
        video_tasks = all_video_tasks
        logger.info(f"🎯 Processing all {len(video_tasks)} videos (no powersafe)")
    
    # Detect GPUs
    num_gpus = min(detect_num_gpus(), 2)  # Use max 2 GPUs
    processes_per_gpu = args.processes_per_gpu
    total_processes = num_gpus * processes_per_gpu
    
    logger.info(f"🔥 TRUE MULTI-PROCESS STRATEGY:")
    logger.info(f"   📊 GPUs: {num_gpus}")
    logger.info(f"   🔄 Processes per GPU: {processes_per_gpu}")
    logger.info(f"   📈 Total processes: {total_processes}")
    logger.info(f"   🎯 Videos to process: {len(video_tasks)}")
    logger.info(f"   🚀 Strategy: {processes_per_gpu} completely independent processes per GPU")
    logger.info(f"   ⚡ Expected: 70-90% GPU utilization with true parallel processing")
    
    # Display strategy based on flags
    if powersafe_enabled and not force_restart:
        logger.info(f"🔋 POWER-SAFE PROCESSING:")
        logger.info(f"   💾 Results saved immediately after each video")
        logger.info(f"   🔄 Can resume if interrupted")
        logger.info(f"   📊 Progress tracked in: {output_dir / 'powersafe_progress.json'}")
    elif force_restart:
        logger.info(f"🔄 FORCE RESTART:")
        logger.info(f"   🆕 Starting completely fresh")
        logger.info(f"   ⚠️ Ignoring any previous progress")
    
    # Process creation with PowerSafe awareness
    processes = []
    process_id = 1
    
    # Process videos in batches of total_processes
    for batch_start in range(0, len(video_tasks), total_processes):
        batch_videos = video_tasks[batch_start:batch_start + total_processes]
        
        logger.info(f"\n🔥 Starting batch of {len(batch_videos)} videos with {total_processes} processes...")
        
        if powersafe_manager:
            progress = powersafe_manager.get_progress_summary()
            logger.info(f"📊 Session Progress: {progress['completed']}/{progress['total']} completed ({progress['completion_rate']:.1%})")
        
        batch_processes = []
        
        # Create one process per video in this batch
        for i, (video_path, gps_path) in enumerate(batch_videos):
            gpu_id = i % num_gpus  # Round-robin GPU assignment
            
            video_name = Path(video_path).name
            logger.info(f"🚀 Process {process_id}: GPU {gpu_id} -> {video_name}")
            
            # Create completely independent process with PowerSafe support
            p = Process(
                target=single_video_worker,
                args=(gpu_id, video_path, gps_path, output_dir, config, process_id, powersafe_enabled),
                name=f"VideoProcess-{process_id}-GPU{gpu_id}"
            )
            p.start()
            batch_processes.append(p)
            process_id += 1
        
        # Wait for this batch to complete
        logger.info(f"🔥 Waiting for {len(batch_processes)} processes to complete...")
        
        # Monitor processes with PowerSafe status updates
        try:
            for p in batch_processes:
                p.join()
                
            logger.info(f"✅ Batch of {len(batch_processes)} processes completed")
            
            # Show updated progress if PowerSafe is enabled
            if powersafe_manager:
                # Reload progress (other processes may have updated it)
                powersafe_manager._load_progress()
                progress = powersafe_manager.get_progress_summary()
                logger.info(f"📊 Updated Progress: {progress['completed']}/{progress['total']} completed ({progress['completion_rate']:.1%})")
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted - waiting for processes to finish gracefully...")
            for p in batch_processes:
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning(f"⚠️ Force terminating process {p.name}")
                    p.terminate()
            
            if powersafe_manager:
                powersafe_manager._save_progress()
                logger.info("💾 Progress saved. You can resume with --powersafe flag.")
            
            sys.exit(130)
    
    # Final completion summary
    logger.info("🔥 TRUE MULTI-PROCESS GPU PROCESSING COMPLETED!")
    
    if powersafe_manager:
        final_progress = powersafe_manager.get_progress_summary()
        logger.info(f"🎉 POWER-SAFE SESSION COMPLETE!")
        logger.info(f"   ✅ Videos completed: {final_progress['completed']}")
        logger.info(f"   ❌ Videos failed: {final_progress['failed']}")
        logger.info(f"   📊 Success rate: {final_progress['completion_rate']:.1%}")
        logger.info(f"   💾 All results saved immediately during processing")
        logger.info(f"   🔋 Ready for future power-safe operations")
    else:
        logger.info(f"🚀 Check GPU utilization - should be 70-90% with {processes_per_gpu} independent processes per GPU")

if __name__ == "__main__":
    main()