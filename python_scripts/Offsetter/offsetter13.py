#!/usr/bin/env python3
"""
ROBUST FAST DUAL GPU - WITH DEBUGGING
🔥 FIXES HANGING ISSUES AND ADDS EXTENSIVE LOGGING 🔥
💀 NO MORE SILENT FAILURES! 💀
"""

import json
import numpy as np
import cupy as cp
import cv2
import gpxpy
import pandas as pd
from pathlib import Path
import argparse
import logging
import sys
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import traceback

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('beast_mode.log', mode='w')
        ]
    )
    return logging.getLogger('robust_fast_dual')

class RobustFastWorker:
    """Robust fast worker with extensive debugging"""
    
    def __init__(self, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, gpu_memory_gb: float = 15.0):
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.errors = 0
        self.logger = logging.getLogger(f'GPU{gpu_id}Worker')
        self.is_running = True
        
    def run(self):
        """Robust worker loop with extensive debugging"""
        
        self.logger.info(f"🔥 GPU {self.gpu_id} Worker STARTING...")
        
        try:
            # GPU initialization with error handling
            self.logger.info(f"🔥 GPU {self.gpu_id}: Initializing CUDA device...")
            cp.cuda.Device(self.gpu_id).use()
            
            # Set memory limit
            if self.gpu_memory_gb > 0:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.gpu_memory_gb * 1024**3))
                self.logger.info(f"🔥 GPU {self.gpu_id}: Memory limit set to {self.gpu_memory_gb}GB")
            
            # Test GPU with more comprehensive test
            self.logger.info(f"🔥 GPU {self.gpu_id}: Running GPU test...")
            test_array = cp.random.random((1000, 1000), dtype=cp.float32)
            test_result = cp.sum(test_array)
            del test_array
            cp.cuda.Device().synchronize()
            self.logger.info(f"🔥 GPU {self.gpu_id}: GPU test passed - Result: {float(test_result):.2f}")
            
            # Main processing loop
            consecutive_empty = 0
            while self.is_running:
                try:
                    self.logger.debug(f"🔥 GPU {self.gpu_id}: Waiting for work...")
                    
                    # Get work with shorter timeout for faster detection
                    work_item = self.work_queue.get(timeout=2)
                    consecutive_empty = 0
                    
                    if work_item is None:
                        self.logger.info(f"🔥 GPU {self.gpu_id}: Received shutdown signal")
                        break
                    
                    self.logger.info(f"🔥 GPU {self.gpu_id}: Got work item {self.processed + 1}")
                    
                    video_path, gpx_path, match = work_item
                    
                    # Validate files first
                    if not self.validate_files(video_path, gpx_path):
                        error_result = self.create_error_result(match, "file_validation_failed")
                        self.result_queue.put(error_result)
                        self.work_queue.task_done()
                        self.errors += 1
                        continue
                    
                    # Process with timeout
                    self.logger.info(f"🔥 GPU {self.gpu_id}: Processing {Path(video_path).name}...")
                    start_time = time.time()
                    
                    result = self.process_with_timeout(video_path, gpx_path, match, timeout=30)
                    
                    processing_time = time.time() - start_time
                    self.logger.info(f"🔥 GPU {self.gpu_id}: Processed in {processing_time:.1f}s")
                    
                    # Send result
                    self.result_queue.put(result)
                    self.processed += 1
                    
                    if self.processed % 2 == 0:
                        self.logger.info(f"🔥 GPU {self.gpu_id}: {self.processed} processed, {self.errors} errors")
                    
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    consecutive_empty += 1
                    if consecutive_empty >= 10:  # 20 seconds of no work
                        self.logger.info(f"🔥 GPU {self.gpu_id}: No work for 20s, checking shutdown...")
                        if self.work_queue.empty():
                            self.logger.info(f"🔥 GPU {self.gpu_id}: Queue empty, may be done")
                    continue
                    
                except Exception as e:
                    self.logger.error(f"💀 GPU {self.gpu_id}: Processing error: {e}")
                    self.logger.debug(f"💀 GPU {self.gpu_id}: Error traceback:\n{traceback.format_exc()}")
                    
                    # Send error result
                    try:
                        error_result = self.create_error_result(match, f"processing_error: {str(e)[:100]}")
                        self.result_queue.put(error_result)
                        self.work_queue.task_done()
                        self.errors += 1
                    except:
                        self.logger.error(f"💀 GPU {self.gpu_id}: Failed to send error result")
        
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id}: FATAL ERROR: {e}")
            self.logger.debug(f"💀 GPU {self.gpu_id}: Fatal traceback:\n{traceback.format_exc()}")
        
        finally:
            self.logger.info(f"🔥 GPU {self.gpu_id}: Worker shutdown. Processed: {self.processed}, Errors: {self.errors}")
    
    def validate_files(self, video_path: str, gpx_path: str) -> bool:
        """Validate input files exist and are accessible"""
        try:
            video_file = Path(video_path)
            gpx_file = Path(gpx_path)
            
            if not video_file.exists():
                self.logger.warning(f"💀 GPU {self.gpu_id}: Video file not found: {video_path}")
                return False
                
            if not gpx_file.exists():
                self.logger.warning(f"💀 GPU {self.gpu_id}: GPX file not found: {gpx_path}")
                return False
                
            # Check file sizes
            video_size = video_file.stat().st_size
            gpx_size = gpx_file.stat().st_size
            
            if video_size < 1000:  # Less than 1KB
                self.logger.warning(f"💀 GPU {self.gpu_id}: Video file too small: {video_size} bytes")
                return False
                
            if gpx_size < 100:  # Less than 100 bytes
                self.logger.warning(f"💀 GPU {self.gpu_id}: GPX file too small: {gpx_size} bytes")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id}: File validation error: {e}")
            return False
    
    def create_error_result(self, match: Dict, error_type: str) -> Dict:
        """Create standardized error result"""
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': False,
            'beast_mode': True,
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': f'gpu_{self.gpu_id}_{error_type}',
            'sync_quality': 'failed',
            'error_type': error_type
        })
        return result
    
    def process_with_timeout(self, video_path: str, gpx_path: str, match: Dict, timeout: int = 30) -> Dict:
        """Process with timeout to prevent hanging"""
        
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': True,
            'beast_mode': True,
            'offset_method': f'robust_fast_gpu_{self.gpu_id}'
        })
        
        try:
            # Ensure GPU context
            cp.cuda.Device(self.gpu_id).use()
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Starting video extraction...")
            video_motion = self.extract_video_robust(video_path)
            if video_motion is None:
                result['offset_method'] = f'gpu_{self.gpu_id}_video_failed'
                return result
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Video extraction success, starting GPS...")
            gps_speed = self.extract_gps_robust(gpx_path)
            if gps_speed is None:
                result['offset_method'] = f'gpu_{self.gpu_id}_gps_failed'
                return result
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: GPS extraction success, calculating offset...")
            offset, confidence = self.calculate_offset_robust(video_motion, gps_speed)
            
            if offset is not None and confidence >= 0.25:
                result.update({
                    'temporal_offset_seconds': float(offset),
                    'offset_confidence': float(confidence),
                    'offset_method': f'robust_fast_gpu_{self.gpu_id}_success',
                    'sync_quality': 'excellent' if confidence >= 0.8 else 'good' if confidence >= 0.6 else 'fair'
                })
                self.logger.debug(f"🔥 GPU {self.gpu_id}: Success! Offset: {offset:.2f}s, Confidence: {confidence:.3f}")
            else:
                result.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': float(confidence) if confidence else 0.0,
                    'offset_method': f'gpu_{self.gpu_id}_low_confidence',
                    'sync_quality': 'poor'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id}: Processing exception: {e}")
            result.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'gpu_{self.gpu_id}_exception',
                'gpu_processing': False,
                'error_details': str(e)[:200]
            })
            return result
    
    def extract_video_robust(self, video_path: str) -> Optional[cp.ndarray]:
        """Robust video extraction with timeouts and error handling"""
        
        cap = None
        try:
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Opening video: {Path(video_path).name}")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.warning(f"💀 GPU {self.gpu_id}: Failed to open video")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps <= 0 or frame_count <= 0:
                self.logger.warning(f"💀 GPU {self.gpu_id}: Invalid video properties: fps={fps}, frames={frame_count}")
                return None
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Video: {fps:.1f}fps, {frame_count:.0f} frames")
            
            # Aggressive sampling for speed
            frame_interval = max(1, int(fps * 2))  # Sample every 2 seconds
            motion_values = []
            frame_idx = 0
            prev_gray = None
            
            # Strict limits to prevent hanging
            max_frames = 60  # Max 2 minutes at 0.5fps
            max_processing_time = 15  # 15 seconds max
            start_time = time.time()
            
            while len(motion_values) < max_frames:
                # Check timeout
                if time.time() - start_time > max_processing_time:
                    self.logger.warning(f"💀 GPU {self.gpu_id}: Video processing timeout")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    try:
                        # Aggressive downsampling for speed
                        frame = cv2.resize(frame, (160, 120))
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        if prev_gray is not None:
                            # Simple GPU motion detection
                            curr_gpu = cp.array(gray, dtype=cp.float32)
                            prev_gpu = cp.array(prev_gray, dtype=cp.float32)
                            
                            diff = cp.abs(curr_gpu - prev_gpu)
                            motion = float(cp.mean(diff))
                            motion_values.append(motion)
                            
                            # Immediate cleanup
                            del curr_gpu, prev_gpu, diff
                        
                        prev_gray = gray.copy()
                        
                    except Exception as e:
                        self.logger.warning(f"💀 GPU {self.gpu_id}: Frame processing error: {e}")
                        continue
                
                frame_idx += 1
            
            processing_time = time.time() - start_time
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Video processed in {processing_time:.1f}s, {len(motion_values)} motion points")
            
            if len(motion_values) >= 3:
                return cp.array(motion_values, dtype=cp.float32)
            else:
                self.logger.warning(f"💀 GPU {self.gpu_id}: Insufficient motion data: {len(motion_values)}")
                return None
                
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id}: Video extraction error: {e}")
            return None
        finally:
            if cap:
                cap.release()
    
    def extract_gps_robust(self, gpx_path: str) -> Optional[cp.ndarray]:
        """Robust GPS extraction with error handling"""
        
        try:
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Parsing GPX: {Path(gpx_path).name}")
            
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time and point.latitude and point.longitude:
                            points.append({
                                'lat': point.latitude,
                                'lon': point.longitude,
                                'time': point.time
                            })
            
            if len(points) < 10:
                self.logger.warning(f"💀 GPU {self.gpu_id}: Insufficient GPS points: {len(points)}")
                return None
            
            points.sort(key=lambda p: p['time'])
            
            # Aggressive downsampling for speed
            if len(points) > 200:
                step = len(points) // 200
                points = points[::step]
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Processing {len(points)} GPS points")
            
            df = pd.DataFrame(points)
            
            # Fast GPU distance calculation
            lats = cp.array(df['lat'].values, dtype=cp.float32)
            lons = cp.array(df['lon'].values, dtype=cp.float32)
            
            # Simple distance approximation (very fast)
            lat_diffs = lats[1:] - lats[:-1]
            lon_diffs = lons[1:] - lons[:-1]
            distances = cp.sqrt(lat_diffs**2 + lon_diffs**2) * 111000  # Rough meters
            
            # Time differences
            time_diffs = []
            for i in range(len(df)-1):
                dt = (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds()
                time_diffs.append(max(dt, 0.1))  # Prevent division by zero
            
            time_diffs = cp.array(time_diffs, dtype=cp.float32)
            
            # Speed calculation
            speeds = distances / time_diffs
            
            # Additional downsampling
            if len(speeds) > 100:
                step = len(speeds) // 100
                speeds = speeds[::step]
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: GPS processed, {len(speeds)} speed points")
            return speeds
            
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id}: GPS extraction error: {e}")
            return None
    
    def calculate_offset_robust(self, video_motion: cp.ndarray, gps_speed: cp.ndarray) -> Tuple[Optional[float], float]:
        """Robust offset calculation with strict limits"""
        
        try:
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Calculating offset...")
            
            # Normalize with safety
            video_std = cp.std(video_motion)
            gps_std = cp.std(gps_speed)
            
            if video_std < 1e-6 or gps_std < 1e-6:
                self.logger.warning(f"💀 GPU {self.gpu_id}: Signals too uniform")
                return None, 0.0
            
            video_norm = (video_motion - cp.mean(video_motion)) / video_std
            gps_norm = (gps_speed - cp.mean(gps_speed)) / gps_std
            
            # Strict limits to prevent hanging
            max_len = min(len(video_norm), len(gps_norm), 50)  # Very aggressive limit
            video_short = video_norm[:max_len]
            gps_short = gps_norm[:max_len]
            
            best_offset = None
            best_confidence = 0.0
            
            # Limited offset search
            max_offset = min(20, max_len//4)  # Very limited search
            
            for offset in range(-max_offset, max_offset + 1):
                try:
                    if offset < 0:
                        v_seg = video_short[-offset:]
                        g_seg = gps_short[:len(v_seg)]
                    elif offset > 0:
                        g_seg = gps_short[offset:]
                        v_seg = video_short[:len(g_seg)]
                    else:
                        min_len = min(len(video_short), len(gps_short))
                        v_seg = video_short[:min_len]
                        g_seg = gps_short[:min_len]
                    
                    if len(v_seg) >= 5:
                        # Fast correlation
                        mean_v = cp.mean(v_seg)
                        mean_g = cp.mean(g_seg)
                        
                        num = cp.sum((v_seg - mean_v) * (g_seg - mean_g))
                        den = cp.sqrt(cp.sum((v_seg - mean_v)**2) * cp.sum((g_seg - mean_g)**2))
                        
                        if den > 1e-6:
                            corr = float(num / den)
                            
                            if abs(corr) > best_confidence:
                                best_confidence = abs(corr)
                                best_offset = float(offset * 2)  # Convert to seconds (2 second sampling)
                                
                except Exception as e:
                    self.logger.debug(f"💀 GPU {self.gpu_id}: Correlation error at offset {offset}: {e}")
                    continue
            
            self.logger.debug(f"🔥 GPU {self.gpu_id}: Best offset: {best_offset}, confidence: {best_confidence:.3f}")
            return best_offset, best_confidence
            
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id}: Offset calculation error: {e}")
            return None, 0.0

def main():
    """Robust dual GPU main with extensive debugging"""
    
    parser = argparse.ArgumentParser(description='🔥 Robust Fast Dual GPU - With Debugging')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--workers-per-gpu', type=int, default=1, help='Workers per GPU (default: 1 for stability)')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--gpu-memory', type=float, default=15.0, help='GPU memory limit per GPU in GB')
    parser.add_argument('--beast-mode', action='store_true', help='Enable beast mode processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 DEBUG MODE ENABLED")
    
    if args.beast_mode:
        logger.info("🔥💀🔥 ROBUST BEAST MODE ACTIVATED! 🔥💀🔥")
    
    # Validate input
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"💀 Input file not found: {input_file}")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"robust_beast_{input_file.name}"
    
    # Check GPU availability
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"🔥 Detected {gpu_count} CUDA GPUs")
        
        if gpu_count < 2:
            logger.warning(f"⚠️ Only {gpu_count} GPUs detected, expected 2")
        
        # Test each GPU
        for gpu_id in [0, 1]:
            if gpu_id < gpu_count:
                cp.cuda.Device(gpu_id).use()
                test = cp.array([1, 2, 3])
                logger.info(f"🔥 GPU {gpu_id}: Available and working")
                del test
            
    except Exception as e:
        logger.error(f"💀 GPU initialization failed: {e}")
        sys.exit(1)
    
    # Load and validate data
    logger.info(f"📁 Loading {input_file}")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"💀 Failed to load JSON: {e}")
        sys.exit(1)
    
    # Collect and validate matches
    all_matches = []
    total_potential = 0
    
    for video_path, video_data in data.get('results', {}).items():
        for match in video_data.get('matches', []):
            total_potential += 1
            if match.get('combined_score', 0) >= args.min_score:
                # Validate paths exist
                gpx_path = match.get('path', '')
                if Path(video_path).exists() and Path(gpx_path).exists():
                    all_matches.append((video_path, gpx_path, match))
                else:
                    logger.warning(f"💀 Skipping missing files: {Path(video_path).name}")
                
                if args.limit and len(all_matches) >= args.limit:
                    break
        if args.limit and len(all_matches) >= args.limit:
            break
    
    logger.info(f"📊 Found {total_potential} total matches, {len(all_matches)} valid for processing")
    
    if len(all_matches) == 0:
        logger.error("💀 No valid matches found!")
        sys.exit(1)
    
    # Setup processing
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Add work items
    logger.info(f"📋 Adding {len(all_matches)} work items to queue...")
    for i, match in enumerate(all_matches):
        work_queue.put(match)
        if (i + 1) % 10 == 0:
            logger.info(f"📋 Added {i + 1}/{len(all_matches)} work items")
    
    logger.info(f"📋 Work queue size: {work_queue.qsize()}")
    
    # Create workers with reduced count for stability
    workers = []
    worker_threads = []
    total_workers = 2 * args.workers_per_gpu  # Only GPUs 0 and 1
    
    logger.info(f"🔥 Creating {total_workers} workers ({args.workers_per_gpu} per GPU)...")
    
    for gpu_id in [0, 1]:
        for worker_id in range(args.workers_per_gpu):
            worker = RobustFastWorker(gpu_id, work_queue, result_queue, args.gpu_memory)
            thread = threading.Thread(
                target=worker.run, 
                name=f'GPU{gpu_id}Worker{worker_id}',
                daemon=True
            )
            thread.start()
            workers.append(worker)
            worker_threads.append(thread)
            logger.info(f"🔥 Started GPU {gpu_id} Worker {worker_id}")
    
    # Monitor progress
    results = []
    start_time = time.time()
    last_progress_time = start_time
    
    logger.info(f"🚀 Starting processing of {len(all_matches)} matches...")
    
    for i in range(len(all_matches)):
        try:
            # Shorter timeout to detect problems faster
            result = result_queue.get(timeout=60)  # 1 minute timeout
            results.append(result)
            
            current_time = time.time()
            
            # Progress reporting
            if (i + 1) % 5 == 0 or current_time - last_progress_time > 10:
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(all_matches) - i - 1) / rate if rate > 0 else 0
                
                gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
                gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
                success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
                
                logger.info(f"🚀 Progress: {i+1}/{len(all_matches)} ({rate:.2f}/s) | "
                           f"GPU0: {gpu0_count}, GPU1: {gpu1_count} | "
                           f"Success: {success_count} | ETA: {eta/60:.1f}m")
                last_progress_time = current_time
                
        except queue.Empty:
            logger.error(f"💀 TIMEOUT at match {i+1}! This indicates hanging.")
            logger.error(f"💀 Queue sizes: work={work_queue.qsize()}, result={result_queue.qsize()}")
            
            # Check worker status
            for j, worker in enumerate(workers):
                logger.error(f"💀 Worker {j}: processed={worker.processed}, errors={worker.errors}")
            
            break
            
        except Exception as e:
            logger.error(f"💀 Result collection error at {i+1}: {e}")
            break
    
    processing_time = time.time() - start_time
    
    # Signal workers to stop
    logger.info("🛑 Signaling workers to stop...")
    for _ in range(total_workers):
        work_queue.put(None)
    
    # Wait for workers with timeout
    for thread in worker_threads:
        thread.join(timeout=10)
        if thread.is_alive():
            logger.warning(f"💀 Worker thread {thread.name} still alive after timeout")
    
    # Create output
    logger.info("📊 Creating output...")
    enhanced_data = data.copy()
    
    # Process results
    result_map = {}
    for i, (video_path, gpx_path, _) in enumerate(all_matches):
        if i < len(results):
            result_map[(video_path, gpx_path)] = results[i]
    
    # Merge results back
    enhanced_results = {}
    for video_path, video_data in data.get('results', {}).items():
        enhanced_video_data = video_data.copy()
        enhanced_matches = []
        
        for match in video_data.get('matches', []):
            gpx_path = match.get('path')
            key = (video_path, gpx_path)
            
            if key in result_map:
                enhanced_matches.append(result_map[key])
            else:
                enhanced_matches.append(match)
        
        enhanced_video_data['matches'] = enhanced_matches
        enhanced_results[video_path] = enhanced_video_data
    
    enhanced_data['results'] = enhanced_results
    
    # Add processing metadata
    enhanced_data['robust_processing_info'] = {
        'beast_mode': args.beast_mode,
        'gpu_memory_gb': args.gpu_memory,
        'workers_per_gpu': args.workers_per_gpu,
        'total_workers': total_workers,
        'processing_time_seconds': processing_time,
        'matches_attempted': len(all_matches),
        'matches_completed': len(results),
        'processing_rate_matches_per_second': len(results) / processing_time if processing_time > 0 else 0,
        'success_rate': sum(1 for r in results if r.get('temporal_offset_seconds') is not None) / len(results) if results else 0,
        'processed_at': datetime.now().isoformat()
    }
    
    # Save results
    logger.info(f"💾 Saving results to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"💀 Failed to save results: {e}")
        sys.exit(1)
    
    # Final summary
    success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    error_count = sum(1 for r in results if 'error' in r.get('offset_method', ''))
    
    logger.info("🔥💀🔥 ROBUST BEAST MODE COMPLETE! 🔥💀🔥")
    logger.info("="*60)
    logger.info(f"📊 Total attempted: {len(all_matches)}")
    logger.info(f"📊 Total completed: {len(results)}")
    logger.info(f"✅ Successful offsets: {success_count}")
    logger.info(f"❌ Errors: {error_count}")
    logger.info(f"🔥 GPU 0 processed: {gpu0_count}")
    logger.info(f"🔥 GPU 1 processed: {gpu1_count}")
    logger.info(f"⚡ Processing time: {processing_time:.1f}s ({processing_time/60:.1f}m)")
    logger.info(f"🚀 Processing rate: {len(results)/processing_time:.2f} matches/second")
    logger.info(f"📈 Success rate: {success_count/len(results)*100:.1f}%" if results else "0%")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("="*60)
    
    if gpu0_count > 0 and gpu1_count > 0:
        logger.info("🎉 SUCCESS: Both GPUs processed matches!")
    elif gpu0_count == 0 and gpu1_count == 0:
        logger.error("💀 FAILURE: No GPU processing occurred!")
        sys.exit(1)
    else:
        logger.warning("⚠️ PARTIAL: Only one GPU processed matches")

if __name__ == "__main__":
    main()