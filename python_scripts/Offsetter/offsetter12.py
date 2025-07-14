#!/usr/bin/env python3
"""
SIMPLE FAST DUAL GPU - ACTUALLY WORKS
🔥 NO COMPLEXITY, NO HANGING, JUST FAST DUAL GPU PROCESSING 🔥
💀 BASED ON DIAGNOSTIC FINDINGS - THREADING WORKS! 💀
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

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('simple_fast_dual')

class SimpleFastWorker:
    """Simple fast worker - no hanging, just processing"""
    
    def __init__(self, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, gpu_memory_gb: float = 15.0):
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.gpu_memory_gb = gpu_memory_gb
        self.processed = 0
        self.logger = logging.getLogger(f'GPU{gpu_id}Worker')
        
    def run(self):
        """Simple worker loop"""
        
        self.logger.info(f"🔥 GPU {self.gpu_id} Worker STARTING")
        
        try:
            # Simple GPU initialization
            cp.cuda.Device(self.gpu_id).use()
            
            # Set memory limit if specified
            if self.gpu_memory_gb > 0:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.gpu_memory_gb * 1024**3))
                self.logger.info(f"🔥 GPU {self.gpu_id} memory limit: {self.gpu_memory_gb}GB")
            
            # Test GPU
            test = cp.array([1, 2, 3])
            result = cp.sum(test)
            self.logger.info(f"🔥 GPU {self.gpu_id} Worker READY - Test: {float(result)}")
            
            while True:
                try:
                    # Get work
                    work_item = self.work_queue.get(timeout=5)
                    
                    if work_item is None:
                        self.logger.info(f"🔥 GPU {self.gpu_id} Worker SHUTDOWN")
                        break
                    
                    video_path, gpx_path, match = work_item
                    
                    # Process quickly
                    result = self.process_simple_fast(video_path, gpx_path, match)
                    
                    # Send result
                    self.result_queue.put(result)
                    self.processed += 1
                    
                    if self.processed % 5 == 0:
                        self.logger.info(f"🔥 GPU {self.gpu_id}: {self.processed} processed")
                    
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"💀 GPU {self.gpu_id} error: {e}")
                    
                    # Send error result
                    error_result = match.copy()
                    error_result.update({
                        'temporal_offset_seconds': None,
                        'offset_confidence': 0.0,
                        'offset_method': f'gpu_{self.gpu_id}_error',
                        'gpu_processing': False,
                        'beast_mode': True  # Add beast_mode flag
                    })
                    self.result_queue.put(error_result)
                    self.work_queue.task_done()
        
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id} FATAL: {e}")
    
    def process_simple_fast(self, video_path: str, gpx_path: str, match: Dict) -> Dict:
        """Simple fast processing - no hanging"""
        
        result = match.copy()
        result.update({
            'gpu_id': self.gpu_id,
            'gpu_processing': True,
            'beast_mode': True,  # Add beast_mode flag
            'offset_method': f'simple_fast_gpu_{self.gpu_id}'
        })
        
        try:
            # Ensure GPU
            cp.cuda.Device(self.gpu_id).use()
            
            # Simple video processing
            video_motion = self.extract_video_simple(video_path)
            if video_motion is None:
                result['offset_method'] = f'gpu_{self.gpu_id}_video_failed'
                return result
            
            # Simple GPS processing  
            gps_speed = self.extract_gps_simple(gpx_path)
            if gps_speed is None:
                result['offset_method'] = f'gpu_{self.gpu_id}_gps_failed'
                return result
            
            # Simple offset calculation
            offset, confidence = self.calculate_offset_simple(video_motion, gps_speed)
            
            if offset is not None and confidence >= 0.25:
                result.update({
                    'temporal_offset_seconds': float(offset),
                    'offset_confidence': float(confidence),
                    'offset_method': f'simple_fast_gpu_{self.gpu_id}_success',
                    'sync_quality': 'good' if confidence >= 0.6 else 'fair'
                })
            else:
                result.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': float(confidence) if confidence else 0.0,
                    'offset_method': f'gpu_{self.gpu_id}_low_confidence'
                })
            
            return result
            
        except Exception as e:
            result.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'gpu_{self.gpu_id}_exception',
                'gpu_processing': False,
                'error': str(e)[:100]
            })
            return result
    
    def extract_video_simple(self, video_path: str) -> Optional[cp.ndarray]:
        """Simple video extraction - no hanging"""
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                cap.release()
                return None
            
            # Simple sampling - every 1 second
            frame_interval = max(1, int(fps))
            motion_values = []
            frame_idx = 0
            prev_gray = None
            
            # Limit frames to prevent hanging
            max_frames = 300  # Max 5 minutes at 1fps
            
            while len(motion_values) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Small resize for speed
                    frame = cv2.resize(frame, (320, 240))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_gray is not None:
                        # Simple motion on GPU
                        curr_gpu = cp.array(gray, dtype=cp.float32)
                        prev_gpu = cp.array(prev_gray, dtype=cp.float32)
                        
                        diff = cp.abs(curr_gpu - prev_gpu)
                        motion = float(cp.mean(diff))
                        motion_values.append(motion)
                        
                        del curr_gpu, prev_gpu, diff
                    
                    prev_gray = gray
                
                frame_idx += 1
            
            cap.release()
            
            if len(motion_values) >= 3:
                return cp.array(motion_values, dtype=cp.float32)
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Video extraction error: {e}")
            return None
    
    def extract_gps_simple(self, gpx_path: str) -> Optional[cp.ndarray]:
        """Simple GPS extraction - no hanging"""
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time:
                            points.append({
                                'lat': point.latitude,
                                'lon': point.longitude,
                                'time': point.time
                            })
            
            if len(points) < 10:
                return None
            
            points.sort(key=lambda p: p['time'])
            
            # Limit points to prevent hanging
            if len(points) > 1000:
                points = points[::len(points)//1000]
            
            df = pd.DataFrame(points)
            
            # Simple distance calculation on GPU
            lats = cp.array(df['lat'].values, dtype=cp.float32)
            lons = cp.array(df['lon'].values, dtype=cp.float32)
            
            # Simple distance approximation (fast)
            lat_diffs = lats[1:] - lats[:-1]
            lon_diffs = lons[1:] - lons[:-1]
            distances = cp.sqrt(lat_diffs**2 + lon_diffs**2) * 111000  # Rough conversion to meters
            
            # Time differences
            time_diffs = cp.array([
                (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds()
                for i in range(len(df)-1)
            ], dtype=cp.float32)
            
            # Simple speed
            speeds = cp.where(time_diffs > 0, distances / time_diffs, 0)
            
            # Simple resampling
            if len(speeds) > 100:
                # Downsample for speed
                step = len(speeds) // 100
                speeds = speeds[::step]
            
            return speeds
            
        except Exception as e:
            self.logger.debug(f"GPS extraction error: {e}")
            return None
    
    def calculate_offset_simple(self, video_motion: cp.ndarray, gps_speed: cp.ndarray) -> Tuple[Optional[float], float]:
        """Simple offset calculation - no hanging"""
        
        try:
            # Simple normalization
            video_norm = (video_motion - cp.mean(video_motion)) / (cp.std(video_motion) + 1e-8)
            gps_norm = (gps_speed - cp.mean(gps_speed)) / (cp.std(gps_speed) + 1e-8)
            
            # Limit correlation size to prevent hanging
            max_len = min(len(video_norm), len(gps_norm), 200)
            video_short = video_norm[:max_len]
            gps_short = gps_norm[:max_len]
            
            # Simple correlation
            best_offset = None
            best_confidence = 0.0
            
            # Try different offsets
            for offset in range(-min(50, max_len//2), min(50, max_len//2) + 1):
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
                
                if len(v_seg) > 5:
                    # Simple correlation coefficient
                    mean_v = cp.mean(v_seg)
                    mean_g = cp.mean(g_seg)
                    
                    num = cp.sum((v_seg - mean_v) * (g_seg - mean_g))
                    den = cp.sqrt(cp.sum((v_seg - mean_v)**2) * cp.sum((g_seg - mean_g)**2))
                    
                    if den > 0:
                        corr = float(num / den)
                        
                        if abs(corr) > best_confidence:
                            best_confidence = abs(corr)
                            best_offset = float(offset)
            
            return best_offset, best_confidence
            
        except Exception as e:
            self.logger.debug(f"Offset calculation error: {e}")
            return None, 0.0

def main():
    """Simple fast dual GPU main"""
    
    parser = argparse.ArgumentParser(description='🔥 Simple Fast Dual GPU - Actually Works')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Workers per GPU')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    # ADD MISSING ARGUMENTS
    parser.add_argument('--gpu-memory', type=float, default=15.0, help='GPU memory limit per GPU in GB')
    parser.add_argument('--beast-mode', action='store_true', help='Enable beast mode processing')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    
    if args.beast_mode:
        logger.info("🔥💀🔥 BEAST MODE ACTIVATED! 🔥💀🔥")
    
    input_file = Path(args.input_file)
    output_file = Path(args.output) if args.output else input_file.parent / f"simple_fast_{input_file.name}"
    
    # Check GPU availability
    try:
        import cupy as cp
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"🔥 Detected {gpu_count} CUDA GPUs")
        
        if gpu_count < 2:
            logger.warning(f"⚠️ Only {gpu_count} GPUs detected, expected 2")
            
    except Exception as e:
        logger.error(f"💀 GPU check failed: {e}")
        sys.exit(1)
    
    # Load data
    logger.info(f"📁 Loading {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Collect matches
    all_matches = []
    for video_path, video_data in data.get('results', {}).items():
        for match in video_data.get('matches', []):
            if match.get('combined_score', 0) >= args.min_score:
                all_matches.append((video_path, match['path'], match))
                if args.limit and len(all_matches) >= args.limit:
                    break
        if args.limit and len(all_matches) >= args.limit:
            break
    
    total_matches = len(all_matches)
    logger.info(f"🔥 Processing {total_matches} matches - SIMPLE AND FAST")
    
    if args.beast_mode:
        logger.info(f"🔥 BEAST MODE: Using {args.gpu_memory}GB per GPU")
    
    # Create queues
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Add work
    for match in all_matches:
        work_queue.put(match)
    
    # Create workers
    workers = []
    total_workers = len([0, 1]) * args.workers_per_gpu
    
    for gpu_id in [0, 1]:
        for worker_id in range(args.workers_per_gpu):
            worker = SimpleFastWorker(gpu_id, work_queue, result_queue, args.gpu_memory)
            thread = threading.Thread(target=worker.run, name=f'GPU{gpu_id}Worker{worker_id}')
            thread.start()
            workers.append(thread)
    
    logger.info(f"🔥 Started {total_workers} simple fast workers")
    
    # Collect results
    results = []
    start_time = time.time()
    
    for i in range(total_matches):
        try:
            result = result_queue.get(timeout=120)  # 2 minute timeout per match
            results.append(result)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(f"🔥 Progress: {i+1}/{total_matches} ({rate:.1f}/s)")
                
        except queue.Empty:
            logger.error(f"💀 Timeout at {i+1}")
            break
        except Exception as e:
            logger.error(f"💀 Collection error: {e}")
            break
    
    # Signal workers to stop
    for _ in range(total_workers):
        work_queue.put(None)
    
    # Wait for workers
    for thread in workers:
        thread.join(timeout=30)
    
    processing_time = time.time() - start_time
    
    # Create output
    enhanced_data = data.copy()
    
    # Merge results
    result_map = {}
    for i, (video_path, gpx_path, _) in enumerate(all_matches):
        if i < len(results):
            result_map[(video_path, gpx_path)] = results[i]
    
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
    enhanced_data['processing_info'] = {
        'beast_mode': args.beast_mode,
        'gpu_memory_gb': args.gpu_memory,
        'workers_per_gpu': args.workers_per_gpu,
        'processing_time_seconds': processing_time,
        'processing_rate_matches_per_second': len(results) / processing_time if processing_time > 0 else 0,
        'processed_at': datetime.now().isoformat()
    }
    
    # Save
    logger.info(f"💾 Saving to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    
    # Summary
    success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    
    logger.info("🔥 SIMPLE FAST DUAL GPU COMPLETE 🔥")
    logger.info(f"📊 Processed: {len(results)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"🔥 GPU 0: {gpu0_count}")
    logger.info(f"🔥 GPU 1: {gpu1_count}")
    logger.info(f"⚡ Time: {processing_time:.1f}s")
    logger.info(f"🚀 Rate: {len(results)/processing_time:.1f}/s")
    
    if args.beast_mode:
        logger.info("🔥💀🔥 BEAST MODE COMPLETE! 🔥💀🔥")

if __name__ == "__main__":
    main()