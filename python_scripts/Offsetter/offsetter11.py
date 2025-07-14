#!/usr/bin/env python3
"""
FIXED THREADING DUAL GPU - ACTUALLY WORKS
🔥 DIAGNOSTIC PROVED THREADING WORKS - JUST NEEDED PROPER IMPLEMENTATION 🔥
💀 BOTH GPUS WORKING CONCURRENTLY IN THREADS 💀
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

# Global locks to ensure proper GPU context switching
gpu_locks = {0: threading.Lock(), 1: threading.Lock()}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('fixed_dual_gpu')

class FixedGPUWorker:
    """Fixed GPU worker that properly handles GPU contexts in threads"""
    
    def __init__(self, gpu_id: int, work_queue: queue.Queue, result_queue: queue.Queue, 
                 max_memory_gb: float):
        self.gpu_id = gpu_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.max_memory_gb = max_memory_gb
        self.processed_count = 0
        self.logger = logging.getLogger(f'gpu_{gpu_id}_worker')
        
    def run(self):
        """Main worker loop"""
        
        self.logger.info(f"🔥 GPU {self.gpu_id} worker STARTING")
        
        try:
            # Initialize GPU context in this thread
            with gpu_locks[self.gpu_id]:
                cp.cuda.Device(self.gpu_id).use()
                
                # Set memory limit
                memory_pool = cp.get_default_memory_pool()
                memory_pool.set_limit(size=int(self.max_memory_gb * 1024**3))
                
                # Test GPU
                test_array = cp.random.rand(1000, 1000)
                test_result = cp.sum(test_array)
                del test_array
                
                self.logger.info(f"🔥 GPU {self.gpu_id} worker READY - Test: {float(test_result):.2f}")
            
            # Process work
            while True:
                try:
                    # Get work item
                    work_item = self.work_queue.get(timeout=5)
                    
                    if work_item is None:  # Shutdown signal
                        self.logger.info(f"🔥 GPU {self.gpu_id} worker SHUTDOWN")
                        break
                    
                    video_path, gpx_path, match = work_item
                    
                    # Process with proper GPU context
                    result = self._process_match_fixed(video_path, gpx_path, match)
                    
                    # Send result
                    self.result_queue.put(result)
                    
                    self.processed_count += 1
                    
                    if self.processed_count % 10 == 0:
                        self.logger.info(f"🔥 GPU {self.gpu_id}: {self.processed_count} processed")
                    
                    # Mark work done
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"💀 GPU {self.gpu_id} work error: {e}")
                    
                    # Send error result
                    error_result = match.copy()
                    error_result.update({
                        'temporal_offset_seconds': None,
                        'offset_confidence': 0.0,
                        'offset_method': f'gpu_{self.gpu_id}_error',
                        'gpu_processing': False,
                        'error': str(e)[:100]
                    })
                    self.result_queue.put(error_result)
                    self.work_queue.task_done()
            
            # Cleanup
            with gpu_locks[self.gpu_id]:
                memory_pool = cp.get_default_memory_pool()
                memory_pool.free_all_blocks()
                cp.cuda.Device().synchronize()
            
            self.logger.info(f"🔥 GPU {self.gpu_id} worker COMPLETE - {self.processed_count} processed")
            
        except Exception as e:
            self.logger.error(f"💀 GPU {self.gpu_id} worker FAILED: {e}")
    
    def _process_match_fixed(self, video_path: str, gpx_path: str, match: Dict) -> Dict:
        """Process match with proper GPU context handling"""
        
        result = match.copy()
        result.update({
            'temporal_offset_seconds': None,
            'offset_confidence': 0.0,
            'offset_method': f'gpu_{self.gpu_id}_fixed',
            'gpu_processing': True,
            'gpu_id': self.gpu_id
        })
        
        try:
            # Ensure GPU context with lock
            with gpu_locks[self.gpu_id]:
                cp.cuda.Device(self.gpu_id).use()
                
                # Process video
                video_motion = self._extract_video_motion_fixed(video_path)
                if video_motion is None:
                    result['offset_method'] = f'gpu_{self.gpu_id}_video_failed'
                    return result
                
                # Process GPX
                gps_speed = self._extract_gps_speed_fixed(gpx_path)
                if gps_speed is None:
                    result['offset_method'] = f'gpu_{self.gpu_id}_gps_failed'
                    return result
                
                # Calculate offset
                offset, confidence = self._calculate_offset_fixed(video_motion, gps_speed)
                
                if offset is not None and confidence >= 0.3:
                    result.update({
                        'temporal_offset_seconds': offset,
                        'offset_confidence': confidence,
                        'offset_method': f'gpu_{self.gpu_id}_fixed_success',
                        'sync_quality': 'excellent' if confidence >= 0.8 else 'good' if confidence >= 0.6 else 'fair'
                    })
                else:
                    result['offset_method'] = f'gpu_{self.gpu_id}_low_confidence'
            
            return result
            
        except Exception as e:
            result.update({
                'offset_method': f'gpu_{self.gpu_id}_exception',
                'gpu_processing': False,
                'error': str(e)[:100]
            })
            return result
    
    def _extract_video_motion_fixed(self, video_path: str) -> Optional[cp.ndarray]:
        """Extract video motion with proper GPU context"""
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                cap.release()
                return None
            
            # Fast sampling - every 1 second
            frame_interval = max(1, int(fps))
            
            motion_values = []
            frame_idx = 0
            prev_frame = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Small resize for speed
                    frame = cv2.resize(frame, (320, 240))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        # GPU processing with proper context
                        curr_gpu = cp.array(gray, dtype=cp.float32)
                        prev_gpu = cp.array(prev_frame, dtype=cp.float32)
                        
                        # Simple motion calculation
                        diff = cp.abs(curr_gpu - prev_gpu)
                        motion = float(cp.mean(diff))
                        motion_values.append(motion)
                        
                        del curr_gpu, prev_gpu, diff
                    
                    prev_frame = gray
                
                frame_idx += 1
            
            cap.release()
            
            if len(motion_values) >= 3:
                return cp.array(motion_values, dtype=cp.float32)
            else:
                return None
                
        except Exception:
            return None
    
    def _extract_gps_speed_fixed(self, gpx_path: str) -> Optional[cp.ndarray]:
        """Extract GPS speed with proper GPU context"""
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    points.extend([{
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'time': point.time
                    } for point in segment.points if point.time])
            
            if len(points) < 10:
                return None
            
            points.sort(key=lambda p: p['time'])
            df = pd.DataFrame(points)
            
            # GPU processing
            lats = cp.array(df['lat'].values, dtype=cp.float64)
            lons = cp.array(df['lon'].values, dtype=cp.float64)
            
            # Vectorized distance calculation
            lat1, lat2 = lats[:-1], lats[1:]
            lon1, lon2 = lons[:-1], lons[1:]
            
            dlat = cp.radians(lat2 - lat1)
            dlon = cp.radians(lon2 - lon1)
            lat1_rad = cp.radians(lat1)
            lat2_rad = cp.radians(lat2)
            
            a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
            distances = 6371000 * 2 * cp.arcsin(cp.sqrt(a))
            
            # Time differences
            time_diffs = cp.array([
                (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds()
                for i in range(len(df)-1)
            ])
            
            # Speed calculation
            speeds = cp.where(time_diffs > 0, distances / time_diffs, 0)
            
            # Simple resampling
            duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
            target_times = cp.arange(0, duration, 1.0)
            time_offsets = cp.cumsum(cp.concatenate([cp.array([0]), time_diffs]))
            
            resampled_speeds = cp.interp(target_times, time_offsets[:-1], speeds)
            
            return resampled_speeds
            
        except Exception:
            return None
    
    def _calculate_offset_fixed(self, video_motion: cp.ndarray, gps_speed: cp.ndarray) -> Tuple[Optional[float], float]:
        """Calculate offset with proper GPU context"""
        
        try:
            # Normalize signals
            video_norm = (video_motion - cp.mean(video_motion)) / (cp.std(video_motion) + 1e-8)
            gps_norm = (gps_speed - cp.mean(gps_speed)) / (cp.std(gps_speed) + 1e-8)
            
            # Cross-correlation using simple correlation (avoid CUBLAS issues)
            # Use simple sliding window correlation instead of FFT
            max_offset = min(len(video_norm), len(gps_norm), 600)  # Max 600 second offset
            
            best_offset = None
            best_confidence = 0.0
            
            for offset in range(-max_offset, max_offset + 1):
                if offset < 0:
                    # GPS ahead
                    v_seg = video_norm[-offset:]
                    g_seg = gps_norm[:len(v_seg)]
                elif offset > 0:
                    # Video ahead
                    g_seg = gps_speed[offset:]
                    v_seg = video_motion[:len(g_seg)]
                else:
                    # No offset
                    min_len = min(len(video_norm), len(gps_norm))
                    v_seg = video_norm[:min_len]
                    g_seg = gps_norm[:min_len]
                
                if len(v_seg) > 10 and len(g_seg) > 10:
                    # Simple correlation coefficient
                    correlation = float(cp.corrcoef(v_seg, g_seg)[0, 1])
                    
                    if not cp.isnan(correlation) and abs(correlation) > best_confidence:
                        best_confidence = abs(correlation)
                        best_offset = float(offset)
            
            if best_offset is not None and best_confidence >= 0.3:
                return best_offset, best_confidence
            else:
                return None, 0.0
                
        except Exception:
            return None, 0.0

def main():
    """Fixed dual GPU main using proper threading"""
    
    parser = argparse.ArgumentParser(description='🔥 FIXED Threading Dual GPU')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--max-gpu-memory', type=float, default=15.0, help='Max GPU memory per GPU')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='Workers per GPU')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output) if args.output else input_file.parent / f"fixed_dual_gpu_{input_file.name}"
    
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
    logger.info(f"🔥 Processing {total_matches} matches with FIXED threading")
    
    # Create queues
    work_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Add work to queue
    for match in all_matches:
        work_queue.put(match)
    
    # Create and start workers
    workers = []
    total_workers = len([0, 1]) * args.workers_per_gpu
    
    for gpu_id in [0, 1]:
        for worker_id in range(args.workers_per_gpu):
            worker = FixedGPUWorker(gpu_id, work_queue, result_queue, args.max_gpu_memory)
            thread = threading.Thread(target=worker.run, name=f'GPU{gpu_id}Worker{worker_id}')
            thread.start()
            workers.append(thread)
    
    logger.info(f"🔥 Started {total_workers} worker threads")
    
    # Collect results
    results = []
    start_time = time.time()
    
    for i in range(total_matches):
        try:
            result = result_queue.get(timeout=300)  # 5 minute timeout
            results.append(result)
            
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(f"🔥 Progress: {i+1}/{total_matches} ({rate:.1f}/s)")
                
        except queue.Empty:
            logger.error(f"💀 Timeout waiting for result {i+1}")
            break
        except Exception as e:
            logger.error(f"💀 Result collection error: {e}")
            break
    
    # Signal workers to stop
    for _ in range(total_workers):
        work_queue.put(None)
    
    # Wait for workers to finish
    for thread in workers:
        thread.join(timeout=30)
    
    processing_time = time.time() - start_time
    
    # Create output - same merging logic as before
    enhanced_data = data.copy()
    
    # Merge results back
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
    
    # Save
    logger.info(f"💾 Saving to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    
    # Summary
    success_count = sum(1 for r in results if r.get('temporal_offset_seconds') is not None)
    gpu0_count = sum(1 for r in results if r.get('gpu_id') == 0)
    gpu1_count = sum(1 for r in results if r.get('gpu_id') == 1)
    
    logger.info("🔥🔥🔥 FIXED DUAL GPU COMPLETE 🔥🔥🔥")
    logger.info(f"📊 Processed: {len(results)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"🔥 GPU 0 processed: {gpu0_count}")
    logger.info(f"🔥 GPU 1 processed: {gpu1_count}")
    logger.info(f"⚡ Time: {processing_time:.1f}s")
    logger.info(f"🚀 Rate: {len(results)/processing_time:.1f} matches/second")

if __name__ == "__main__":
    main()