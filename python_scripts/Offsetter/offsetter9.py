#!/usr/bin/env python3
"""
FAST DUAL GPU MULTIPROCESSING - ACTUALLY WORKS
🔥 NO THREADING BULLSHIT, JUST MULTIPROCESSING 🔥
💀 SEPARATE PROCESSES = SEPARATE GPU CONTEXTS 💀
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
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('fast_dual_gpu')

def gpu_worker_process(gpu_id: int, matches_queue: mp.Queue, results_queue: mp.Queue, 
                      max_memory_gb: float, worker_id: int):
    """GPU worker process - completely separate process per GPU"""
    
    # Setup logging for this process
    logger = setup_logging()
    logger.info(f"🔥 GPU {gpu_id} Worker {worker_id} STARTING")
    
    try:
        # Initialize GPU in this process
        cp.cuda.Device(gpu_id).use()
        
        # Set memory limit
        memory_pool = cp.get_default_memory_pool()
        memory_pool.set_limit(size=int(max_memory_gb * 1024**3))
        
        # Test GPU
        test_array = cp.random.rand(1000, 1000)
        test_result = cp.sum(test_array)
        del test_array
        
        logger.info(f"🔥 GPU {gpu_id} Worker {worker_id} READY - Test: {float(test_result):.2f}")
        
        processed_count = 0
        
        while True:
            try:
                # Get work from queue
                work_item = matches_queue.get(timeout=10)
                
                if work_item is None:  # Shutdown signal
                    logger.info(f"🔥 GPU {gpu_id} Worker {worker_id} SHUTDOWN")
                    break
                
                video_path, gpx_path, match = work_item
                
                # Process this match
                result = process_single_match_fast(video_path, gpx_path, match, gpu_id)
                
                # Send result back
                results_queue.put(result)
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"🔥 GPU {gpu_id} Worker {worker_id}: {processed_count} processed")
                
            except Exception as e:
                logger.error(f"💀 GPU {gpu_id} Worker {worker_id} error: {e}")
                # Send error result
                error_result = match.copy()
                error_result.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': 0.0,
                    'offset_method': f'gpu_{gpu_id}_worker_{worker_id}_error',
                    'gpu_processing': False,
                    'error': str(e)[:100]
                })
                results_queue.put(error_result)
        
        # Cleanup
        memory_pool.free_all_blocks()
        cp.cuda.Device().synchronize()
        
        logger.info(f"🔥 GPU {gpu_id} Worker {worker_id} COMPLETE - {processed_count} processed")
        
    except Exception as e:
        logger.error(f"💀 GPU {gpu_id} Worker {worker_id} FAILED: {e}")

def process_single_match_fast(video_path: str, gpx_path: str, match: Dict, gpu_id: int) -> Dict:
    """Process single match with maximum speed"""
    
    result = match.copy()
    result.update({
        'temporal_offset_seconds': None,
        'offset_confidence': 0.0,
        'offset_method': f'gpu_{gpu_id}_fast',
        'gpu_processing': True,
        'gpu_id': gpu_id
    })
    
    try:
        # Force GPU context
        cp.cuda.Device(gpu_id).use()
        
        # Process video - FAST VERSION
        video_motion = extract_video_motion_fast(video_path, gpu_id)
        if video_motion is None:
            result['offset_method'] = f'gpu_{gpu_id}_video_failed'
            return result
        
        # Process GPX - FAST VERSION
        gps_speed = extract_gps_speed_fast(gpx_path, gpu_id)
        if gps_speed is None:
            result['offset_method'] = f'gpu_{gpu_id}_gps_failed'
            return result
        
        # Calculate offset - FAST VERSION
        offset, confidence = calculate_offset_fast(video_motion, gps_speed, gpu_id)
        
        if offset is not None and confidence >= 0.3:
            result.update({
                'temporal_offset_seconds': offset,
                'offset_confidence': confidence,
                'offset_method': f'gpu_{gpu_id}_fast_success',
                'sync_quality': 'excellent' if confidence >= 0.8 else 'good' if confidence >= 0.6 else 'fair'
            })
        else:
            result['offset_method'] = f'gpu_{gpu_id}_low_confidence'
        
        return result
        
    except Exception as e:
        result.update({
            'offset_method': f'gpu_{gpu_id}_exception',
            'gpu_processing': False,
            'error': str(e)[:100]
        })
        return result

def extract_video_motion_fast(video_path: str, gpu_id: int) -> Optional[np.ndarray]:
    """Extract video motion - FAST VERSION"""
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return None
        
        # FAST sampling - every 1 second
        frame_interval = max(1, int(fps))
        
        motion_values = []
        frame_idx = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize to small size for speed
                frame = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Simple frame difference on GPU
                    curr_gpu = cp.array(gray, dtype=cp.float32)
                    prev_gpu = cp.array(prev_frame, dtype=cp.float32)
                    
                    diff = cp.abs(curr_gpu - prev_gpu)
                    motion = float(cp.mean(diff))
                    motion_values.append(motion)
                
                prev_frame = gray
            
            frame_idx += 1
        
        cap.release()
        
        if len(motion_values) >= 3:
            return np.array(motion_values)
        else:
            return None
            
    except Exception:
        return None

def extract_gps_speed_fast(gpx_path: str, gpu_id: int) -> Optional[np.ndarray]:
    """Extract GPS speed - FAST VERSION"""
    
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
        
        # GPU vectorized distance calculation
        lats = cp.array(df['lat'].values, dtype=cp.float64)
        lons = cp.array(df['lon'].values, dtype=cp.float64)
        
        # Haversine distance - vectorized
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
        
        # Simple resampling - every 1 second
        duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        target_times = cp.arange(0, duration, 1.0)
        time_offsets = cp.cumsum(cp.concatenate([cp.array([0]), time_diffs]))
        
        resampled_speeds = cp.interp(target_times, time_offsets[:-1], speeds)
        
        return cp.asnumpy(resampled_speeds)
        
    except Exception:
        return None

def calculate_offset_fast(video_motion: np.ndarray, gps_speed: np.ndarray, gpu_id: int) -> Tuple[Optional[float], float]:
    """Calculate offset - FAST VERSION"""
    
    try:
        # Upload to GPU
        video_gpu = cp.array(video_motion, dtype=cp.float32)
        gps_gpu = cp.array(gps_speed, dtype=cp.float32)
        
        # Simple normalization
        video_norm = (video_gpu - cp.mean(video_gpu)) / (cp.std(video_gpu) + 1e-8)
        gps_norm = (gps_gpu - cp.mean(gps_gpu)) / (cp.std(gps_gpu) + 1e-8)
        
        # Cross-correlation using GPU FFT
        pad_len = len(video_norm) + len(gps_norm) - 1
        pad_len = 1 << (pad_len - 1).bit_length()
        
        video_padded = cp.pad(video_norm, (0, pad_len - len(video_norm)))
        gps_padded = cp.pad(gps_norm, (0, pad_len - len(gps_norm)))
        
        video_fft = cp.fft.fft(video_padded)
        gps_fft = cp.fft.fft(gps_padded)
        
        correlation = cp.fft.ifft(cp.conj(video_fft) * gps_fft).real
        
        # Find peak
        peak_idx = cp.argmax(correlation)
        confidence = float(correlation[peak_idx] / len(video_norm))
        
        # Convert to offset
        offset_samples = float(peak_idx - len(video_norm) + 1)
        offset_seconds = offset_samples * 1.0  # 1 second sampling
        
        if abs(offset_seconds) > 600:  # 10 minutes max
            return None, 0.0
        
        return offset_seconds, abs(confidence)
        
    except Exception:
        return None, 0.0

def main():
    """FAST dual GPU main using multiprocessing"""
    
    parser = argparse.ArgumentParser(description='🔥 FAST Dual GPU Multiprocessing')
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
    output_file = Path(args.output) if args.output else input_file.parent / f"fast_dual_gpu_{input_file.name}"
    
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
    logger.info(f"🔥 Processing {total_matches} matches")
    
    # Create queues
    matches_queue = mp.Queue()
    results_queue = mp.Queue()
    
    # Add all matches to queue
    for match in all_matches:
        matches_queue.put(match)
    
    # Create workers
    workers = []
    total_workers = len([0, 1]) * args.workers_per_gpu
    
    for gpu_id in [0, 1]:
        for worker_id in range(args.workers_per_gpu):
            # Add shutdown signal
            matches_queue.put(None)
            
            # Create worker process
            worker = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, matches_queue, results_queue, args.max_gpu_memory, worker_id)
            )
            worker.start()
            workers.append(worker)
    
    logger.info(f"🔥 Started {total_workers} workers")
    
    # Collect results
    results = []
    start_time = time.time()
    
    for i in range(total_matches):
        try:
            result = results_queue.get(timeout=300)  # 5 minute timeout
            results.append(result)
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(f"🔥 Progress: {i+1}/{total_matches} ({rate:.1f}/s)")
                
        except Exception as e:
            logger.error(f"💀 Result collection error: {e}")
            break
    
    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate()
    
    processing_time = time.time() - start_time
    
    # Create output
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
    
    logger.info("🔥🔥🔥 FAST DUAL GPU COMPLETE 🔥🔥🔥")
    logger.info(f"📊 Processed: {len(results)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"⚡ Time: {processing_time:.1f}s")
    logger.info(f"🚀 Rate: {len(results)/processing_time:.1f} matches/second")

if __name__ == "__main__":
    main()