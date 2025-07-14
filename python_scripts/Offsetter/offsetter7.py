#!/usr/bin/env python3
"""
FIXED SIMPLE DUAL GPU - ACTUALLY WORKS
🔥 NO PYTORCH COMPLEXITY, JUST RAW GPU POWER 🔥
💀 CUPY ONLY FOR MAXIMUM SPEED 💀
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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_dual_gpu')

class SimpleGPUProcessor:
    """Simple GPU processor using CuPy only - NO PYTORCH COMPLEXITY"""
    
    def __init__(self, gpu_id: int, max_memory_gb: float = 15.0):
        self.gpu_id = gpu_id
        
        # Initialize CuPy on this GPU
        cp.cuda.Device(gpu_id).use()
        
        # Set memory limits
        memory_pool = cp.get_default_memory_pool()
        memory_pool.set_limit(size=int(max_memory_gb * 1024**3))
        
        logger.info(f"🔥 GPU {gpu_id} initialized with {max_memory_gb}GB limit")
    
    def process_matches_batch(self, matches_batch: List[Tuple]) -> List[Dict]:
        """Process entire batch of matches on this GPU"""
        
        logger.info(f"🔥 GPU {self.gpu_id}: Processing {len(matches_batch)} matches")
        
        results = []
        processed = 0
        
        for video_path, gpx_path, match in matches_batch:
            try:
                # Process video and GPS on GPU
                video_data = self._process_video_simple(video_path)
                gps_data = self._process_gpx_simple(gpx_path)
                
                if video_data is not None and gps_data is not None:
                    # Calculate offset on GPU
                    offset_result = self._calculate_offset_simple(video_data, gps_data)
                    
                    # Create enhanced match
                    enhanced_match = match.copy()
                    enhanced_match.update(offset_result)
                    enhanced_match['gpu_id'] = self.gpu_id
                    
                    results.append(enhanced_match)
                else:
                    # Processing failed
                    enhanced_match = match.copy()
                    enhanced_match.update({
                        'temporal_offset_seconds': None,
                        'offset_confidence': 0.0,
                        'offset_method': f'gpu_{self.gpu_id}_processing_failed',
                        'gpu_processing': False
                    })
                    results.append(enhanced_match)
                
                processed += 1
                
                # Log progress
                if processed % 25 == 0:
                    logger.info(f"🔥 GPU {self.gpu_id}: Processed {processed}/{len(matches_batch)}")
                
            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: Match processing failed: {e}")
                
                enhanced_match = match.copy()
                enhanced_match.update({
                    'temporal_offset_seconds': None,
                    'offset_confidence': 0.0,
                    'offset_method': f'gpu_{self.gpu_id}_error',
                    'gpu_processing': False,
                    'error': str(e)[:100]
                })
                results.append(enhanced_match)
        
        logger.info(f"🔥 GPU {self.gpu_id}: Batch complete - {processed} processed")
        return results
    
    def _process_video_simple(self, video_path: str) -> Optional[Dict]:
        """Simple video processing using CuPy only"""
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Get properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0 or frame_count <= 0:
            cap.release()
            return None
        
        duration = frame_count / fps
        if duration < 5.0:
            cap.release()
            return None
        
        # Simple processing - sample every 0.5 seconds
        frame_interval = max(1, int(fps * 0.5))  # Every 0.5 seconds
        
        motion_values = []
        timestamps = []
        
        frame_idx = 0
        prev_frame_gpu = None
        
        # Create large batch arrays on GPU for efficiency
        frame_batch = []
        batch_size = 50  # Process 50 frames at once
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize for efficiency
                if frame.shape[1] > 640:
                    frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
                
                frame_batch.append((frame, frame_idx / fps))
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    batch_motion = self._process_frame_batch_cupy(frame_batch)
                    motion_values.extend(batch_motion)
                    timestamps.extend([ts for _, ts in frame_batch])
                    frame_batch = []
            
            frame_idx += 1
        
        # Process remaining frames
        if frame_batch:
            batch_motion = self._process_frame_batch_cupy(frame_batch)
            motion_values.extend(batch_motion)
            timestamps.extend([ts for _, ts in frame_batch])
        
        cap.release()
        
        if len(motion_values) < 3:
            return None
        
        # Store on GPU
        motion_gpu = cp.array(motion_values, dtype=cp.float32)
        timestamps_gpu = cp.array(timestamps, dtype=cp.float32)
        
        return {
            'motion_magnitude': motion_gpu,
            'timestamps': timestamps_gpu,
            'duration': duration,
            'fps': fps,
            'frame_count': len(motion_values),
            'gpu_id': self.gpu_id
        }
    
    def _process_frame_batch_cupy(self, frame_batch: List[Tuple]) -> List[float]:
        """Process frame batch using CuPy only - NO PYTORCH"""
        
        if len(frame_batch) <= 1:
            return [0.0] * len(frame_batch)
        
        # Convert frames to CuPy arrays
        frames_gpu = []
        
        for frame, _ in frame_batch:
            # Convert to grayscale and upload to GPU
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_gpu = cp.array(gray, dtype=cp.float32)
            
            # Simple Gaussian blur using CuPy convolution
            blurred_gpu = self._gaussian_blur_cupy(gray_gpu)
            frames_gpu.append(blurred_gpu)
        
        # Calculate motion between consecutive frames
        motion_values = [0.0]  # First frame has no motion
        
        for i in range(1, len(frames_gpu)):
            # Simple frame difference
            diff = cp.abs(frames_gpu[i] - frames_gpu[i-1])
            
            # Calculate motion magnitude using various methods
            motion_mag = float(cp.mean(diff))
            motion_energy = float(cp.sum(diff ** 2))
            motion_max = float(cp.max(diff))
            
            # Combine metrics for robust motion detection
            combined_motion = motion_mag + 0.1 * motion_energy / 1000000 + 0.1 * motion_max
            motion_values.append(combined_motion)
        
        return motion_values
    
    def _gaussian_blur_cupy(self, image_gpu: cp.ndarray, kernel_size: int = 5) -> cp.ndarray:
        """Simple Gaussian blur using CuPy convolution"""
        
        # Create simple averaging kernel (approximate Gaussian)
        kernel = cp.ones((kernel_size, kernel_size), dtype=cp.float32) / (kernel_size ** 2)
        
        # Pad image
        pad_size = kernel_size // 2
        padded = cp.pad(image_gpu, pad_size, mode='edge')
        
        # Simple convolution using sliding window
        output = cp.zeros_like(image_gpu)
        
        for i in range(image_gpu.shape[0]):
            for j in range(image_gpu.shape[1]):
                # Extract window and apply kernel
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = cp.sum(window * kernel)
        
        return output
    
    def _process_gpx_simple(self, gpx_path: str) -> Optional[Dict]:
        """Simple GPX processing using CuPy"""
        
        try:
            # Load GPX
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            # Extract points
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
            
            duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
            if duration < 10.0:
                return None
            
            # GPU processing with CuPy - MASSIVE ARRAYS
            lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)
            lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
            
            # Massive vectorized distance calculations
            lat1, lat2 = lats_gpu[:-1], lats_gpu[1:]
            lon1, lon2 = lons_gpu[:-1], lons_gpu[1:]
            
            # Convert to radians in batch
            lat1_rad = cp.radians(lat1)
            lat2_rad = cp.radians(lat2)
            lon1_rad = cp.radians(lon1)
            lon2_rad = cp.radians(lon2)
            
            # Haversine formula - fully vectorized
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
            distances = 6371000 * 2 * cp.arcsin(cp.sqrt(a))
            
            # Time differences
            time_diffs = cp.array([
                (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds() 
                for i in range(len(df)-1)
            ], dtype=cp.float32)
            
            # Speed calculations - all on GPU
            speeds = cp.where(time_diffs > 0, distances / time_diffs, 0)
            
            # Acceleration calculations
            accelerations = cp.zeros_like(speeds)
            accelerations[1:] = cp.where(
                time_diffs[1:] > 0,
                (speeds[1:] - speeds[:-1]) / time_diffs[1:],
                0
            )
            
            # Resample to 1Hz using GPU interpolation
            time_offsets = cp.cumsum(cp.concatenate([cp.array([0]), time_diffs]))
            target_times = cp.arange(0, duration, 1.0, dtype=cp.float32)
            
            # Massive GPU interpolation
            resampled_speed = cp.interp(target_times, time_offsets[:-1], speeds)
            resampled_accel = cp.interp(target_times, time_offsets[:-1], accelerations)
            
            return {
                'speed': resampled_speed,
                'acceleration': resampled_accel,
                'time_offsets': target_times,
                'duration': duration,
                'point_count': len(speeds),
                'start_time': df['time'].iloc[0],
                'end_time': df['time'].iloc[-1],
                'gpu_id': self.gpu_id
            }
            
        except Exception as e:
            logger.debug(f"GPX processing error: {e}")
            return None
    
    def _calculate_offset_simple(self, video_data: Dict, gps_data: Dict) -> Dict:
        """Simple offset calculation using massive GPU arrays"""
        
        try:
            # Get signals
            video_signal = video_data.get('motion_magnitude')
            gps_signal = gps_data.get('speed')
            
            if video_signal is None or gps_signal is None:
                raise ValueError("Missing signals")
            
            # Ensure CuPy arrays
            if isinstance(video_signal, np.ndarray):
                video_signal = cp.array(video_signal)
            if isinstance(gps_signal, np.ndarray):
                gps_signal = cp.array(gps_signal)
            
            # Normalize signals on GPU
            video_norm = self._normalize_signal_gpu(video_signal)
            gps_norm = self._normalize_signal_gpu(gps_signal)
            
            # MASSIVE FFT cross-correlation
            max_len = len(video_norm) + len(gps_norm) - 1
            pad_len = 1 << (max_len - 1).bit_length()
            
            # Pad signals
            video_padded = cp.pad(video_norm, (0, pad_len - len(video_norm)))
            gps_padded = cp.pad(gps_norm, (0, pad_len - len(gps_norm)))
            
            # MASSIVE GPU FFT operations
            video_fft = cp.fft.fft(video_padded)
            gps_fft = cp.fft.fft(gps_padded)
            
            # Cross-correlation
            correlation = cp.fft.ifft(cp.conj(video_fft) * gps_fft).real
            
            # Find peak
            peak_idx = cp.argmax(correlation)
            confidence = float(correlation[peak_idx] / len(video_norm))
            
            # Convert to offset
            offset_samples = float(peak_idx - len(video_norm) + 1)
            offset_seconds = offset_samples * 1.0  # 1Hz sampling
            
            # Validate offset
            if abs(offset_seconds) > 600.0 or abs(confidence) < 0.3:
                return {
                    'temporal_offset_seconds': None,
                    'offset_confidence': 0.0,
                    'offset_method': f'gpu_{self.gpu_id}_below_threshold',
                    'gpu_processing': True
                }
            
            return {
                'temporal_offset_seconds': offset_seconds,
                'offset_confidence': abs(confidence),
                'offset_method': f'simple_cupy_gpu_{self.gpu_id}',
                'gpu_processing': True,
                'sync_quality': self._assess_sync_quality(abs(confidence))
            }
            
        except Exception as e:
            logger.debug(f"Offset calculation error: {e}")
            return {
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'gpu_{self.gpu_id}_calculation_failed',
                'gpu_processing': False,
                'error': str(e)[:100]
            }
    
    def _normalize_signal_gpu(self, signal: cp.ndarray) -> cp.ndarray:
        """Normalize signal on GPU"""
        if len(signal) == 0:
            return signal
        
        mean = cp.mean(signal)
        std = cp.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean
    
    def _assess_sync_quality(self, confidence: float) -> str:
        """Assess sync quality"""
        if confidence >= 0.8:
            return 'excellent'
        elif confidence >= 0.6:
            return 'good'
        elif confidence >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def cleanup(self):
        """Cleanup GPU resources"""
        try:
            memory_pool = cp.get_default_memory_pool()
            memory_pool.free_all_blocks()
            cp.cuda.Device().synchronize()
            logger.info(f"🔥 GPU {self.gpu_id}: Cleanup complete")
        except Exception as e:
            logger.debug(f"GPU {self.gpu_id} cleanup error: {e}")

def process_gpu_batch_simple(gpu_id: int, matches_batch: List[Tuple], max_memory_gb: float) -> List[Dict]:
    """Process batch on specific GPU - SIMPLE AND FAST"""
    
    try:
        logger.info(f"🔥 GPU {gpu_id}: Starting batch with {len(matches_batch)} matches")
        
        # Initialize processor
        processor = SimpleGPUProcessor(gpu_id, max_memory_gb)
        
        # Process all matches
        results = processor.process_matches_batch(matches_batch)
        
        # Cleanup
        processor.cleanup()
        
        logger.info(f"🔥 GPU {gpu_id}: Batch complete")
        return results
        
    except Exception as e:
        logger.error(f"💀 GPU {gpu_id} batch processing FAILED: {e}")
        
        # Return failed results
        failed_results = []
        for _, _, match in matches_batch:
            enhanced_match = match.copy()
            enhanced_match.update({
                'temporal_offset_seconds': None,
                'offset_confidence': 0.0,
                'offset_method': f'gpu_{gpu_id}_batch_failed',
                'gpu_processing': False,
                'error': str(e)[:100]
            })
            failed_results.append(enhanced_match)
        
        return failed_results

def main():
    """Simple dual GPU main function"""
    parser = argparse.ArgumentParser(description='🔥 Fixed Simple Dual GPU - Actually Works')
    
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file (default: simple_dual_gpu_INPUTNAME.json)')
    parser.add_argument('--max-gpu-memory', type=float, default=15.0, help='Max GPU memory per GPU in GB')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--limit', type=int, help='Limit matches for testing')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Input validation
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else input_file.parent / f"simple_dual_gpu_{input_file.name}"
    
    # Check CuPy
    if not cp.cuda.is_available():
        logger.error("❌ CuPy CUDA not available!")
        sys.exit(1)
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    if gpu_count < 2:
        logger.error(f"❌ Need 2 GPUs, found {gpu_count}")
        sys.exit(1)
    
    # Load data
    logger.info(f"📁 Loading data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Collect matches
    all_matches = []
    video_results = data.get('results', {})
    
    for video_path, video_data in video_results.items():
        matches = video_data.get('matches', [])
        for match in matches:
            if match.get('combined_score', 0) >= args.min_score:
                all_matches.append((video_path, match['path'], match))
                if args.limit and len(all_matches) >= args.limit:
                    break
        if args.limit and len(all_matches) >= args.limit:
            break
    
    total_matches = len(all_matches)
    if total_matches == 0:
        logger.error("❌ No matches found!")
        sys.exit(1)
    
    logger.info("🔥🔥🔥 FIXED SIMPLE DUAL GPU PROCESSING 🔥🔥🔥")
    logger.info(f"🎯 Total matches: {total_matches}")
    logger.info(f"🔥 Using CuPy only - NO PYTORCH COMPLEXITY")
    
    # SPLIT MATCHES IN HALF - PROPERLY
    mid_point = len(all_matches) // 2
    gpu0_matches = all_matches[:mid_point]
    gpu1_matches = all_matches[mid_point:]
    
    logger.info(f"🔥 GPU 0: {len(gpu0_matches)} matches")
    logger.info(f"🔥 GPU 1: {len(gpu1_matches)} matches")
    
    # VERIFY DIFFERENT MATCHES
    gpu0_first = gpu0_matches[0][0] if gpu0_matches else "none"
    gpu1_first = gpu1_matches[0][0] if gpu1_matches else "none"
    logger.info(f"🔥 GPU 0 first file: {Path(gpu0_first).name}")
    logger.info(f"🔥 GPU 1 first file: {Path(gpu1_first).name}")
    
    # Process both GPUs simultaneously
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both GPU tasks
        future_gpu0 = executor.submit(process_gpu_batch_simple, 0, gpu0_matches, args.max_gpu_memory)
        future_gpu1 = executor.submit(process_gpu_batch_simple, 1, gpu1_matches, args.max_gpu_memory)
        
        # Get results
        gpu0_results = future_gpu0.result()
        gpu1_results = future_gpu1.result()
    
    processing_time = time.time() - start_time
    
    # Combine results in original order
    all_results = gpu0_results + gpu1_results
    
    # Create match map for merging
    result_map = {}
    for i, (video_path, gpx_path, _) in enumerate(all_matches):
        result_map[(video_path, gpx_path)] = all_results[i]
    
    # Merge into original structure
    enhanced_results = {}
    total_processed = 0
    total_success = 0
    
    for video_path, video_data in video_results.items():
        enhanced_video_data = video_data.copy()
        enhanced_matches = []
        
        for match in video_data.get('matches', []):
            gpx_path = match.get('path')
            key = (video_path, gpx_path)
            
            if key in result_map:
                enhanced_match = result_map[key]
                enhanced_matches.append(enhanced_match)
                
                total_processed += 1
                if enhanced_match.get('temporal_offset_seconds') is not None:
                    total_success += 1
            else:
                enhanced_matches.append(match)
        
        enhanced_video_data['matches'] = enhanced_matches
        enhanced_results[video_path] = enhanced_video_data
    
    # Create output
    enhanced_data = data.copy()
    enhanced_data['results'] = enhanced_results
    
    enhanced_data['simple_dual_gpu_info'] = {
        'processed_at': datetime.now().isoformat(),
        'total_matches_processed': total_processed,
        'successful_offsets': total_success,
        'success_rate': total_success / total_processed if total_processed > 0 else 0,
        'processing_time_seconds': processing_time,
        'processing_rate_matches_per_second': total_processed / processing_time if processing_time > 0 else 0,
        'gpu_0_matches': len(gpu0_matches),
        'gpu_1_matches': len(gpu1_matches),
        'fixed_version': True,
        'cupy_only': True,
        'no_pytorch_complexity': True
    }
    
    # Save results
    logger.info(f"💾 Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    
    # Summary
    logger.info("🔥🔥🔥 FIXED SIMPLE DUAL GPU COMPLETE 🔥🔥🔥")
    logger.info(f"📊 Processed: {total_processed}")
    logger.info(f"✅ Successful: {total_success}")
    logger.info(f"📈 Success rate: {total_success/total_processed*100:.1f}%")
    logger.info(f"⚡ Time: {processing_time:.1f}s")
    logger.info(f"🚀 Rate: {total_processed/processing_time:.1f} matches/second")
    logger.info(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    main()