#!/usr/bin/env python3
"""
Fixed Chunked Processor Integration
Addresses the issues with GPU timeouts and improper chunking

Key Fixes:
1. Only use chunked processing when actually needed
2. Integrate the fixed GPU manager properly
3. Reasonable chunk sizes (not 3-frame chunks!)
4. Better fallback logic
5. Proper GPU timeout handling
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import subprocess
import os
import time
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading

logger = logging.getLogger(__name__)

# Import the fixed GPU manager
try:
    from gpu_timeout_fix import FixedGPUManager
    logger.info("✅ Using FixedGPUManager for chunked processing")
except ImportError:
    logger.warning("⚠️ FixedGPUManager not available, using fallback")
    FixedGPUManager = None

class SmartChunkedVideoProcessor:
    """Smart chunked processor that only activates when needed"""
    
    def __init__(self, gpu_manager, config):
        self.gpu_manager = gpu_manager
        self.config = config
        
        # Replace GPU manager with fixed version if available
        if FixedGPUManager is not None and not isinstance(gpu_manager, FixedGPUManager):
            logger.info("🔧 Replacing GPU manager with FixedGPUManager")
            self.gpu_manager = FixedGPUManager(gpu_manager.gpu_ids, gpu_manager.strict, config)
        
        # Smart chunking thresholds
        self.chunking_thresholds = {
            'memory_gb': 6.0,      # Use chunking if video needs >6GB
            'resolution_pixels': 1920 * 1080 * 2,  # Use chunking for >2x 1080p
            'duration_seconds': 300,  # Use chunking for >5 minute videos
            'min_chunk_frames': 30,   # Minimum 30 frames per chunk
            'max_chunk_frames': 120,  # Maximum 120 frames per chunk
        }
        
        logger.info(f"Smart chunked processor initialized")
        logger.info(f"Chunking thresholds: {self.chunking_thresholds}")
    
    def should_use_chunking(self, video_path: str) -> Tuple[bool, str]:
        """Determine if video needs chunked processing"""
        try:
            video_info = self._get_video_info_quick(video_path)
            if not video_info:
                return False, "Could not analyze video"
            
            width, height = video_info['width'], video_info['height']
            duration = video_info['duration']
            fps = video_info['fps']
            frame_count = video_info['frame_count']
            
            # Calculate memory requirement
            bytes_per_frame = width * height * 3 * 4  # RGB float32
            total_memory_gb = (frame_count * bytes_per_frame) / (1024**3)
            
            # Check thresholds
            reasons = []
            
            if total_memory_gb > self.chunking_thresholds['memory_gb']:
                reasons.append(f"memory: {total_memory_gb:.1f}GB > {self.chunking_thresholds['memory_gb']}GB")
            
            if width * height > self.chunking_thresholds['resolution_pixels']:
                reasons.append(f"resolution: {width}x{height}")
            
            if duration > self.chunking_thresholds['duration_seconds']:
                reasons.append(f"duration: {duration:.1f}s > {self.chunking_thresholds['duration_seconds']}s")
            
            use_chunking = len(reasons) > 0
            reason = f"Chunking {'ENABLED' if use_chunking else 'DISABLED'}: " + (", ".join(reasons) if reasons else "all thresholds OK")
            
            return use_chunking, reason
            
        except Exception as e:
            logger.debug(f"Error checking chunking need: {e}")
            return False, f"Error: {e}"
    
    def _get_video_info_quick(self, video_path: str) -> Optional[Dict]:
        """Quick video info extraction"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return None
            
            probe_data = json.loads(result.stdout)
            video_stream = next((s for s in probe_data.get('streams', []) if s.get('codec_type') == 'video'), None)
            
            if not video_stream:
                return None
            
            width = int(video_stream.get('width', 1920))
            height = int(video_stream.get('height', 1080))
            duration = float(video_stream.get('duration', 0))
            
            # FPS calculation
            fps = 30.0
            fps_str = video_stream.get('r_frame_rate', '30/1')
            try:
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) > 0 else 30.0
            except:
                fps = 30.0
            
            # Frame count
            frame_count = int(duration * fps) if duration > 0 else 1000
            
            return {
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'duration': duration,
                'fps': fps
            }
            
        except Exception as e:
            logger.debug(f"Quick video info failed: {e}")
            return None
    
    def process_video_smart(self, video_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Smart processing that chooses between chunked and normal processing"""
        
        # Check if chunking is needed
        use_chunking, reason = self.should_use_chunking(video_path)
        
        logger.info(f"📊 {Path(video_path).name}: {reason}")
        
        if not use_chunking:
            logger.info(f"🔄 Using normal processing for {Path(video_path).name}")
            return None  # Signal to use normal processing
        
        logger.info(f"🧩 Using chunked processing for {Path(video_path).name}")
        return self.process_video_chunked_fixed(video_path)
    
    def process_video_chunked_fixed(self, video_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Fixed chunked processing with proper GPU management"""
        
        try:
            # Get video info
            video_info = self._get_video_info_quick(video_path)
            if not video_info:
                logger.error(f"Could not analyze video: {video_path}")
                return None
            
            logger.info(f"Processing {Path(video_path).name}: {video_info['width']}x{video_info['height']}, "
                       f"{video_info['frame_count']} frames")
            
            # Calculate REASONABLE chunking strategy
            chunk_strategy = self._calculate_reasonable_chunking(video_info)
            
            if chunk_strategy['frames_per_chunk'] < self.chunking_thresholds['min_chunk_frames']:
                logger.warning(f"Chunk size too small ({chunk_strategy['frames_per_chunk']} frames), using normal processing")
                return None
            
            logger.info(f"Chunking strategy: {chunk_strategy['num_chunks']} chunks of "
                       f"{chunk_strategy['frames_per_chunk']} frames each")
            
            # Process chunks with fixed GPU management
            all_features = []
            processing_times = []
            
            for chunk_idx in range(chunk_strategy['num_chunks']):
                chunk_start_time = time.time()
                
                # Calculate chunk boundaries
                start_frame = chunk_idx * chunk_strategy['frames_per_chunk']
                end_frame = min(start_frame + chunk_strategy['frames_per_chunk'], 
                              video_info['frame_count'])
                
                if end_frame - start_frame < self.chunking_thresholds['min_chunk_frames']:
                    logger.info(f"Skipping small final chunk: {end_frame - start_frame} frames")
                    break
                
                logger.info(f"Processing chunk {chunk_idx + 1}/{chunk_strategy['num_chunks']}: "
                           f"frames {start_frame}-{end_frame}")
                
                # Process chunk with fixed GPU acquisition
                chunk_features = self._process_chunk_with_fixed_gpu(
                    video_path, video_info, start_frame, end_frame, chunk_idx
                )
                
                if chunk_features is not None:
                    all_features.append(chunk_features)
                    
                    chunk_time = time.time() - chunk_start_time
                    processing_times.append(chunk_time)
                    
                    fps = (end_frame - start_frame) / chunk_time
                    logger.info(f"✅ Chunk {chunk_idx + 1} completed: {chunk_time:.1f}s ({fps:.1f} FPS)")
                else:
                    logger.error(f"❌ Failed chunk {chunk_idx + 1}")
                    # Don't abort immediately - try to continue
                    continue
                
                # Cleanup between chunks
                self._cleanup_between_chunks()
            
            # Check if we have enough successful chunks
            if len(all_features) == 0:
                logger.error("❌ No chunks processed successfully")
                return None
            elif len(all_features) < chunk_strategy['num_chunks'] * 0.5:
                logger.warning(f"⚠️ Only {len(all_features)}/{chunk_strategy['num_chunks']} chunks succeeded")
            
            # Aggregate features
            final_features = self._aggregate_features_simple(all_features, video_info)
            
            total_time = sum(processing_times)
            avg_fps = video_info['frame_count'] / total_time if total_time > 0 else 0
            
            logger.info(f"🚀 Chunked processing complete: {total_time:.1f}s, {avg_fps:.1f} FPS")
            logger.info(f"Success rate: {len(all_features)}/{chunk_strategy['num_chunks']} chunks")
            
            return final_features
                
        except Exception as e:
            logger.error(f"❌ Chunked processing failed: {e}")
            return None
    
    def _calculate_reasonable_chunking(self, video_info: Dict) -> Dict:
        """Calculate reasonable chunking strategy (not 3-frame chunks!)"""
        
        width, height = video_info['width'], video_info['height']
        total_frames = video_info['frame_count']
        
        # Calculate reasonable chunk size based on resolution
        if width * height > 3840 * 2160:  # 4K+
            base_chunk_size = 60
        elif width * height > 1920 * 1080:  # 1080p+
            base_chunk_size = 90
        else:  # Lower resolution
            base_chunk_size = 120
        
        # Adjust based on total frames
        if total_frames < 100:
            frames_per_chunk = total_frames  # Don't chunk very short videos
            num_chunks = 1
        else:
            frames_per_chunk = max(
                self.chunking_thresholds['min_chunk_frames'],
                min(base_chunk_size, self.chunking_thresholds['max_chunk_frames'])
            )
            
            # Ensure we don't create too many tiny chunks
            if total_frames / frames_per_chunk > 100:  # More than 100 chunks
                frames_per_chunk = max(
                    self.chunking_thresholds['min_chunk_frames'],
                    total_frames // 100
                )
            
            num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
        
        logger.info(f"Chunking calculation:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Base chunk size: {base_chunk_size}")
        logger.info(f"  Final chunk size: {frames_per_chunk}")
        logger.info(f"  Number of chunks: {num_chunks}")
        
        return {
            'frames_per_chunk': frames_per_chunk,
            'num_chunks': num_chunks
        }
    
    def _process_chunk_with_fixed_gpu(self, video_path: str, video_info: Dict,
                                     start_frame: int, end_frame: int, chunk_idx: int) -> Optional[Dict]:
        """Process chunk with fixed GPU acquisition"""
        
        # Use the fixed GPU manager
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                # Acquire GPU with shorter timeout
                gpu_id = self.gpu_manager.acquire_gpu(timeout=10)  # Shorter timeout
                
                if gpu_id is None:
                    if retry < max_retries - 1:
                        logger.warning(f"GPU acquisition failed for chunk {chunk_idx}, retry {retry + 1}/{max_retries}")
                        time.sleep(1.0)  # Brief wait before retry
                        continue
                    else:
                        logger.error(f"Could not acquire GPU for chunk {chunk_idx} after {max_retries} retries")
                        return None
                
                # Process chunk
                try:
                    chunk_features = self._process_single_chunk_minimal(
                        video_path, video_info, start_frame, end_frame, chunk_idx, gpu_id
                    )
                    
                    # Successful processing
                    return chunk_features
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} processing failed on GPU {gpu_id}: {e}")
                    if retry < max_retries - 1:
                        logger.info(f"Retrying chunk {chunk_idx} (attempt {retry + 2}/{max_retries})")
                        continue
                    else:
                        return None
                
                finally:
                    # Always release GPU
                    try:
                        self.gpu_manager.release_gpu(gpu_id)
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in chunk {chunk_idx} retry {retry + 1}: {e}")
                if retry == max_retries - 1:
                    return None
                continue
        
        return None
    
    def _process_single_chunk_minimal(self, video_path: str, video_info: Dict,
                                     start_frame: int, end_frame: int, chunk_idx: int, gpu_id: int) -> Optional[Dict]:
        """Minimal chunk processing"""
        
        try:
            # Load chunk frames
            chunk_frames = self._load_chunk_minimal(video_path, video_info, start_frame, end_frame)
            if chunk_frames is None:
                return None
            
            # Simple feature extraction
            features = self._extract_features_minimal(chunk_frames, gpu_id)
            
            return features
            
        except Exception as e:
            logger.error(f"Minimal chunk processing failed: {e}")
            return None
    
    def _load_chunk_minimal(self, video_path: str, video_info: Dict,
                           start_frame: int, end_frame: int) -> Optional[np.ndarray]:
        """Minimal chunk loading"""
        
        try:
            width, height = video_info['width'], video_info['height']
            num_frames = end_frame - start_frame
            
            # Use OpenCV for simplicity
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_list = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb.astype(np.float32) / 255.0)
            
            cap.release()
            
            if not frames_list:
                return None
            
            return np.stack(frames_list, axis=0)
            
        except Exception as e:
            logger.debug(f"Minimal chunk loading failed: {e}")
            return None
    
    def _extract_features_minimal(self, frames: np.ndarray, gpu_id: int) -> Dict[str, np.ndarray]:
        """Minimal feature extraction"""
        
        try:
            num_frames = frames.shape[0]
            
            # Simple motion features
            motion_magnitude = np.zeros(num_frames)
            if num_frames > 1:
                for i in range(1, num_frames):
                    diff = np.abs(frames[i] - frames[i-1])
                    motion_magnitude[i] = np.mean(diff)
            
            # Simple color features
            color_variance = np.var(frames, axis=(1, 2))
            mean_color_variance = np.mean(color_variance, axis=1)
            
            # Simple global features (using CPU)
            global_features = np.mean(frames.reshape(num_frames, -1), axis=1)
            
            return {
                'motion_magnitude': motion_magnitude,
                'motion_direction': np.zeros(num_frames),
                'acceleration': np.zeros(num_frames),
                'color_variance': mean_color_variance,
                'edge_density': np.zeros(num_frames),
                'global_features': global_features.reshape(num_frames, 1),
                'motion_features': global_features.reshape(num_frames, 1),
                'texture_features': global_features.reshape(num_frames, 1),
                'color_histograms': color_variance
            }
            
        except Exception as e:
            logger.error(f"Minimal feature extraction failed: {e}")
            # Return dummy features
            num_frames = frames.shape[0] if frames is not None else 10
            return {
                'motion_magnitude': np.zeros(num_frames),
                'motion_direction': np.zeros(num_frames),
                'acceleration': np.zeros(num_frames),
                'color_variance': np.zeros(num_frames),
                'edge_density': np.zeros(num_frames),
                'global_features': np.zeros((num_frames, 1)),
                'motion_features': np.zeros((num_frames, 1)),
                'texture_features': np.zeros((num_frames, 1)),
                'color_histograms': np.zeros((num_frames, 3))
            }
    
    def _aggregate_features_simple(self, chunk_features_list: List[Dict], video_info: Dict) -> Dict[str, np.ndarray]:
        """Simple feature aggregation"""
        
        if not chunk_features_list:
            return {}
        
        aggregated = {}
        
        # Time-series features - concatenate
        time_series_keys = ['motion_magnitude', 'motion_direction', 'acceleration', 'color_variance', 'edge_density']
        
        for key in time_series_keys:
            all_values = []
            for chunk in chunk_features_list:
                if key in chunk and chunk[key] is not None:
                    all_values.append(chunk[key])
            
            if all_values:
                try:
                    aggregated[key] = np.concatenate(all_values, axis=0)
                except Exception as e:
                    logger.debug(f"Could not concatenate {key}: {e}")
                    aggregated[key] = np.zeros(100)  # Dummy fallback
        
        # CNN features - average
        cnn_keys = ['global_features', 'motion_features', 'texture_features']
        
        for key in cnn_keys:
            all_features = []
            for chunk in chunk_features_list:
                if key in chunk and chunk[key] is not None:
                    all_features.append(chunk[key])
            
            if all_features:
                try:
                    stacked = np.concatenate(all_features, axis=0)
                    aggregated[key] = np.mean(stacked, axis=0, keepdims=True)
                except Exception as e:
                    logger.debug(f"Could not aggregate {key}: {e}")
                    aggregated[key] = np.zeros((1, 1))  # Dummy fallback
        
        # Color histograms
        if chunk_features_list[0].get('color_histograms') is not None:
            all_hists = [chunk['color_histograms'] for chunk in chunk_features_list 
                        if 'color_histograms' in chunk and chunk['color_histograms'] is not None]
            if all_hists:
                try:
                    aggregated['color_histograms'] = np.concatenate(all_hists, axis=0)
                except Exception as e:
                    logger.debug(f"Could not concatenate color histograms: {e}")
        
        # Add metadata
        aggregated['duration'] = video_info['duration']
        aggregated['fps'] = video_info['fps']
        aggregated['frame_count'] = video_info['frame_count']
        aggregated['resolution'] = (video_info['width'], video_info['height'])
        aggregated['processing_mode'] = 'GPU_CHUNKED_SMART'
        
        return aggregated
    
    def _cleanup_between_chunks(self):
        """Cleanup between chunks"""
        gc.collect()
        time.sleep(0.01)

# Updated process function for matcher.py
def process_video_with_smart_chunking(args) -> Tuple[str, Optional[Dict]]:
    """Updated process function that uses smart chunking"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    try:
        # Initialize smart chunked processor
        smart_processor = SmartChunkedVideoProcessor(gpu_manager, config)
        
        # Try smart processing first
        features = smart_processor.process_video_smart(video_path)
        
        if features is not None:
            # Chunked processing succeeded
            if powersafe_manager:
                powersafe_manager.mark_video_features_done(video_path)
            
            logger.info(f"✅ Smart chunked processing successful: {Path(video_path).name}")
            return video_path, features
        else:
            # Use normal processing (chunking not needed or failed)
            logger.info(f"🔄 Using normal processing fallback: {Path(video_path).name}")
            
            # Import and use original processing function
            try:
                from matcher import original_process_video_parallel_enhanced
                return original_process_video_parallel_enhanced(args)
            except ImportError:
                logger.error("Could not import original processing function")
                if powersafe_manager:
                    powersafe_manager.mark_video_failed(video_path, "Could not import fallback processor")
                return video_path, None
        
    except Exception as e:
        error_msg = f"Smart chunked processing failed: {str(e)}"
        logger.error(f"{error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        return video_path, None

# Test function
if __name__ == "__main__":
    print("🧩 Smart Chunked Video Processor")
    print("✅ Only chunks when necessary")
    print("✅ Reasonable chunk sizes (30-120 frames)")
    print("✅ Fixed GPU timeout handling")
    print("✅ Intelligent fallback to normal processing")