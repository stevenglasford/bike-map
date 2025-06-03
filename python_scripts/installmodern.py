#!/usr/bin/env python3
"""
Modern GPU Video Processing using current libraries
Uses torchaudio, torchvision, or av with GPU acceleration
"""

import torch
import torchvision
import numpy as np
import cupy as cp
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import av
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModernGPUVideoProcessor:
    """GPU video processor using current, maintained libraries"""
    
    def __init__(self, gpu_id: int = 0, target_size: Tuple[int, int] = (640, 360)):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.target_size = target_size
        
        # Check available backends
        self.backend = self._select_backend()
        logger.info(f"Using video backend: {self.backend}")
    
    def _select_backend(self) -> str:
        """Select best available backend"""
        # Check for torchvision with video support
        try:
            import torchvision.io
            if hasattr(torchvision.io, 'read_video'):
                # Test if it works
                torchvision.io._video_opt._HAS_VIDEO_OPT = True
                return 'torchvision'
        except:
            pass
        
        # Check for torchaudio with ffmpeg
        try:
            import torchaudio
            if hasattr(torchaudio, 'io') and hasattr(torchaudio.io, 'StreamReader'):
                return 'torchaudio'
        except:
            pass
        
        # Fallback to PyAV (CPU decode -> GPU)
        try:
            import av
            return 'av'
        except:
            pass
        
        # Last resort: OpenCV
        return 'opencv'
    
    def decode_video_gpu(self, video_path: str, sample_fps: float = 2.0) -> Dict:
        """Decode video with best available method"""
        video_path = str(video_path)
        
        if self.backend == 'torchvision':
            return self._decode_torchvision(video_path, sample_fps)
        elif self.backend == 'torchaudio':
            return self._decode_torchaudio(video_path, sample_fps)
        elif self.backend == 'av':
            return self._decode_av_gpu(video_path, sample_fps)
        else:
            return self._decode_opencv_gpu(video_path, sample_fps)
    
    def _decode_torchvision(self, video_path: str, sample_fps: float) -> Dict:
        """Use torchvision for decoding"""
        try:
            import torchvision.io as io
            
            # Read video metadata
            video_meta = io.read_video_timestamps(video_path)
            fps = video_meta[1]
            
            # Read video
            video, audio, info = io.read_video(video_path, pts_unit='sec')
            
            # Move to GPU
            video_gpu = video.to(self.device)
            
            # Downsample frames
            total_frames = video_gpu.shape[0]
            frame_interval = max(1, int(fps / sample_fps))
            sampled_indices = torch.arange(0, total_frames, frame_interval)
            video_sampled = video_gpu[sampled_indices]
            
            # Resize if needed
            if video_sampled.shape[2] != self.target_size[1] or video_sampled.shape[3] != self.target_size[0]:
                video_sampled = video_sampled.permute(0, 3, 1, 2)  # NHWC -> NCHW
                video_sampled = torch.nn.functional.interpolate(
                    video_sampled.float(),
                    size=self.target_size[::-1],  # (H, W)
                    mode='bilinear',
                    align_corners=False
                )
                video_sampled = video_sampled.permute(0, 2, 3, 1)  # NCHW -> NHWC
            
            # Convert to CuPy
            frames_cp = cp.asarray(video_sampled.cpu().numpy())
            
            return {
                'frames': frames_cp,
                'fps': fps,
                'duration': total_frames / fps,
                'total_frames': total_frames
            }
            
        except Exception as e:
            logger.error(f"Torchvision decode failed: {e}")
            return self._decode_av_gpu(video_path, sample_fps)
    
    def _decode_torchaudio(self, video_path: str, sample_fps: float) -> Dict:
        """Use torchaudio StreamReader for decoding"""
        try:
            import torchaudio
            from torchaudio.io import StreamReader
            
            frames = []
            
            # Open stream
            streamer = StreamReader(video_path)
            
            # Get video stream info
            video_stream = None
            for i, stream in enumerate(streamer.streams):
                if stream.media_type == 'video':
                    video_stream = i
                    fps = stream.frame_rate
                    total_frames = stream.num_frames
                    break
            
            if video_stream is None:
                raise ValueError("No video stream found")
            
            # Configure video stream
            streamer.add_basic_video_stream(
                video_stream,
                frames_per_chunk=32,
                format='rgb24'
            )
            
            # Read frames
            frame_count = 0
            sample_interval = max(1, int(fps / sample_fps))
            
            for (video_chunk,) in streamer.stream():
                if video_chunk is not None:
                    # Sample frames
                    for i in range(0, video_chunk.shape[0], sample_interval):
                        if i < video_chunk.shape[0]:
                            frame = video_chunk[i]
                            # Resize if needed
                            if frame.shape[1] != self.target_size[1] or frame.shape[2] != self.target_size[0]:
                                frame = torch.nn.functional.interpolate(
                                    frame.unsqueeze(0),
                                    size=self.target_size[::-1],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                            frames.append(frame)
                    frame_count += video_chunk.shape[0]
            
            if frames:
                # Stack and move to GPU
                frames_tensor = torch.stack(frames).to(self.device)
                frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # NCHW -> NHWC
                
                # Convert to CuPy
                frames_cp = cp.asarray(frames_tensor.cpu().numpy())
                
                return {
                    'frames': frames_cp,
                    'fps': fps,
                    'duration': total_frames / fps if total_frames else frame_count / fps,
                    'total_frames': total_frames or frame_count
                }
            
        except Exception as e:
            logger.error(f"Torchaudio decode failed: {e}")
            return self._decode_av_gpu(video_path, sample_fps)
    
    def _decode_av_gpu(self, video_path: str, sample_fps: float) -> Dict:
        """Use PyAV with GPU upload"""
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            fps = float(stream.average_rate)
            total_frames = stream.frames
            duration = float(stream.duration * stream.time_base) if stream.duration else 0
            
            # Decode parameters
            stream.codec_context.skip_frame = 'NONKEY'  # Faster decoding
            
            frames = []
            frame_interval = max(1, int(fps / sample_fps))
            frame_count = 0
            
            for frame in container.decode(stream):
                if frame_count % frame_interval == 0:
                    # Convert to numpy
                    img = frame.to_ndarray(format='rgb24')
                    
                    # Resize
                    if img.shape[:2] != self.target_size[::-1]:
                        import cv2
                        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
                    
                    frames.append(img)
                
                frame_count += 1
                
                # Batch upload to GPU
                if len(frames) >= 32:
                    batch = np.stack(frames)
                    yield cp.asarray(batch)
                    frames = []
            
            # Final batch
            if frames:
                batch = np.stack(frames)
                frames_cp = cp.asarray(batch)
            else:
                frames_cp = cp.array([])
            
            container.close()
            
            return {
                'frames': frames_cp,
                'fps': fps,
                'duration': duration or (total_frames / fps if total_frames else 0),
                'total_frames': total_frames or frame_count
            }
            
        except Exception as e:
            logger.error(f"PyAV decode failed: {e}")
            return self._decode_opencv_gpu(video_path, sample_fps)
    
    def _decode_opencv_gpu(self, video_path: str, sample_fps: float) -> Dict:
        """Fallback to OpenCV with immediate GPU upload"""
        import cv2
        
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        frame_interval = max(1, int(fps / sample_fps))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                if frame.shape[:2] != self.target_size[::-1]:
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        # Batch upload to GPU
        if frames:
            frames_np = np.stack(frames)
            frames_cp = cp.asarray(frames_np)
        else:
            frames_cp = cp.array([])
        
        return {
            'frames': frames_cp,
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames
        }


# Install script
def install_video_backends():
    """Print installation commands for video backends"""
    print("\n=== INSTALLATION COMMANDS ===\n")
    
    print("# 1. Install PyAV (recommended, most reliable):")
    print("conda install av -c conda-forge")
    print("# or")
    print("pip install av\n")
    
    print("# 2. Install torchvision with video support:")
    print("conda install torchvision -c pytorch")
    print("# or")
    print("pip install torchvision\n")
    
    print("# 3. Install torchaudio with ffmpeg:")
    print("conda install torchaudio -c pytorch")
    print("pip install torchaudio\n")
    
    print("# 4. Ensure ffmpeg is installed:")
    print("conda install ffmpeg -c conda-forge")
    print("# or")
    print("sudo apt-get install ffmpeg\n")


if __name__ == "__main__":
    install_video_backends()