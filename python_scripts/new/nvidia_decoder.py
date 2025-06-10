import cv2
import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class NVIDIAVideoDecoder:
    """NVIDIA Hardware-accelerated video decoder using FFmpeg and OpenCV CUDA"""
    
    def __init__(self, gpu_ids=[0, 1]):
        self.gpu_ids = gpu_ids
        self.current_gpu = 0
        
        # Check for NVIDIA hardware decoder availability
        self.has_nvdec = self._check_nvdec_support()
        self.has_opencv_cuda = self._check_opencv_cuda()
        
        if self.has_nvdec:
            logger.info("NVIDIA NVDEC hardware decoder available")
        if self.has_opencv_cuda:
            logger.info("OpenCV CUDA support available")
        
        if not (self.has_nvdec or self.has_opencv_cuda):
            logger.warning("No GPU acceleration available, falling back to CPU")
    
    def _check_nvdec_support(self):
        """Check if NVDEC is available through FFmpeg"""
        try:
            result = subprocess.run([
                'ffmpeg', '-hide_banner', '-hwaccels'
            ], capture_output=True, text=True, timeout=10)
            
            return 'cuda' in result.stdout.lower() or 'nvdec' in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_opencv_cuda(self):
        """Check if OpenCV was compiled with CUDA support"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            return False
    
    def decode_video_batch(self, video_path, sample_rate=2.0, target_size=(640, 360)):
        """Decode video with NVIDIA hardware acceleration"""
        
        # Try NVDEC first (fastest), then OpenCV CUDA, then fallback
        if self.has_nvdec:
            return self._decode_with_nvdec_ffmpeg(video_path, sample_rate, target_size)
        elif self.has_opencv_cuda:
            return self._decode_with_opencv_cuda(video_path, sample_rate, target_size)
        else:
            return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
    
    def _decode_with_nvdec_ffmpeg(self, video_path, sample_rate, target_size):
        """Decode using NVDEC through FFmpeg"""
        try:
            # Get video info first
            fps, duration, total_frames = self._get_video_info(video_path)
            if fps <= 0:
                return None, 0, 0, []
            
            # Calculate frame sampling
            frame_interval = max(1, int(fps / sample_rate))
            
            # Use FFmpeg with NVDEC to decode frames directly to numpy
            gpu_id = self.gpu_ids[self.current_gpu % len(self.gpu_ids)]
            self.current_gpu += 1
            
            # Auto-detect video codec for optimal decoder selection
            codec_info = self._detect_video_codec(video_path)
            decoder = self._select_best_decoder(codec_info)
            
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(gpu_id),
                '-c:v', decoder,
                '-i', video_path,
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]},fps={sample_rate}',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-hide_banner',
                '-loglevel', 'error',
                'pipe:1'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            frames = []
            frame_size = target_size[0] * target_size[1] * 3
            frame_count = 0
            
            while True:
                frame_data = process.stdout.read(frame_size)
                if len(frame_data) != frame_size:
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((target_size[1], target_size[0], 3))
                frames.append(frame)
                frame_count += 1
            
            process.stdout.close()
            process.wait()
            
            if frames:
                # Convert to torch tensor
                frames_array = np.stack(frames)
                frames_tensor = torch.from_numpy(frames_array).float() / 255.0
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
                
                # Calculate frame indices
                frame_indices = list(range(0, len(frames) * frame_interval, frame_interval))
                
                return frames_tensor, fps, duration, frame_indices
            
        except Exception as e:
            logger.error(f"NVDEC decoding failed for {video_path}: {e}")
            return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
        
        return None, 0, 0, []
    
    def _detect_video_codec(self, video_path):
        """Detect video codec for optimal decoder selection"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_streams', '-select_streams', 'v:0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                if 'streams' in info and info['streams']:
                    return info['streams'][0].get('codec_name', 'unknown')
        except:
            pass
        return 'unknown'
    
    def _select_best_decoder(self, codec_info):
        """Select best NVIDIA decoder based on codec"""
        codec_map = {
            'h264': 'h264_cuvid',
            'hevc': 'hevc_cuvid', 
            'h265': 'hevc_cuvid',
            'mpeg4': 'mpeg4_cuvid',
            'mpeg2video': 'mpeg2_cuvid',
            'vc1': 'vc1_cuvid',
            'vp8': 'vp8_cuvid',
            'vp9': 'vp9_cuvid',
            'av1': 'av1_cuvid'
        }
        return codec_map.get(codec_info, 'h264_cuvid')  # Default to h264_cuvid
    
    def _decode_with_opencv_cuda(self, video_path, sample_rate, target_size):
        """Decode using OpenCV with CUDA support"""
        try:
            # Set CUDA device
            gpu_id = self.gpu_ids[self.current_gpu % len(self.gpu_ids)]
            cv2.cuda.setDevice(gpu_id)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            frame_interval = max(1, int(fps / sample_rate))
            
            # Create CUDA streams and GPU matrices for processing
            stream = cv2.cuda_Stream()
            gpu_frame = cv2.cuda_GpuMat()
            gpu_resized = cv2.cuda_GpuMat()
            
            frames = []
            frame_indices = []
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Upload to GPU
                gpu_frame.upload(frame, stream)
                
                # Resize on GPU
                cv2.cuda.resize(gpu_frame, target_size, gpu_resized, stream=stream)
                
                # Convert color space on GPU
                gpu_rgb = cv2.cuda_GpuMat()
                cv2.cuda.cvtColor(gpu_resized, gpu_rgb, cv2.COLOR_BGR2RGB, stream=stream)
                
                # Download from GPU
                stream.waitForCompletion()
                processed_frame = gpu_rgb.download()
                
                frames.append(processed_frame)
                frame_indices.append(i)
            
            cap.release()
            
            if frames:
                frames_tensor = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1) for f in frames
                ]).float() / 255.0
                
                return frames_tensor, fps, duration, frame_indices
            
        except Exception as e:
            logger.error(f"OpenCV CUDA decoding failed for {video_path}: {e}")
            return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
        
        return None, 0, 0, []
    
    def _decode_with_cpu_fallback(self, video_path, sample_rate, target_size):
        """CPU fallback decoder"""
        logger.warning(f"Using CPU fallback for {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0, 0, []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        frame_interval = max(1, int(fps / sample_rate))
        frames = []
        frame_indices = []
        
        for i in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_indices.append(i)
        
        cap.release()
        
        if frames:
            frames_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1) for f in frames
            ]).float() / 255.0
            return frames_tensor, fps, duration, frame_indices
        
        return None, 0, 0, []
    
    def _get_video_info(self, video_path):
        """Get video information using FFprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                # Fallback to OpenCV
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                return fps, duration, frame_count
            
            import json
            info = json.loads(result.stdout)
            
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                fps_str = video_stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 0
                else:
                    fps = float(fps_str)
                
                duration = float(video_stream.get('duration', 0))
                total_frames = int(video_stream.get('nb_frames', fps * duration))
                
                return fps, duration, total_frames
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
        
        return 0, 0, 0


class NVIDIAVideoProcessor:
    """High-level video processor with NVIDIA acceleration"""
    
    def __init__(self, gpu_ids=[0, 1]):
        self.decoder = NVIDIAVideoDecoder(gpu_ids)
        self.gpu_ids = gpu_ids
    
    def process_video_batch(self, video_paths, sample_rate=2.0, target_size=(640, 360)):
        """Process multiple videos with load balancing across GPUs"""
        results = {}
        
        for video_path in video_paths:
            try:
                frames, fps, duration, indices = self.decoder.decode_video_batch(
                    video_path, sample_rate, target_size
                )
                
                if frames is not None:
                    results[video_path] = {
                        'frames': frames,
                        'fps': fps,
                        'duration': duration,
                        'frame_indices': indices
                    }
                else:
                    results[video_path] = None
                    
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results[video_path] = None
        
        return results


# Alternative: Direct NVIDIA Video Codec SDK approach (if you want maximum performance)
class DirectNVENCDecoder:
    """Direct NVIDIA codec approach using subprocess with specific codec parameters"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.temp_dir = tempfile.mkdtemp()
    
    def decode_video_to_frames(self, video_path, sample_rate=2.0, target_size=(640, 360)):
        """Decode video directly to frame files using NVDEC"""
        try:
            output_pattern = os.path.join(self.temp_dir, "frame_%06d.jpg")
            
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(self.gpu_id),
                '-hwaccel_output_format', 'cuda',
                '-c:v', 'h264_cuvid',
                '-i', video_path,
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]},fps={sample_rate}',
                '-q:v', '2',  # High quality
                '-f', 'image2',
                output_pattern,
                '-hide_banner',
                '-loglevel', 'error',
                '-y'  # Overwrite files
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Load frames from temporary files
                frame_files = sorted([f for f in os.listdir(self.temp_dir) if f.startswith('frame_')])
                frames = []
                
                for frame_file in frame_files:
                    frame_path = os.path.join(self.temp_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    os.remove(frame_path)  # Clean up
                
                if frames:
                    frames_tensor = torch.stack([
                        torch.from_numpy(f).permute(2, 0, 1) for f in frames
                    ]).float() / 255.0
                    return frames_tensor
            
        except Exception as e:
            logger.error(f"Direct NVENC decoding failed: {e}")
        
        return None
    
    def __del__(self):
        """Cleanup temporary directory"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass