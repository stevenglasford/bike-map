import cv2
import numpy as np
import torch
import subprocess
import json
import logging

logger = logging.getLogger(__name__)

class SimpleNVIDIADecoder:
    """Simple NVIDIA Hardware-accelerated video decoder using FFmpeg only"""
    
    def __init__(self, gpu_ids=[0]):
        self.gpu_ids = gpu_ids
        self.current_gpu = 0
        
        # Check for NVIDIA hardware decoder availability
        self.has_nvdec = self._check_nvdec_support()
        
        if self.has_nvdec:
            print("✓ NVIDIA NVDEC hardware decoder available")
        else:
            
            print("⚠ No NVIDIA acceleration available, will use CPU fallback")
            exit(1)
    
    def _check_nvdec_support(self):
        """Check if NVDEC is available through FFmpeg"""
        try:
            # Check hardware accelerators
            result = subprocess.run([
                'ffmpeg', '-hwaccels'
            ], capture_output=True, text=True, timeout=10)
            
            has_cuda = 'cuda' in result.stdout.lower()
            
            # Check for NVDEC decoders
            result2 = subprocess.run([
                'ffmpeg', '-decoders'
            ], capture_output=True, text=True, timeout=10)
            
            has_cuvid = 'cuvid' in result2.stdout.lower()
            
            return has_cuda and has_cuvid
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"FFmpeg check failed: {e}")
            return False
    
    def decode_video_batch(self, video_path, sample_rate=2.0, target_size=(640, 360)):
        """Decode video with NVIDIA hardware acceleration"""
        
        if self.has_nvdec:
            print(f"Using NVIDIA hardware acceleration for {video_path}")
            return self._decode_with_nvdec_ffmpeg(video_path, sample_rate, target_size)
        else:
            print(f"Using CPU fallback for {video_path}")
            return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
    
    def _decode_with_nvdec_ffmpeg(self, video_path, sample_rate, target_size):
        """Decode using NVDEC through FFmpeg"""
        try:
            # Get video info first
            fps, duration, total_frames = self._get_video_info(video_path)
            if fps <= 0:
                print(f"Could not get video info for {video_path}")
                return None, 0, 0, []
            
            print(f"Video info: {fps:.1f} fps, {duration:.1f}s duration")
            
            # Auto-detect video codec for optimal decoder selection
            codec_info = self._detect_video_codec(video_path)
            decoder = self._select_best_decoder(codec_info)
            
            print(f"Using decoder: {decoder} for codec: {codec_info}")
            
            gpu_id = self.gpu_ids[self.current_gpu % len(self.gpu_ids)]
            self.current_gpu += 1
            
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
                '-loglevel', 'warning',
                'pipe:1'
            ]
            
            print(f"Running FFmpeg command...")
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            frames = []
            frame_size = target_size[0] * target_size[1] * 3
            
            while True:
                frame_data = process.stdout.read(frame_size)
                if len(frame_data) != frame_size:
                    break
                
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((target_size[1], target_size[0], 3))
                frames.append(frame)
                
                if len(frames) % 30 == 0:
                    print(f"  Decoded {len(frames)} frames...")
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode()}")
                return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
            
            if frames:
                print(f"✓ Successfully decoded {len(frames)} frames using NVIDIA hardware")
                frames_array = np.stack(frames)
                frames_tensor = torch.from_numpy(frames_array).float() / 255.0
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)
                
                frame_indices = list(range(len(frames)))
                return frames_tensor, fps, duration, frame_indices
            else:
                print("No frames decoded")
                return None, 0, 0, []
            
        except Exception as e:
            print(f"NVDEC decoding failed for {video_path}: {e}")
            return self._decode_with_cpu_fallback(video_path, sample_rate, target_size)
    
    def _detect_video_codec(self, video_path):
        """Detect video codec for optimal decoder selection"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_streams', '-select_streams', 'v:0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                if 'streams' in info and info['streams']:
                    codec = info['streams'][0].get('codec_name', 'unknown')
                    return codec
        except Exception as e:
            print(f"Codec detection failed: {e}")
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
        return codec_map.get(codec_info, 'h264_cuvid')
    
    def _decode_with_cpu_fallback(self, video_path, sample_rate, target_size):
        """CPU fallback decoder"""
        print(f"Using CPU fallback for {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return None, 0, 0, []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"CPU decoding: {fps:.1f} fps, {frame_count} frames, {duration:.1f}s")
        
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
                
                if len(frames) % 30 == 0:
                    print(f"  CPU decoded {len(frames)} frames...")
        
        cap.release()
        
        if frames:
            print(f"✓ CPU decoded {len(frames)} frames")
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
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                return fps, duration, frame_count
            
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
            print(f"Error getting video info for {video_path}: {e}")
        
        return 0, 0, 0