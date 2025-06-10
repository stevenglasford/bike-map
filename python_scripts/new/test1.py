import cv2
import numpy as np
import torch
import subprocess
import json
import tempfile
import os

class WorkingCUDADecoder:
    """CUDA decoder that works with your setup (avoids CUVID)"""
    
    def __init__(self, gpu_ids=[0, 1]):
        self.gpu_ids = gpu_ids
        self.current_gpu = 0
        
        print("🔍 Testing working CUDA methods...")
        self.working_method = self._find_working_method()
        
        if not self.working_method:
            raise RuntimeError("❌ No working CUDA methods found!")
        
        print(f"✅ Using method: {self.working_method}")
    
    def _find_working_method(self):
        """Find a working CUDA acceleration method"""
        
        # Method 1: Basic CUDA hwaccel (we know this works)
        if self._test_basic_cuda_hwaccel():
            print("✅ Method 1: Basic CUDA hwaccel - WORKING")
            return "basic_cuda"
        
        return None
    
    def _test_basic_cuda_hwaccel(self):
        """Test basic CUDA hwaccel (we know this works from your output)"""
        try:
            cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-f', 'lavfi', 
                '-i', 'testsrc=duration=1:size=64x64:rate=1',
                '-f', 'null', '-', '-loglevel', 'quiet'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def decode_video_batch(self, video_path, sample_rate=2.0, target_size=(640, 360)):
        """Decode video with working CUDA acceleration"""
        
        print(f"🚀 Using CUDA acceleration for {video_path}")
        
        # Get video info
        fps, duration, total_frames = self._get_video_info(video_path)
        if fps <= 0:
            raise RuntimeError(f"Could not read video: {video_path}")
        
        print(f"📹 Video: {fps:.1f}fps, {duration:.1f}s duration")
        
        try:
            return self._decode_with_basic_cuda(video_path, sample_rate, target_size, fps, duration)
        except Exception as e:
            print(f"❌ CUDA method failed: {e}")
            raise RuntimeError("CUDA decoding failed")
    
    def _decode_with_basic_cuda(self, video_path, sample_rate, target_size, fps, duration):
        """Decode with basic CUDA hwaccel (no CUVID)"""
        
        gpu_id = self.gpu_ids[self.current_gpu % len(self.gpu_ids)]
        self.current_gpu += 1
        
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-vf', f'scale_cuda={target_size[0]}:{target_size[1]}',
            '-r', str(sample_rate),
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-hide_banner',
            '-loglevel', 'warning',
            'pipe:1'
        ]
        
        return self._run_ffmpeg_decode(cmd, target_size, fps, duration)
    
    def _run_ffmpeg_decode(self, cmd, target_size, fps, duration):
        """Run FFmpeg command and collect frames"""
        
        print(f"🔧 Running CUDA decode...")
        
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
            
            if len(frames) % 60 == 0:
                print(f"  📊 CUDA decoded {len(frames)} frames...")
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            print(f"FFmpeg stderr: {error_msg}")
            raise RuntimeError(f"FFmpeg failed: {error_msg}")
        
        if not frames:
            raise RuntimeError("No frames decoded")
        
        print(f"✅ CUDA decoded {len(frames)} frames successfully!")
        
        frames_array = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        frame_indices = list(range(len(frames)))
        return frames_tensor, fps, duration, frame_indices
    
    def _get_video_info(self, video_path):
        """Get video information"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_format', '-show_streams', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                for stream in info['streams']:
                    if stream['codec_type'] == 'video':
                        fps_str = stream.get('r_frame_rate', '0/1')
                        if '/' in fps_str:
                            num, den = map(int, fps_str.split('/'))
                            fps = num / den if den != 0 else 0
                        else:
                            fps = float(fps_str)
                        
                        duration = float(stream.get('duration', 0))
                        total_frames = int(stream.get('nb_frames', fps * duration))
                        return fps, duration, total_frames
        except:
            pass
        
        return 0, 0, 0


def test_working_cuda(video_path):
    """Test the working CUDA decoder"""
    import time
    
    print("🚀 TESTING WORKING CUDA ACCELERATION")
    print("=" * 60)
    
    try:
        decoder = WorkingCUDADecoder(gpu_ids=[0, 1])
        
        print(f"\n🎬 Processing: {video_path}")
        start = time.time()
        
        frames, fps, duration, indices = decoder.decode_video_batch(
            video_path, sample_rate=2.0, target_size=(640, 360)
        )
        
        elapsed = time.time() - start
        
        print("=" * 60)
        print(f"✅ CUDA SUCCESS!")
        print(f"📊 Video duration: {duration:.1f}s")
        print(f"⚡ Processing time: {elapsed:.2f}s")
        print(f"🚀 Speed: {duration/elapsed:.1f}x real-time")
        print(f"🎯 Frames extracted: {len(frames)}")
        
        if elapsed < duration / 3:
            print(f"🔥 EXCELLENT CUDA PERFORMANCE!")
        else:
            print(f"✅ Good CUDA performance")
        
        return True
        
    except Exception as e:
        print(f"💥 FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_working_cuda(sys.argv[1])
    else:
        print("Usage: python working_cuda_decoder.py /path/to/video.mp4")