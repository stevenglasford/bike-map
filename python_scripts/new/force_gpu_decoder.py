import cv2
import numpy as np
import torch
import subprocess
import json
import tempfile
import os

class ForceGPUDecoder:
    """Force GPU acceleration with multiple fallback approaches"""
    
    def __init__(self, gpu_ids=[0]):
        self.gpu_ids = gpu_ids
        self.current_gpu = 0
        
        print("🔍 Testing GPU acceleration methods...")
        self.available_methods = self._test_gpu_methods()
        
        if not self.available_methods:
            raise RuntimeError("❌ NO GPU ACCELERATION AVAILABLE! Check your NVIDIA drivers and FFmpeg build.")
        
        print(f"✅ Found {len(self.available_methods)} working GPU method(s)")
    
    def _test_gpu_methods(self):
        """Test all possible GPU acceleration methods"""
        methods = []
        
        # Method 1: Basic NVDEC
        if self._test_basic_nvdec():
            methods.append("basic_nvdec")
            print("✅ Method 1: Basic NVDEC - WORKING")
        
        # Method 2: NVDEC with hwaccel_output_format
        if self._test_hwaccel_output():
            methods.append("hwaccel_output")
            print("✅ Method 2: NVDEC with hwaccel_output_format - WORKING")
        
        # Method 3: Direct CUVID without hwaccel
        if self._test_direct_cuvid():
            methods.append("direct_cuvid")
            print("✅ Method 3: Direct CUVID - WORKING")
        
        return methods
    
    def _test_basic_nvdec(self):
        """Test basic NVDEC functionality"""
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=64x64:rate=1',
                '-hwaccel', 'cuda', '-c:v', 'h264_cuvid',
                '-f', 'null', '-', '-loglevel', 'quiet'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _test_hwaccel_output(self):
        """Test NVDEC with hardware output format"""
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=64x64:rate=1',
                '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-c:v', 'h264_cuvid', '-f', 'null', '-', '-loglevel', 'quiet'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _test_direct_cuvid(self):
        """Test direct CUVID without hwaccel"""
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=64x64:rate=1',
                '-c:v', 'h264_cuvid', '-f', 'null', '-', '-loglevel', 'quiet'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def decode_video_batch(self, video_path, sample_rate=2.0, target_size=(640, 360)):
        """Decode video with forced GPU acceleration"""
        
        print(f"🚀 FORCING GPU acceleration for {video_path}")
        
        # Get video info
        fps, duration, total_frames = self._get_video_info(video_path)
        if fps <= 0:
            raise RuntimeError(f"Could not read video: {video_path}")
        
        codec_info = self._detect_video_codec(video_path)
        print(f"📹 Video: {fps:.1f}fps, {duration:.1f}s, codec: {codec_info}")
        
        # Try each GPU method until one works
        for method in self.available_methods:
            print(f"🔄 Trying GPU method: {method}")
            
            try:
                result = self._decode_with_method(video_path, method, sample_rate, target_size, fps, duration)
                if result[0] is not None:
                    print(f"✅ SUCCESS with method: {method}")
                    return result
            except Exception as e:
                print(f"❌ Method {method} failed: {e}")
                continue
        
        raise RuntimeError("🔥 ALL GPU METHODS FAILED!")
    
    def _decode_with_method(self, video_path, method, sample_rate, target_size, fps, duration):
        """Decode with specific GPU method"""
        
        codec_info = self._detect_video_codec(video_path)
        decoder = self._select_best_decoder(codec_info)
        gpu_id = self.gpu_ids[self.current_gpu % len(self.gpu_ids)]
        
        if method == "basic_nvdec":
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(gpu_id),
                '-c:v', decoder,
                '-i', video_path,
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]}',
                '-r', str(sample_rate),
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-hide_banner',
                '-loglevel', 'error',
                'pipe:1'
            ]
        
        elif method == "hwaccel_output":
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(gpu_id),
                '-hwaccel_output_format', 'cuda',
                '-c:v', decoder,
                '-i', video_path,
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]},hwdownload,format=rgb24',
                '-r', str(sample_rate),
                '-f', 'rawvideo',
                '-hide_banner',
                '-loglevel', 'error',
                'pipe:1'
            ]
        
        elif method == "direct_cuvid":
            cmd = [
                'ffmpeg',
                '-c:v', decoder,
                '-i', video_path,
                '-vf', f'scale_cuda={target_size[0]}:{target_size[1]},hwdownload,format=rgb24',
                '-r', str(sample_rate),
                '-f', 'rawvideo',
                '-hide_banner',
                '-loglevel', 'error',
                'pipe:1'
            ]
        
        print(f"🔧 Command: {' '.join(cmd[:8])}...")
        
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
                print(f"  📊 GPU decoded {len(frames)} frames...")
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            print(f"❌ FFmpeg error: {error_msg}")
            raise RuntimeError(f"GPU decoding failed: {error_msg}")
        
        if not frames:
            raise RuntimeError("No frames decoded")
        
        print(f"🎯 GPU decoded {len(frames)} frames successfully!")
        
        frames_array = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        
        frame_indices = list(range(len(frames)))
        return frames_tensor, fps, duration, frame_indices
    
    def _detect_video_codec(self, video_path):
        """Detect video codec"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_streams', '-select_streams', 'v:0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                if 'streams' in info and info['streams']:
                    codec = info['streams'][0].get('codec_name', 'unknown')
                    return codec
        except:
            pass
        return 'h264'
    
    def _select_best_decoder(self, codec_info):
        """Select NVIDIA decoder"""
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
    
    def _get_video_info(self, video_path):
        """Get video info"""
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


def test_force_gpu(video_path):
    """Test forced GPU acceleration"""
    import time
    
    print("🚀 FORCING GPU ACCELERATION TEST")
    print("=" * 60)
    
    try:
        decoder = ForceGPUDecoder(gpu_ids=[0])
        
        print(f"\n🎬 Processing: {video_path}")
        start = time.time()
        
        frames, fps, duration, indices = decoder.decode_video_batch(
            video_path, sample_rate=2.0, target_size=(640, 360)
        )
        
        elapsed = time.time() - start
        
        print("=" * 60)
        print(f"✅ GPU ACCELERATION SUCCESS!")
        print(f"📊 Video duration: {duration:.1f}s")
        print(f"⚡ Processing time: {elapsed:.2f}s")
        print(f"🚀 Speed: {duration/elapsed:.1f}x real-time")
        print(f"🎯 Frames extracted: {len(frames)}")
        
        if elapsed < duration / 5:
            print(f"🔥 EXCELLENT GPU PERFORMANCE!")
        else:
            print(f"⚠️  GPU performance could be better")
        
        return True
        
    except Exception as e:
        print(f"💥 TOTAL FAILURE: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_force_gpu(sys.argv[1])
    else:
        print("Usage: python force_gpu_decoder.py /path/to/video.mp4")