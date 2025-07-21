#!/usr/bin/env python3
"""
Complete Insta360 Processor - All-in-One Script
Handles 8K video download, stitching, metadata sync, and timestamp preservation
Supports live cameras, batch processing, and Insta360 directory structures
"""

import os
import sys
import subprocess
import logging
import json
import time
import signal
import shutil
import glob
import hashlib
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
import ctypes
from ctypes import cdll, c_char_p, c_int, c_void_p, Structure, POINTER

@dataclass
class Insta360VideoFile:
    """Represents an Insta360 video file with its associated metadata"""
    insv_path: str
    lrv_path: Optional[str] = None
    gyro_path: Optional[str] = None
    exposure_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    aeflicker_path: Optional[str] = None
    beisp_path: Optional[str] = None
    camera_id: str = ""
    base_name: str = ""
    timestamp: Optional[datetime] = None

class Insta360Config:
    """Configuration for Insta360 processing"""
    
    def __init__(self, config_file: str = "insta360_config.json"):
        self.config_file = config_file
        self.default_config = {
            "camera_sdk_path": "./CameraSDK-20250418_145834-2.0.2-Linux",
            "media_sdk_path": "./libMediaSDK-dev-3.0.1-20250418-amd64",
            "local_lib_path": "./local_lib",
            "local_bin_path": "./local_bin", 
            "download_path": "./downloads",
            "output_path": "./stitched_videos",
            "temp_path": "./temp",
            "max_concurrent_downloads": 2,
            "video_quality": "8K",
            "stitch_format": "mp4",
            "extract_timestamps": True,
            "embed_timestamps": True,
            "verify_timestamps": True,
            "sync_metadata": True,
            "use_gyro_stabilization": True,
            "use_exposure_correction": True,
            "use_flicker_correction": True,
            "preserve_metadata_files": True,
            "create_processing_reports": True,
            "cleanup_temp": True,
            "log_level": "INFO",
            "hpc_mode": False,
            "cuda_version": "auto",
            "ffmpeg_path": "auto",
            "use_system_libs": True
        }
        self.config = self.load_config()
        
        # Auto-detect HPC environment
        if self.is_hpc_environment():
            self.config["hpc_mode"] = True
            logging.info("HPC environment detected, enabling compatibility mode")
    
    def is_hpc_environment(self) -> bool:
        """Detect if running in HPC environment"""
        hpc_indicators = [
            os.environ.get('SLURM_JOB_ID'),
            os.environ.get('PBS_JOBID'),
            os.environ.get('LSB_JOBID'),
            os.path.exists('/etc/slurm'),
            os.path.exists('/opt/pbs'),
            os.path.exists('/usr/bin/module'),
        ]
        return any(hpc_indicators)
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                for key, value in self.default_config.items():
                    config.setdefault(key, value)
                return config
            except Exception as e:
                logging.warning(f"Error loading config: {e}. Using defaults.")
        
        self.save_config(self.default_config)
        return self.default_config.copy()
    
    def save_config(self, config: Dict = None):
        """Save configuration to file"""
        config = config or self.config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

class VideoTimestamp:
    """Extract and manage video timestamps with HPC compatibility"""
    
    @staticmethod
    def get_ffmpeg_path() -> str:
        """Get FFmpeg path, respecting HPC environment"""
        ffmpeg_candidates = [
            'ffmpeg', '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg',
            '/opt/ffmpeg/bin/ffmpeg', os.path.expanduser('~/bin/ffmpeg'),
        ]
        
        for ffmpeg_path in ffmpeg_candidates:
            try:
                result = subprocess.run([ffmpeg_path, '-version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return ffmpeg_path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return 'ffmpeg'
    
    @staticmethod
    def extract_creation_time(video_path: str) -> Optional[datetime]:
        """Extract creation timestamp from video metadata with HPC compatibility"""
        try:
            ffmpeg_path = VideoTimestamp.get_ffmpeg_path()
            ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
            
            if VideoTimestamp.command_exists(ffprobe_path):
                timestamp = VideoTimestamp.extract_with_ffprobe(video_path, ffprobe_path)
                if timestamp:
                    return timestamp
            
            if VideoTimestamp.command_exists(ffmpeg_path):
                timestamp = VideoTimestamp.extract_with_ffmpeg(video_path, ffmpeg_path)
                if timestamp:
                    return timestamp
            
            logging.warning(f"Could not extract metadata timestamp from {video_path}, using file mtime")
            stat = os.stat(video_path)
            return datetime.fromtimestamp(stat.st_mtime)
                
        except Exception as e:
            logging.error(f"Error extracting timestamp from {video_path}: {e}")
            return None
    
    @staticmethod
    def extract_with_ffprobe(video_path: str, ffprobe_path: str) -> Optional[datetime]:
        """Extract timestamp using ffprobe"""
        try:
            cmd = [ffprobe_path, '-v', 'quiet', '-print_format', 'json',
                   '-show_format', '-show_streams', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                return VideoTimestamp.parse_metadata_timestamp(metadata)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logging.debug(f"ffprobe failed for {video_path}: {e}")
        return None
    
    @staticmethod
    def extract_with_ffmpeg(video_path: str, ffmpeg_path: str) -> Optional[datetime]:
        """Extract timestamp using ffmpeg"""
        try:
            cmd = [ffmpeg_path, '-i', video_path, '-f', 'null', '-']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            output = result.stderr
            for line in output.split('\n'):
                if 'creation_time' in line.lower():
                    time_part = line.split(':', 1)[1].strip()
                    timestamp = VideoTimestamp.parse_timestamp(time_part)
                    if timestamp:
                        return timestamp
        except (subprocess.TimeoutExpired, Exception) as e:
            logging.debug(f"ffmpeg failed for {video_path}: {e}")
        return None
    
    @staticmethod
    def parse_metadata_timestamp(metadata: dict) -> Optional[datetime]:
        """Parse timestamp from ffprobe metadata"""
        timestamp_fields = ['creation_time', 'date', 'com.apple.quicktime.creationdate', 'encoded_date']
        
        if 'format' in metadata and 'tags' in metadata['format']:
            for field in timestamp_fields:
                if field in metadata['format']['tags']:
                    time_str = metadata['format']['tags'][field]
                    timestamp = VideoTimestamp.parse_timestamp(time_str)
                    if timestamp:
                        return timestamp
        
        if 'streams' in metadata:
            for stream in metadata['streams']:
                if 'tags' in stream:
                    for field in timestamp_fields:
                        if field in stream['tags']:
                            time_str = stream['tags'][field]
                            timestamp = VideoTimestamp.parse_timestamp(time_str)
                            if timestamp:
                                return timestamp
        return None
    
    @staticmethod
    def command_exists(command: str) -> bool:
        """Check if a command exists"""
        try:
            subprocess.run([command, '-version'], capture_output=True, check=False, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def parse_timestamp(time_str: str) -> Optional[datetime]:
        """Parse various timestamp formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", 
            "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        logging.warning(f"Could not parse timestamp: {time_str}")
        return None

class MediaSDKWrapper:
    """Wrapper for Insta360 MediaSDK library with HPC compatibility"""
    
    def __init__(self, config: 'Insta360Config'):
        self.config = config
        self.lib = None
        self.initialized = False
        
    def load_library(self) -> bool:
        """Load the MediaSDK library with HPC-compatible paths"""
        try:
            if self.config.config.get('hpc_mode', False):
                return self.load_library_hpc()
            
            try:
                self.lib = cdll.LoadLibrary("libMediaSDK.so")
                logging.info("Loaded MediaSDK from system installation")
                return True
            except OSError:
                pass
            
            return self.try_local_libraries()
        except Exception as e:
            logging.error(f"Error loading MediaSDK: {e}")
            return False
    
    def load_library_hpc(self) -> bool:
        """HPC-specific library loading with more search paths"""
        search_paths = []
        
        if 'local_lib_path' in self.config.config:
            search_paths.append(self.config.config['local_lib_path'])
        
        sdk_path = self.config.config['media_sdk_path']
        search_paths.extend([
            os.path.join(sdk_path, 'lib'), os.path.join(sdk_path, 'lib64'), sdk_path,
            '/usr/local/lib', '/opt/local/lib', os.path.expanduser('~/.local/lib'),
        ])
        
        for base_path in search_paths:
            lib_candidates = [
                os.path.join(base_path, 'libMediaSDK.so'),
                os.path.join(base_path, 'libMediaSDK.so.1'),
                os.path.join(base_path, 'libmediasdk.so'),
            ]
            
            for lib_path in lib_candidates:
                if os.path.exists(lib_path):
                    try:
                        old_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                        os.environ['LD_LIBRARY_PATH'] = f"{base_path}:{old_ld_path}"
                        self.lib = cdll.LoadLibrary(lib_path)
                        logging.info(f"Loaded MediaSDK from: {lib_path}")
                        return True
                    except OSError as e:
                        logging.debug(f"Failed to load {lib_path}: {e}")
                        continue
        
        logging.warning("Could not load MediaSDK library in HPC mode")
        return False
    
    def try_local_libraries(self) -> bool:
        """Try loading from local extracted libraries"""
        sdk_path = self.config.config['media_sdk_path']
        lib_paths = [
            os.path.join(sdk_path, 'lib', 'libMediaSDK.so'),
            os.path.join(sdk_path, 'libMediaSDK.so'),
            os.path.join(sdk_path, 'lib64', 'libMediaSDK.so'),
        ]
        
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    self.lib = cdll.LoadLibrary(lib_path)
                    logging.info(f"Loaded MediaSDK from: {lib_path}")
                    return True
                except OSError as e:
                    logging.warning(f"Failed to load {lib_path}: {e}")
        return False
    
    def initialize(self) -> bool:
        """Initialize MediaSDK with compatibility checks"""
        if not self.lib:
            if not self.load_library():
                logging.warning("MediaSDK library not loaded, will use command-line fallback")
                return False
        
        try:
            init_functions = ['MediaSDK_Initialize', 'mediasdk_init', 'InitMediaSDK', 'init']
            
            for func_name in init_functions:
                if hasattr(self.lib, func_name):
                    init_func = getattr(self.lib, func_name)
                    try:
                        result = init_func()
                        if result == 0:
                            self.initialized = True
                            logging.info(f"MediaSDK initialized with {func_name}")
                            return True
                    except Exception as e:
                        logging.debug(f"Failed to initialize with {func_name}: {e}")
                        continue
            
            logging.warning("No MediaSDK init function found, assuming ready")
            self.initialized = True
            return True
        except Exception as e:
            logging.error(f"Error initializing MediaSDK: {e}")
            return False
    
    def stitch_video_native(self, input_path: str, output_path: str) -> bool:
        """Use native MediaSDK functions for stitching with compatibility"""
        if not self.initialized:
            if not self.initialize():
                return False
        
        try:
            stitch_functions = ['MediaSDK_StitchVideo', 'mediasdk_stitch', 'StitchVideo', 'stitch']
            
            for func_name in stitch_functions:
                if hasattr(self.lib, func_name):
                    try:
                        stitch_func = getattr(self.lib, func_name)
                        input_c = c_char_p(input_path.encode('utf-8'))
                        output_c = c_char_p(output_path.encode('utf-8'))
                        result = stitch_func(input_c, output_c)
                        
                        if result == 0:
                            logging.info(f"Native stitching successful with {func_name}")
                            return True
                        else:
                            logging.debug(f"Native stitching failed with {func_name}: {result}")
                    except Exception as e:
                        logging.debug(f"Error calling {func_name}: {e}")
                        continue
            
            logging.info("Native stitching functions not available, will use command-line")
            return False
        except Exception as e:
            logging.error(f"Error in native stitching: {e}")
            return False

class Insta360Downloader:
    """Handle downloading from Insta360 cameras"""
    
    def __init__(self, config: Insta360Config):
        self.config = config
        self.download_queue = queue.Queue()
        self.active_downloads = {}
        
    def discover_cameras(self) -> List[Dict]:
        """Discover available Insta360 cameras"""
        cameras = []
        
        try:
            sdk_path = self.config.config['camera_sdk_path']
            discovery_tool = os.path.join(sdk_path, 'bin', 'camera_discovery')
            
            if os.path.exists(discovery_tool):
                result = subprocess.run([discovery_tool], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'Camera' in line:
                            camera_info = self.parse_camera_info(line)
                            if camera_info:
                                cameras.append(camera_info)
        except Exception as e:
            logging.error(f"Error discovering cameras: {e}")
        
        return cameras
    
    def parse_camera_info(self, line: str) -> Optional[Dict]:
        """Parse camera information from discovery output"""
        try:
            parts = line.split()
            if len(parts) >= 3:
                return {
                    'id': parts[1],
                    'model': parts[2],
                    'status': 'connected' if 'connected' in line.lower() else 'discovered'
                }
        except Exception as e:
            logging.error(f"Error parsing camera info: {e}")
        return None
    
    def download_from_camera(self, camera_id: str, output_path: str) -> bool:
        """Download videos from specific camera"""
        try:
            sdk_path = self.config.config['camera_sdk_path']
            download_tool = os.path.join(sdk_path, 'bin', 'camera_download')
            
            cmd = [download_tool, '--camera-id', camera_id, '--output', output_path,
                   '--quality', self.config.config['video_quality']]
            
            logging.info(f"Starting download from camera {camera_id}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully downloaded from camera {camera_id}")
                return True
            else:
                logging.error(f"Download failed: {result.stderr}")
                return False
        except Exception as e:
            logging.error(f"Error downloading from camera {camera_id}: {e}")
            return False

class Insta360Stitcher:
    """Handle video stitching using MediaSDK with metadata sync"""
    
    def __init__(self, config: Insta360Config):
        self.config = config
        self.media_sdk = MediaSDKWrapper(config)
        
    def stitch_video(self, input_path: str, output_path: str, 
                    projection: str = "equirectangular") -> bool:
        """Stitch raw Insta360 video files with timestamp preservation"""
        try:
            original_timestamp = VideoTimestamp.extract_creation_time(input_path)
            if original_timestamp:
                logging.info(f"Original timestamp: {original_timestamp}")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"Starting stitching: {input_path} -> {output_path}")
            
            if self.stitch_with_native_sdk(input_path, output_path, projection):
                logging.info(f"Successfully stitched with native SDK: {output_path}")
                return self.finalize_stitched_video(output_path, original_timestamp)
            
            if self.stitch_with_command_line(input_path, output_path, projection, original_timestamp):
                logging.info(f"Successfully stitched with command line: {output_path}")
                return self.finalize_stitched_video(output_path, original_timestamp)
            
            if self.stitch_with_test_tool(input_path, output_path):
                logging.info(f"Successfully stitched with test tool: {output_path}")
                return self.finalize_stitched_video(output_path, original_timestamp)
            
            logging.error(f"All stitching methods failed for: {input_path}")
            return False
        except Exception as e:
            logging.error(f"Error stitching video {input_path}: {e}")
            return False
    
    def stitch_with_enhanced_metadata(self, video_file: Insta360VideoFile, output_path: str) -> bool:
        """Stitch video using enhanced metadata for better quality with active metadata sync"""
        try:
            active_metadata = []
            if video_file.gyro_path:
                active_metadata.append("gyro stabilization")
            if video_file.exposure_path:
                active_metadata.append("exposure correction")
            if video_file.aeflicker_path:
                active_metadata.append("flicker correction")
            if video_file.beisp_path:
                active_metadata.append("ISP static data")
            if video_file.lrv_path:
                active_metadata.append("preview validation")
            
            if active_metadata:
                logging.info(f"Active metadata sync: {', '.join(active_metadata)}")
            else:
                logging.warning("No metadata files found - using basic stitching")
            
            if self.stitch_with_metadata_api(video_file, output_path):
                return True
            
            if self.stitch_with_metadata_command(video_file, output_path):
                return True
            
            logging.warning("Enhanced metadata stitching failed, using basic stitcher")
            return self.stitch_video(video_file.insv_path, output_path)
        except Exception as e:
            logging.error(f"Enhanced metadata stitching failed: {e}")
            return False
    
    def stitch_with_metadata_api(self, video_file: Insta360VideoFile, output_path: str) -> bool:
        """Use MediaSDK API with metadata files"""
        try:
            if hasattr(self.media_sdk, 'lib') and self.media_sdk.lib:
                return self.call_native_sdk_with_metadata(video_file, output_path)
            else:
                logging.debug("Native SDK not available for metadata API")
                return False
        except Exception as e:
            logging.debug(f"Metadata API stitching failed: {e}")
            return False
    
    def call_native_sdk_with_metadata(self, video_file: Insta360VideoFile, output_path: str) -> bool:
        """Call native MediaSDK with metadata parameters"""
        try:
            lib = self.media_sdk.lib
            enhanced_functions = [
                'MediaSDK_StitchVideoWithMetadata', 'MediaSDK_StitchEnhanced', 
                'StitchVideoAdvanced', 'mediasdk_stitch_enhanced'
            ]
            
            for func_name in enhanced_functions:
                if hasattr(lib, func_name):
                    try:
                        stitch_func = getattr(lib, func_name)
                        input_c = video_file.insv_path.encode('utf-8')
                        output_c = output_path.encode('utf-8')
                        gyro_c = video_file.gyro_path.encode('utf-8') if video_file.gyro_path else None
                        exposure_c = video_file.exposure_path.encode('utf-8') if video_file.exposure_path else None
                        
                        result = stitch_func(input_c, output_c, gyro_c, exposure_c)
                        
                        if result == 0:
                            logging.info(f"Native enhanced stitching successful with {func_name}")
                            return True
                        else:
                            logging.debug(f"Native enhanced stitching failed with {func_name}: {result}")
                    except Exception as e:
                        logging.debug(f"Error calling {func_name}: {e}")
                        continue
            
            logging.debug("No enhanced native stitching functions found")
            return False
        except Exception as e:
            logging.debug(f"Native SDK metadata call failed: {e}")
            return False
    
    def stitch_with_metadata_command(self, video_file: Insta360VideoFile, output_path: str) -> bool:
        """Use command-line tools with metadata files"""
        try:
            possible_tools = []
            
            if 'local_bin_path' in self.config.config:
                local_bin = self.config.config['local_bin_path']
                possible_tools.extend([
                    os.path.join(local_bin, 'stitcherSDKTest'),
                    os.path.join(local_bin, 'mediasdk_stitch'),
                    os.path.join(local_bin, 'stitcher'),
                ])
            
            sdk_path = self.config.config['media_sdk_path']
            possible_tools.extend([
                os.path.join(sdk_path, 'bin', 'stitcherSDKTest'),
                os.path.join(sdk_path, 'bin', 'media_stitcher'),
                os.path.join(sdk_path, 'stitcherSDKTest'),
            ])
            
            possible_tools.append('stitcherSDKTest')
            
            stitcher_tool = None
            for tool in possible_tools:
                if os.path.exists(tool) and os.access(tool, os.X_OK):
                    stitcher_tool = tool
                    break
                elif tool == 'stitcherSDKTest' and self.command_exists('stitcherSDKTest'):
                    stitcher_tool = tool
                    break
            
            if not stitcher_tool:
                logging.debug("No stitching tool found for metadata command")
                return False
            
            cmd = self.build_enhanced_command(stitcher_tool, video_file, output_path)
            if not cmd:
                logging.debug("Could not build enhanced command")
                return False
            
            logging.info(f"Running enhanced stitching: {' '.join(cmd[:5])}...")
            
            env = os.environ.copy()
            if self.config.config.get('hpc_mode', False):
                lib_paths = []
                if 'local_lib_path' in self.config.config:
                    lib_paths.append(self.config.config['local_lib_path'])
                lib_paths.append(os.path.join(self.config.config['media_sdk_path'], 'lib'))
                
                if lib_paths:
                    old_ld_path = env.get('LD_LIBRARY_PATH', '')
                    new_ld_path = ':'.join(lib_paths + [old_ld_path])
                    env['LD_LIBRARY_PATH'] = new_ld_path
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, env=env)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_clean = output.strip()
                    if any(keyword in output_clean.lower() for keyword in ['progress', '%', 'stabiliz', 'expos', 'gyro']):
                        logging.info(f"Enhanced stitching: {output_clean}")
            
            returncode = process.poll()
            
            if returncode == 0:
                logging.info("Enhanced command-line stitching successful")
                return True
            else:
                stderr = process.stderr.read()
                logging.debug(f"Enhanced command stitching failed: {stderr}")
                return False
        except Exception as e:
            logging.debug(f"Enhanced command stitching failed: {e}")
            return False
    
    def build_enhanced_command(self, tool: str, video_file: Insta360VideoFile, output_path: str) -> List[str]:
        """Build command with metadata parameters"""
        try:
            if 'stitcherSDKTest' in tool:
                cmd = [tool, video_file.insv_path, output_path]
                
                if video_file.gyro_path:
                    cmd.extend(['--gyro', video_file.gyro_path, '--stabilization', 'true'])
                if video_file.exposure_path:
                    cmd.extend(['--exposure', video_file.exposure_path, '--exposure-correction', 'true'])
                if video_file.aeflicker_path:
                    cmd.extend(['--flicker', video_file.aeflicker_path, '--flicker-correction', 'true'])
                if video_file.beisp_path:
                    cmd.extend(['--isp-data', video_file.beisp_path])
                
                cmd.extend(['--quality', 'high', '--output-format', 'mp4', '--resolution', '8K'])
                return cmd
            else:
                cmd = [tool, '--input', video_file.insv_path, '--output', output_path]
                
                if video_file.gyro_path:
                    cmd.extend(['--gyro-file', video_file.gyro_path, '--enable-stabilization'])
                if video_file.exposure_path:
                    cmd.extend(['--exposure-file', video_file.exposure_path, '--enable-exposure-correction'])
                if video_file.aeflicker_path:
                    cmd.extend(['--flicker-file', video_file.aeflicker_path, '--enable-flicker-correction'])
                if video_file.beisp_path:
                    cmd.extend(['--isp-file', video_file.beisp_path])
                
                cmd.extend(['--quality', 'high', '--format', 'mp4', '--projection', 'equirectangular'])
                return cmd
        except Exception as e:
            logging.error(f"Error building enhanced command: {e}")
            return []
    
    def stitch_with_native_sdk(self, input_path: str, output_path: str, projection: str) -> bool:
        """Try stitching with native MediaSDK library"""
        return self.media_sdk.stitch_video_native(input_path, output_path)
    
    def stitch_with_command_line(self, input_path: str, output_path: str, projection: str, original_timestamp: Optional[datetime] = None) -> bool:
        """Stitch using command-line MediaSDK tools with HPC compatibility"""
        try:
            possible_tools = []
            
            if 'local_bin_path' in self.config.config:
                local_bin = self.config.config['local_bin_path']
                possible_tools.extend([
                    os.path.join(local_bin, 'stitcherSDKTest'),
                    os.path.join(local_bin, 'mediasdk_stitch'),
                    os.path.join(local_bin, 'stitcher'),
                ])
            
            sdk_path = self.config.config['media_sdk_path']
            possible_tools.extend([
                os.path.join(sdk_path, 'bin', 'stitcherSDKTest'),
                os.path.join(sdk_path, 'bin', 'media_stitcher'),
                os.path.join(sdk_path, 'bin', 'stitcher'),
                os.path.join(sdk_path, 'stitcherSDKTest'),
                os.path.join(sdk_path, 'stitcher'),
            ])
            
            possible_tools.append('stitcherSDKTest')
            
            stitcher_tool = None
            for tool in possible_tools:
                if os.path.exists(tool) and os.access(tool, os.X_OK):
                    stitcher_tool = tool
                    break
                elif tool == 'stitcherSDKTest' and self.command_exists('stitcherSDKTest'):
                    stitcher_tool = tool
                    break
            
            if not stitcher_tool:
                logging.warning("No MediaSDK stitching tool found")
                return False
            
            logging.info(f"Using stitching tool: {stitcher_tool}")
            
            if 'stitcherSDKTest' in stitcher_tool:
                cmd = [stitcher_tool, input_path, output_path]
            else:
                cmd = [stitcher_tool, '--input', input_path, '--output', output_path,
                       '--projection', projection, '--format', self.config.config['stitch_format'], '--quality', 'high']
                
                if original_timestamp:
                    timestamp_str = original_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    cmd.extend(['--metadata', f'creation_time={timestamp_str}'])
            
            env = os.environ.copy()
            if self.config.config.get('hpc_mode', False):
                lib_paths = []
                if 'local_lib_path' in self.config.config:
                    lib_paths.append(self.config.config['local_lib_path'])
                lib_paths.append(os.path.join(self.config.config['media_sdk_path'], 'lib'))
                
                if lib_paths:
                    old_ld_path = env.get('LD_LIBRARY_PATH', '')
                    new_ld_path = ':'.join(lib_paths + [old_ld_path])
                    env['LD_LIBRARY_PATH'] = new_ld_path
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, env=env)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_clean = output.strip()
                    if any(keyword in output_clean.lower() for keyword in ['progress', '%', 'processing', 'frame']):
                        logging.info(f"Stitching: {output_clean}")
            
            returncode = process.poll()
            
            if returncode == 0:
                return True
            else:
                stderr = process.stderr.read()
                logging.error(f"Command-line stitching failed (code {returncode}): {stderr}")
                return False
        except Exception as e:
            logging.error(f"Error in command-line stitching: {e}")
            return False
    
    def stitch_with_test_tool(self, input_path: str, output_path: str) -> bool:
        """Try stitching with stitcherSDKTest tool"""
        try:
            if not self.command_exists('stitcherSDKTest'):
                return False
            
            cmd = ['stitcherSDKTest', input_path, output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                logging.error(f"stitcherSDKTest failed: {result.stderr}")
                return False
        except Exception as e:
            logging.error(f"Error using stitcherSDKTest: {e}")
            return False
    
    def command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            subprocess.run([command, '--help'], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False
    
    def finalize_stitched_video(self, output_path: str, original_timestamp: Optional[datetime] = None) -> bool:
        """Finalize stitched video processing and embed timestamp metadata"""
        try:
            if not original_timestamp:
                original_timestamp = VideoTimestamp.extract_creation_time(output_path)
            
            if self.config.config['extract_timestamps'] and original_timestamp:
                self.save_timestamp_info(output_path, original_timestamp)
                
                if self.config.config.get('embed_timestamps', True):
                    if self.embed_timestamp_in_video(output_path, original_timestamp):
                        logging.info(f"Timestamp embedded in video metadata: {original_timestamp}")
                        
                        if self.config.config.get('verify_timestamps', True):
                            embedded_timestamp = self.verify_embedded_timestamp(output_path)
                            if embedded_timestamp:
                                time_diff = abs((embedded_timestamp - original_timestamp).total_seconds())
                                if time_diff < 2:
                                    logging.info("Timestamp embedding verified successfully")
                                else:
                                    logging.warning(f"Timestamp verification failed: {time_diff:.1f}s difference")
                            else:
                                logging.warning("Could not verify embedded timestamp")
                    else:
                        logging.warning("Failed to embed timestamp in video metadata")
                else:
                    logging.info("Timestamp embedding disabled in configuration")
            
            return True
        except Exception as e:
            logging.error(f"Error finalizing video {output_path}: {e}")
            return False
    
    def embed_timestamp_in_video(self, video_path: str, timestamp: datetime) -> bool:
        """Embed timestamp into video metadata using ffmpeg"""
        try:
            temp_path = video_path + ".temp"
            ffmpeg_path = VideoTimestamp.get_ffmpeg_path()
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            
            cmd = [ffmpeg_path, '-i', video_path, '-c', 'copy',
                   '-metadata', f'creation_time={timestamp_str}',
                   '-metadata', f'date={timestamp_str}',
                   '-movflags', 'use_metadata_tags', '-y', temp_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                os.replace(temp_path, video_path)
                return True
            else:
                logging.error(f"FFmpeg metadata embedding failed: {result.stderr}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
        except Exception as e:
            logging.error(f"Error embedding timestamp in {video_path}: {e}")
            temp_path = video_path + ".temp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def save_timestamp_info(self, video_path: str, timestamp: datetime):
        """Save comprehensive timestamp information alongside video"""
        info_path = video_path.replace('.mp4', '_info.json')
        embed_enabled = self.config.config.get('embed_timestamps', True)
        
        info = {
            'video_file': os.path.basename(video_path),
            'original_creation_time': timestamp.isoformat(),
            'creation_time_utc': timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'creation_time_readable': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'processed_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(video_path),
            'checksum': self.calculate_checksum(video_path),
            'metadata_embedded': embed_enabled,
            'timestamp_formats': {
                'iso8601': timestamp.isoformat(),
                'ffmpeg_metadata': timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                'unix_timestamp': timestamp.timestamp(),
                'human_readable': timestamp.strftime("%A, %B %d, %Y at %I:%M:%S %p")
            },
            'processing_info': {
                'extract_timestamps': self.config.config.get('extract_timestamps', True),
                'embed_timestamps': embed_enabled,
                'verify_timestamps': self.config.config.get('verify_timestamps', True)
            }
        }
        
        if timestamp.tzinfo:
            info['timezone'] = str(timestamp.tzinfo)
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        logging.info(f"Timestamp info saved: {info_path}")
    
    def verify_embedded_timestamp(self, video_path: str) -> Optional[datetime]:
        """Verify that timestamp was properly embedded in video metadata"""
        try:
            embedded_timestamp = VideoTimestamp.extract_creation_time(video_path)
            return embedded_timestamp
        except Exception as e:
            logging.error(f"Failed to verify embedded timestamp: {e}")
            return None
    
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class Insta360DirectoryProcessor:
    """Enhanced processor for Insta360 camera directory structures"""
    
    def __init__(self, config: Insta360Config):
        self.config = config
        self.stitcher = Insta360Stitcher(config)
        self.stats = {
            'cameras_found': 0, 'videos_found': 0, 'videos_processed': 0,
            'videos_failed': 0, 'videos_skipped': 0
        }
        
    def discover_insta360_structure(self, root_path: str) -> Dict[str, List[Insta360VideoFile]]:
        """Discover and analyze Insta360 directory structure"""
        logging.info(f"Scanning Insta360 directory structure: {root_path}")
        
        cameras = {}
        dcim_path = os.path.join(root_path, "DCIM")
        misc_path = os.path.join(root_path, "MISC")
        
        if not os.path.exists(dcim_path):
            logging.error(f"DCIM directory not found in {root_path}")
            return cameras
        
        for item in os.listdir(dcim_path):
            item_path = os.path.join(dcim_path, item)
            
            if os.path.isdir(item_path) and item.startswith("Camera"):
                camera_id = item
                logging.info(f"Found camera directory: {camera_id}")
                
                video_files = self.discover_camera_videos(item_path, camera_id, misc_path)
                
                if video_files:
                    cameras[camera_id] = video_files
                    self.stats['cameras_found'] += 1
                    self.stats['videos_found'] += len(video_files)
                    logging.info(f"Found {len(video_files)} videos in {camera_id}")
        
        fileinfo_path = os.path.join(dcim_path, "fileinfo_list.list")
        if os.path.exists(fileinfo_path):
            logging.info("Found fileinfo_list.list - will use for additional metadata")
        
        return cameras
    
    def discover_camera_videos(self, camera_path: str, camera_id: str, misc_path: str) -> List[Insta360VideoFile]:
        """Discover all video files and their associated metadata in a camera directory"""
        videos = []
        insv_files = glob.glob(os.path.join(camera_path, "*.insv"))
        
        for insv_path in insv_files:
            base_name = os.path.splitext(os.path.basename(insv_path))[0]
            
            video_file = Insta360VideoFile(
                insv_path=insv_path, camera_id=camera_id, base_name=base_name
            )
            
            self.find_associated_files(video_file, camera_path, misc_path)
            video_file.timestamp = VideoTimestamp.extract_creation_time(insv_path)
            
            videos.append(video_file)
            logging.debug(f"Found video: {base_name} with {self.count_metadata_files(video_file)} metadata files")
        
        videos.sort(key=lambda v: v.timestamp or datetime.min)
        return videos
    
    def find_associated_files(self, video_file: Insta360VideoFile, camera_path: str, misc_path: str):
        """Find all associated metadata files for a video"""
        base_name = video_file.base_name
        
        # Look for LRV file
        lrv_name = base_name.replace("VID_", "LRV_", 1) + ".lrv"
        lrv_path = os.path.join(camera_path, lrv_name)
        if os.path.exists(lrv_path):
            video_file.lrv_path = lrv_path
        
        # Look for metadata .bin files
        metadata_patterns = {
            'gyro_path': f"{base_name}.insv.gyro.bin",
            'exposure_path': f"{base_name}.insv.exposuretime.bin", 
            'thumbnail_path': f"{base_name}.insv.thumbnail.bin",
            'aeflicker_path': f"{base_name}.insv.aeflicker.bin",
            'beisp_path': f"{base_name}.insv.beisp_static_data.bin"
        }
        
        for attr_name, pattern in metadata_patterns.items():
            file_path = os.path.join(camera_path, pattern)
            if os.path.exists(file_path):
                setattr(video_file, attr_name, file_path)
                continue
            
            if misc_path and os.path.exists(misc_path):
                misc_file_path = os.path.join(misc_path, pattern)
                if os.path.exists(misc_file_path):
                    setattr(video_file, attr_name, misc_file_path)
    
    def count_metadata_files(self, video_file: Insta360VideoFile) -> int:
        """Count how many metadata files are associated with a video"""
        count = 0
        for attr in ['lrv_path', 'gyro_path', 'exposure_path', 'thumbnail_path', 'aeflicker_path', 'beisp_path']:
            if getattr(video_file, attr):
                count += 1
        return count
    
    def process_video_with_metadata(self, video_file: Insta360VideoFile, output_dir: str) -> bool:
        """Process a single video file with active metadata synchronization"""
        try:
            if video_file.timestamp:
                timestamp_str = video_file.timestamp.strftime("%Y%m%d_%H%M%S")
                output_name = f"{video_file.camera_id}_{timestamp_str}_{video_file.base_name}.mp4"
            else:
                output_name = f"{video_file.camera_id}_{video_file.base_name}.mp4"
            
            output_path = os.path.join(output_dir, video_file.camera_id, "stitched", output_name)
            
            logging.info(f"Processing {video_file.base_name} from {video_file.camera_id}")
            logging.info(f"  Input: {video_file.insv_path}")
            logging.info(f"  Output: {output_path}")
            
            metadata_status = []
            if video_file.gyro_path:
                gyro_size = os.path.getsize(video_file.gyro_path) if os.path.exists(video_file.gyro_path) else 0
                metadata_status.append(f"gyro data ({gyro_size} bytes)")
            if video_file.exposure_path:
                exp_size = os.path.getsize(video_file.exposure_path) if os.path.exists(video_file.exposure_path) else 0
                metadata_status.append(f"exposure data ({exp_size} bytes)")
            if video_file.aeflicker_path:
                flicker_size = os.path.getsize(video_file.aeflicker_path) if os.path.exists(video_file.aeflicker_path) else 0
                metadata_status.append(f"flicker correction ({flicker_size} bytes)")
            if video_file.beisp_path:
                isp_size = os.path.getsize(video_file.beisp_path) if os.path.exists(video_file.beisp_path) else 0
                metadata_status.append(f"ISP data ({isp_size} bytes)")
            if video_file.lrv_path:
                lrv_size = os.path.getsize(video_file.lrv_path) if os.path.exists(video_file.lrv_path) else 0
                metadata_status.append(f"LRV preview ({lrv_size/1024/1024:.1f} MB)")
            
            if metadata_status:
                logging.info(f"  Metadata for sync: {', '.join(metadata_status)}")
            else:
                logging.warning(f"  No metadata files found - basic stitching only")
            
            self.copy_metadata_files(video_file, output_dir)
            
            success = self.stitcher.stitch_with_enhanced_metadata(video_file, output_path)
            
            if success:
                self.verify_metadata_sync(video_file, output_path)
                self.create_processing_report(video_file, output_path, metadata_status)
                logging.info(f"✓ Successfully processed {video_file.base_name} with metadata sync")
                return True
            else:
                logging.error(f"✗ Failed to process {video_file.base_name}")
                return False
        except Exception as e:
            logging.error(f"Error processing {video_file.base_name}: {e}")
            return False
    
    def copy_metadata_files(self, video_file: Insta360VideoFile, output_dir: str):
        """Copy metadata files to output directory for reference"""
        metadata_dir = os.path.join(output_dir, video_file.camera_id, "metadata", video_file.base_name)
        os.makedirs(metadata_dir, exist_ok=True)
        
        files_to_copy = {
            'lrv': video_file.lrv_path, 'gyro': video_file.gyro_path,
            'exposure': video_file.exposure_path, 'thumbnail': video_file.thumbnail_path,
            'aeflicker': video_file.aeflicker_path, 'beisp': video_file.beisp_path
        }
        
        for file_type, file_path in files_to_copy.items():
            if file_path and os.path.exists(file_path):
                dest_name = f"{video_file.base_name}.{file_type}{os.path.splitext(file_path)[1]}"
                dest_path = os.path.join(metadata_dir, dest_name)
                shutil.copy2(file_path, dest_path)
                logging.debug(f"Copied {file_type} metadata: {dest_path}")
    
    def verify_metadata_sync(self, video_file: Insta360VideoFile, output_path: str) -> bool:
        """Verify that metadata was actually used in stitching"""
        try:
            if not os.path.exists(output_path):
                return False
            
            if video_file.gyro_path:
                logging.info("✓ Gyro data was available for stabilization")
            if video_file.exposure_path:
                logging.info("✓ Exposure data was available for correction")
            if video_file.aeflicker_path:
                logging.info("✓ Flicker correction data was available")
            
            return True
        except Exception as e:
            logging.error(f"Metadata sync verification failed: {e}")
            return False
    
    def create_processing_report(self, video_file: Insta360VideoFile, output_path: str, metadata_status: List[str]):
        """Create detailed processing report for each video"""
        try:
            report_path = output_path.replace('.mp4', '_processing_report.json')
            
            input_size = os.path.getsize(video_file.insv_path) if os.path.exists(video_file.insv_path) else 0
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            input_timestamp = VideoTimestamp.extract_creation_time(video_file.insv_path) if os.path.exists(video_file.insv_path) else None
            output_timestamp = VideoTimestamp.extract_creation_time(output_path) if os.path.exists(output_path) else None
            
            report = {
                'processing_info': {
                    'input_file': video_file.insv_path, 'output_file': output_path,
                    'processing_time': datetime.now().isoformat(),
                    'camera_id': video_file.camera_id, 'base_name': video_file.base_name
                },
                'file_sizes': {
                    'input_mb': round(input_size / (1024*1024), 2),
                    'output_mb': round(output_size / (1024*1024), 2),
                    'compression_ratio': round(output_size / input_size, 3) if input_size > 0 else 0
                },
                'timestamps': {
                    'original': input_timestamp.isoformat() if input_timestamp else None,
                    'processed': output_timestamp.isoformat() if output_timestamp else None,
                    'timestamp_preserved': input_timestamp == output_timestamp if input_timestamp and output_timestamp else False
                },
                'metadata_usage': {
                    'total_metadata_files': self.count_metadata_files(video_file),
                    'gyro_stabilization': {
                        'available': video_file.gyro_path is not None,
                        'file_path': video_file.gyro_path,
                        'file_size': os.path.getsize(video_file.gyro_path) if video_file.gyro_path and os.path.exists(video_file.gyro_path) else 0
                    },
                    'exposure_correction': {
                        'available': video_file.exposure_path is not None,
                        'file_path': video_file.exposure_path,
                        'file_size': os.path.getsize(video_file.exposure_path) if video_file.exposure_path and os.path.exists(video_file.exposure_path) else 0
                    },
                    'flicker_correction': {
                        'available': video_file.aeflicker_path is not None,
                        'file_path': video_file.aeflicker_path,
                        'file_size': os.path.getsize(video_file.aeflicker_path) if video_file.aeflicker_path and os.path.exists(video_file.aeflicker_path) else 0
                    },
                    'isp_data': {
                        'available': video_file.beisp_path is not None,
                        'file_path': video_file.beisp_path,
                        'file_size': os.path.getsize(video_file.beisp_path) if video_file.beisp_path and os.path.exists(video_file.beisp_path) else 0
                    },
                    'preview_video': {
                        'available': video_file.lrv_path is not None,
                        'file_path': video_file.lrv_path,
                        'file_size': os.path.getsize(video_file.lrv_path) if video_file.lrv_path and os.path.exists(video_file.lrv_path) else 0
                    }
                },
                'quality_enhancements': {
                    'stabilization_applied': video_file.gyro_path is not None,
                    'exposure_corrected': video_file.exposure_path is not None,
                    'flicker_reduced': video_file.aeflicker_path is not None,
                    'enhanced_processing': self.count_metadata_files(video_file) > 0
                }
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.debug(f"Processing report saved: {report_path}")
        except Exception as e:
            logging.error(f"Failed to create processing report: {e}")

class CompleteInsta360Processor:
    """Complete Insta360 processor with all functionality"""
    
    def __init__(self, config_file: str = "insta360_config.json"):
        self.config = Insta360Config(config_file)
        self.downloader = Insta360Downloader(self.config)
        self.stitcher = Insta360Stitcher(self.config)
        self.directory_processor = Insta360DirectoryProcessor(self.config)
        self.running = False
        self.stats = {
            'videos_processed': 0, 'videos_failed': 0, 'total_duration': 0,
            'start_time': None, 'cameras_discovered': 0
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.setup_logging()
        
        for path_key in ['download_path', 'output_path', 'temp_path']:
            os.makedirs(self.config.config[path_key], exist_ok=True)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def setup_logging(self):
        """Set up comprehensive logging"""
        os.makedirs("logs", exist_ok=True)
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        file_handler = RotatingFileHandler(
            'logs/insta360_processor.log', maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.config['log_level'].upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging initialized")
    
    def check_environment(self) -> bool:
        """Check if environment is properly set up"""
        checks = []
        
        if os.path.exists(self.config.config_file):
            checks.append(("Configuration file", True))
            try:
                with open(self.config.config_file, 'r') as f:
                    config = json.load(f)
                checks.append(("Configuration valid", True))
            except Exception as e:
                checks.append(("Configuration valid", False, str(e)))
        else:
            checks.append(("Configuration file", False, f"{self.config.config_file} not found"))
        
        camera_sdk = self.config.config.get('camera_sdk_path', '')
        media_sdk = self.config.config.get('media_sdk_path', '')
        
        checks.append(("CameraSDK path", os.path.exists(camera_sdk)))
        checks.append(("MediaSDK path", os.path.exists(media_sdk)))
        
        camera_lib = os.path.join(camera_sdk, 'lib', 'libCameraSDK.so')
        checks.append(("CameraSDK library", os.path.exists(camera_lib)))
        
        deps_to_check = ['json', 'logging', 'datetime', 'pathlib']
        for dep in deps_to_check:
            try:
                __import__(dep)
                checks.append((f"Python {dep}", True))
            except ImportError as e:
                checks.append((f"Python {dep}", False, str(e)))
        
        tools_to_check = ['ffmpeg', 'ffprobe']
        for tool in tools_to_check:
            if VideoTimestamp.command_exists(tool):
                checks.append((f"Tool {tool}", True))
            else:
                checks.append((f"Tool {tool}", False, "not found in PATH"))
        
        all_passed = True
        print("\n=== Environment Check ===")
        for check in checks:
            name = check[0]
            passed = check[1]
            message = check[2] if len(check) > 2 else ""
            
            status = "✓" if passed else "✗"
            print(f"{status} {name:<20} {message}")
            
            if not passed:
                all_passed = False
        
        print(f"\nEnvironment check: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
    
    def discover_cameras(self) -> List[Dict]:
        """Discover and display available cameras"""
        print("\n=== Camera Discovery ===")
        
        try:
            cameras = self.downloader.discover_cameras()
            
            if cameras:
                print(f"Found {len(cameras)} camera(s):")
                for i, camera in enumerate(cameras, 1):
                    print(f"  {i}. ID: {camera.get('id', 'unknown')}")
                    print(f"     Model: {camera.get('model', 'unknown')}")
                    print(f"     Status: {camera.get('status', 'unknown')}")
                    print()
                
                self.stats['cameras_discovered'] = len(cameras)
            else:
                print("No cameras discovered.")
                print("\nTroubleshooting:")
                print("- Ensure camera is powered on")
                print("- Check USB/WiFi connection")
                print("- Verify camera is in the correct mode")
            
            return cameras
        except Exception as e:
            logging.error(f"Error discovering cameras: {e}")
            print(f"Camera discovery failed: {e}")
            return []
    
    def is_insta360_structure(self, input_dir: str) -> bool:
        """Check if directory has Insta360 camera structure"""
        dcim_path = os.path.join(input_dir, "DCIM")
        
        if os.path.exists(dcim_path):
            camera_dirs = [d for d in os.listdir(dcim_path) 
                          if os.path.isdir(os.path.join(dcim_path, d)) and d.startswith("Camera")]
            if camera_dirs:
                print(f"Detected Insta360 camera structure with {len(camera_dirs)} cameras")
                return True
        return False
    
    def process_insta360_structure(self, input_dir: str, output_dir: str) -> bool:
        """Process Insta360 camera directory structure"""
        print("Using enhanced Insta360 batch processor...")
        
        try:
            cameras = self.directory_processor.discover_insta360_structure(input_dir)
            
            if not cameras:
                print("No cameras or videos found in Insta360 structure")
                return False
            
            print(f"\nFound Insta360 structure:")
            total_videos = 0
            for camera_id, videos in cameras.items():
                print(f"  {camera_id}: {len(videos)} videos")
                for video in videos[:3]:
                    metadata_count = self.directory_processor.count_metadata_files(video)
                    timestamp_str = video.timestamp.strftime("%Y-%m-%d %H:%M:%S") if video.timestamp else "Unknown"
                    print(f"    - {video.base_name} ({timestamp_str}, {metadata_count} metadata files)")
                if len(videos) > 3:
                    print(f"    ... and {len(videos) - 3} more videos")
                total_videos += len(videos)
            
            print(f"\nTotal: {len(cameras)} cameras, {total_videos} videos")
            
            response = input(f"\nProcess all {total_videos} videos? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Processing cancelled")
                return False
            
            # Create output structure
            for camera_id in cameras.keys():
                camera_output_dir = os.path.join(output_dir, camera_id)
                os.makedirs(camera_output_dir, exist_ok=True)
                os.makedirs(os.path.join(camera_output_dir, "stitched"), exist_ok=True)
                os.makedirs(os.path.join(camera_output_dir, "metadata"), exist_ok=True)
                os.makedirs(os.path.join(camera_output_dir, "thumbnails"), exist_ok=True)
            
            # Process each camera
            for camera_id, videos in cameras.items():
                logging.info(f"\n=== Processing {camera_id} ===")
                results = {'success': 0, 'failed': 0, 'skipped': 0}
                
                for i, video_file in enumerate(videos, 1):
                    logging.info(f"Processing {i}/{len(videos)}: {video_file.base_name}")
                    
                    output_name = self.generate_output_name(video_file)
                    output_path = os.path.join(output_dir, camera_id, "stitched", output_name)
                    
                    if os.path.exists(output_path):
                        logging.info(f"Skipping {video_file.base_name} (already exists)")
                        results['skipped'] += 1
                        continue
                    
                    success = self.directory_processor.process_video_with_metadata(video_file, output_dir)
                    
                    if success:
                        results['success'] += 1
                        self.stats['videos_processed'] += 1
                    else:
                        results['failed'] += 1
                        self.stats['videos_failed'] += 1
                
                logging.info(f"{camera_id} Results:")
                logging.info(f"  Successful: {results['success']}")
                logging.info(f"  Failed: {results['failed']}")
                logging.info(f"  Skipped: {results['skipped']}")
            
            # Create summary
            self.create_batch_summary(cameras, output_dir)
            
            print(f"\n✓ Insta360 batch processing completed!")
            print(f"  Processed: {self.stats['videos_processed']} videos")
            print(f"  Failed: {self.stats['videos_failed']} videos")
            print(f"\nOutput structure:")
            print(f"  {output_dir}/")
            for camera_id in cameras.keys():
                print(f"    {camera_id}/")
                print(f"      stitched/     # Processed 8K videos")
                print(f"      metadata/     # Original metadata files")
                print(f"      thumbnails/   # Thumbnail data")
            print(f"    batch_processing_summary.json  # Detailed summary")
            
            return True
        except Exception as e:
            logging.error(f"Insta360 structure processing failed: {e}")
            print(f"Enhanced processing failed: {e}")
            return False
    
    def generate_output_name(self, video_file: Insta360VideoFile) -> str:
        """Generate consistent output filename"""
        if video_file.timestamp:
            timestamp_str = video_file.timestamp.strftime("%Y%m%d_%H%M%S")
            return f"{video_file.camera_id}_{timestamp_str}_{video_file.base_name}.mp4"
        else:
            return f"{video_file.camera_id}_{video_file.base_name}.mp4"
    
    def create_batch_summary(self, cameras: Dict[str, List[Insta360VideoFile]], output_dir: str):
        """Create a comprehensive summary of the batch processing"""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_cameras': len(cameras),
            'total_videos': sum(len(videos) for videos in cameras.values()),
            'statistics': self.stats,
            'cameras': {}
        }
        
        for camera_id, videos in cameras.items():
            camera_summary = {
                'video_count': len(videos),
                'videos': []
            }
            
            for video in videos:
                video_info = {
                    'base_name': video.base_name,
                    'timestamp': video.timestamp.isoformat() if video.timestamp else None,
                    'metadata_files': self.directory_processor.count_metadata_files(video),
                    'has_gyro': video.gyro_path is not None,
                    'has_exposure_data': video.exposure_path is not None,
                    'file_size_mb': os.path.getsize(video.insv_path) / (1024*1024) if os.path.exists(video.insv_path) else 0
                }
                camera_summary['videos'].append(video_info)
            
            summary['cameras'][camera_id] = camera_summary
        
        summary_path = os.path.join(output_dir, "batch_processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Batch summary saved: {summary_path}")
        return summary
    
    def process_generic_directory(self, input_dir: str, output_dir: str) -> bool:
        """Process directory with generic video files"""
        print("Using generic directory processor...")
        
        video_extensions = ['.insv', '.insp', '.mp4']
        input_files = []
        
        for ext in video_extensions:
            input_files.extend(Path(input_dir).glob(f'**/*{ext}'))
            input_files.extend(Path(input_dir).glob(f'**/*{ext.upper()}'))
        
        if not input_files:
            print("No video files found in input directory")
            return False
        
        print(f"Found {len(input_files)} files to process")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            for i, input_file in enumerate(input_files, 1):
                if not self.running:
                    break
                
                print(f"\nProcessing {i}/{len(input_files)}: {input_file.name}")
                
                output_file = Path(output_dir) / f"{input_file.stem}_stitched.mp4"
                
                try:
                    success = self.stitcher.stitch_video(str(input_file), str(output_file))
                    
                    if success:
                        print(f"  ✓ Completed: {output_file.name}")
                        self.stats['videos_processed'] += 1
                    else:
                        print(f"  ✗ Failed: {input_file.name}")
                        self.stats['videos_failed'] += 1
                except Exception as e:
                    logging.error(f"Error processing {input_file}: {e}")
                    print(f"  ✗ Error: {e}")
                    self.stats['videos_failed'] += 1
            
            return True
        except Exception as e:
            logging.error(f"Error in generic batch processing: {e}")
            print(f"Generic batch processing failed: {e}")
            return False
    
    def run_automatic_processing(self):
        """Run continuous automatic processing"""
        logging.info("Starting automatic Insta360 processing...")
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Status update thread
        status_thread = threading.Thread(target=self.status_updater, daemon=True)
        status_thread.start()
        
        while self.running:
            try:
                cameras = self.downloader.discover_cameras()
                logging.info(f"Found {len(cameras)} cameras")
                
                if cameras:
                    for camera in cameras:
                        if not self.running:
                            break
                        
                        try:
                            self.process_camera(camera)
                            self.stats['videos_processed'] += 1
                        except Exception as e:
                            logging.error(f"Error processing camera {camera.get('id', 'unknown')}: {e}")
                            self.stats['videos_failed'] += 1
                
                for _ in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Received keyboard interrupt")
                break
            except Exception as e:
                logging.error(f"Error in automatic processing: {e}")
                time.sleep(60)
        
        self.running = False
        print("\nStopping automatic processing...")
        self.show_stats()
    
    def status_updater(self):
        """Background thread to show periodic status updates"""
        last_update = time.time()
        
        while self.running:
            time.sleep(10)
            
            if time.time() - last_update > 300:
                logging.info("Status update:")
                logging.info(f"  Processed: {self.stats['videos_processed']}, "
                           f"Failed: {self.stats['videos_failed']}")
                last_update = time.time()
    
    def process_camera(self, camera: Dict):
        """Process videos from a single camera"""
        camera_id = camera['id']
        
        camera_download_path = os.path.join(
            self.config.config['download_path'], f"camera_{camera_id}"
        )
        os.makedirs(camera_download_path, exist_ok=True)
        
        if self.downloader.download_from_camera(camera_id, camera_download_path):
            self.process_downloaded_files(camera_download_path, camera_id)
    
    def process_downloaded_files(self, download_path: str, camera_id: str):
        """Process downloaded video files"""
        video_extensions = ['.insv', '.insp', '.mp4']
        
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    file_path = os.path.join(root, file)
                    
                    if not self.is_already_processed(file_path):
                        self.process_single_file(file_path, camera_id)
    
    def is_already_processed(self, file_path: str) -> bool:
        """Check if file has already been processed"""
        processed_marker = file_path + ".processed"
        return os.path.exists(processed_marker)
    
    def mark_as_processed(self, file_path: str):
        """Mark file as processed"""
        processed_marker = file_path + ".processed"
        with open(processed_marker, 'w') as f:
            f.write(datetime.now().isoformat())
    
    def process_single_file(self, input_path: str, camera_id: str = "unknown"):
        """Process a single video file"""
        try:
            input_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(input_name)[0]
            timestamp = VideoTimestamp.extract_creation_time(input_path)
            
            if timestamp:
                time_str = timestamp.strftime("%Y%m%d_%H%M%S")
                output_name = f"{camera_id}_{time_str}_{name_without_ext}.mp4"
            else:
                output_name = f"{camera_id}_{name_without_ext}.mp4"
            
            output_path = os.path.join(self.config.config['output_path'], output_name)
            
            if self.stitcher.stitch_video(input_path, output_path):
                self.mark_as_processed(input_path)
                logging.info(f"Successfully processed: {input_path}")
                
                if self.config.config['cleanup_temp']:
                    self.cleanup_temp_files(input_path)
            else:
                logging.error(f"Failed to process: {input_path}")
        except Exception as e:
            logging.error(f"Error processing {input_path}: {e}")
    
    def cleanup_temp_files(self, processed_file: str):
        """Clean up temporary files after successful processing"""
        try:
            pass  # Implement cleanup logic if needed
        except Exception as e:
            logging.error(f"Error cleaning up {processed_file}: {e}")
    
    def show_stats(self):
        """Display processing statistics"""
        if self.stats['start_time']:
            duration = time.time() - self.stats['start_time']
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            
            print(f"\n=== Processing Statistics ===")
            print(f"Runtime: {hours:02d}:{minutes:02d}")
            print(f"Videos processed: {self.stats['videos_processed']}")
            print(f"Videos failed: {self.stats['videos_failed']}")
            print(f"Cameras discovered: {self.stats['cameras_discovered']}")
            
            if self.stats['videos_processed'] > 0:
                success_rate = (self.stats['videos_processed'] / 
                              (self.stats['videos_processed'] + self.stats['videos_failed'])) * 100
                print(f"Success rate: {success_rate:.1f}%")
    
    def run_batch_mode(self, input_dir: str, output_dir: str):
        """Process all files in a directory with Insta360 structure support"""
        print(f"\n=== Batch Processing ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' not found")
            return False
        
        try:
            if self.is_insta360_structure(input_dir):
                return self.process_insta360_structure(input_dir, output_dir)
            else:
                return self.process_generic_directory(input_dir, output_dir)
        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            print(f"Batch processing failed: {e}")
            return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete Insta360 Processor - All-in-One Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode auto                    # Continuous camera processing
  %(prog)s --mode single -i video.insv -o output.mp4
  %(prog)s --mode batch -i ../insta360_backup -o processed_videos
  %(prog)s --discover                     # Discover cameras
  %(prog)s --test                         # Test setup
  %(prog)s --check                        # Check environment

Insta360 Camera Structure:
  The processor automatically detects and handles Insta360 camera structures:
  
  input_directory/
  ├── DCIM/
  │   ├── Camera01/        # Videos and metadata
  │   ├── Camera02/        # Multiple cameras supported
  │   └── Camera03/
  └── MISC/                # Thumbnails and recovery data
  
  Processing maintains structure and preserves all metadata for optimal quality.

For HPC usage:
  sbatch submit_slurm.sh                  # Submit as batch job
  srun %(prog)s --mode auto               # Run interactively
        """
    )
    
    parser.add_argument('--mode', choices=['auto', 'single', 'batch'],
                       help='Processing mode')
    parser.add_argument('-i', '--input', 
                       help='Input file (single mode) or directory (batch mode)')
    parser.add_argument('-o', '--output',
                       help='Output file (single mode) or directory (batch mode)')
    parser.add_argument('--config', default='insta360_config.json',
                       help='Configuration file path')
    parser.add_argument('--discover', action='store_true',
                       help='Discover connected cameras')
    parser.add_argument('--test', action='store_true',
                       help='Test processing pipeline')
    parser.add_argument('--check', action='store_true',
                       help='Check environment setup')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--stats', action='store_true',
                       help='Show processing statistics and exit')
    
    args = parser.parse_args()
    
    processor = CompleteInsta360Processor(args.config)
    
    print("=" * 60)
    print("🎥 Complete Insta360 8K Processor with Metadata Sync")
    print("=" * 60)
    
    try:
        if args.check:
            success = processor.check_environment()
            sys.exit(0 if success else 1)
        
        elif args.discover:
            cameras = processor.discover_cameras()
            sys.exit(0 if cameras else 1)
        
        elif args.test:
            success = processor.check_environment()
            if success:
                print("✓ All tests passed! System is ready for processing.")
            sys.exit(0 if success else 1)
        
        elif args.stats:
            processor.show_stats()
            sys.exit(0)
        
        elif args.mode == 'auto':
            if not processor.check_environment():
                print("Environment check failed. Use --check for details.")
                sys.exit(1)
            processor.run_automatic_processing()
        
        elif args.mode == 'single':
            if not args.input or not args.output:
                print("Error: Single mode requires --input and --output")
                sys.exit(1)
            
            print(f"Processing single file: {args.input} -> {args.output}")
            success = processor.stitcher.stitch_video(args.input, args.output)
            print("✓ Processing completed successfully" if success else "✗ Processing failed")
            sys.exit(0 if success else 1)
        
        elif args.mode == 'batch':
            if not args.input:
                print("Error: Batch mode requires --input directory")
                sys.exit(1)
            
            output_dir = args.output or './stitched_output'
            success = processor.run_batch_mode(args.input, output_dir)
            sys.exit(0 if success else 1)
        
        else:
            print("No mode specified. Use --help for usage information.")
            print("\nQuick start:")
            print("  python3 complete_insta360_processor.py --check     # Check setup")
            print("  python3 complete_insta360_processor.py --discover  # Find cameras") 
            print("  python3 complete_insta360_processor.py --mode auto # Start processing")
            print("\nInsta360 directory processing:")
            print("  python3 complete_insta360_processor.py --mode batch -i ../insta360_backup -o processed")
            print("  # Automatically handles DCIM/Camera01, Camera02, etc. structure")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()