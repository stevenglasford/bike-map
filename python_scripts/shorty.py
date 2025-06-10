import cv2
import numpy as np
import gpxpy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import argparse
import os
import glob
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from scipy.signal import correlate
from scipy.stats import pearsonr
import warnings

# Setup
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGPUVideoProcessor:
    """Simplified GPU video processor focused on motion extraction"""
    
    def __init__(self, devices):
        self.devices = devices
        self.primary_device = devices[0]
        torch.cuda.set_device(self.primary_device)
        
        # Initialize optical flow on primary GPU
        self.flow_calc = None
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(self.primary_device.index)
            self.flow_calc = cv2.cuda_FarnebackOpticalFlow.create()
            logger.info("CUDA optical flow initialized")
    
    def extract_motion_features(self, video_path):
        """Extract simple but effective motion features"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract motion data
            motion_magnitudes = []
            motion_directions = []
            accelerations = []
            
            prev_gray = None
            prev_motion = None
            
            # Sample every few frames for efficiency
            frame_step = max(1, int(fps / 10))  # ~10 samples per second
            
            for frame_idx in range(0, frame_count, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize to standard size
                frame = cv2.resize(frame, (320, 180))  # Smaller for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Calculate optical flow
                    if self.flow_calc and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        # GPU optical flow
                        gpu_prev = cv2.cuda_GpuMat()
                        gpu_curr = cv2.cuda_GpuMat()
                        gpu_prev.upload(prev_gray)
                        gpu_curr.upload(gray)
                        flow = self.flow_calc.calc(gpu_prev, gpu_curr, None).download()
                    else:
                        # CPU fallback
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    
                    # Calculate motion magnitude and direction
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    direction = np.arctan2(flow[..., 1], flow[..., 0])
                    
                    # Motion statistics
                    avg_magnitude = np.mean(magnitude)
                    avg_direction = np.mean(np.abs(direction))
                    
                    motion_magnitudes.append(avg_magnitude)
                    motion_directions.append(avg_direction)
                    
                    # Calculate acceleration
                    if prev_motion is not None:
                        acceleration = avg_magnitude - prev_motion
                        accelerations.append(acceleration)
                    else:
                        accelerations.append(0)
                    
                    prev_motion = avg_magnitude
                else:
                    motion_magnitudes.append(0)
                    motion_directions.append(0)
                    accelerations.append(0)
                
                prev_gray = gray
            
            cap.release()
            
            if len(motion_magnitudes) < 10:  # Need minimum data
                return None
            
            # Convert to tensors and move to GPU
            features = {
                'motion_magnitude': torch.tensor(motion_magnitudes, device=self.primary_device, dtype=torch.float32),
                'motion_direction': torch.tensor(motion_directions, device=self.primary_device, dtype=torch.float32),
                'acceleration': torch.tensor(accelerations, device=self.primary_device, dtype=torch.float32),
                'duration': frame_count / fps if fps > 0 else 0,
                'fps': fps,
                'sample_count': len(motion_magnitudes)
            }
            
            logger.info(f"Processed {video_path}: {len(motion_magnitudes)} motion samples")
            return features
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None
    
    def process_videos(self, video_paths):
        """Process all videos"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = {executor.submit(self.extract_motion_features, path): path for path in video_paths}
            
            for future in tqdm(as_completed(futures), total=len(video_paths), desc="Processing videos"):
                path = futures[future]
                try:
                    result = future.result()
                    results[path] = result
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results[path] = None
        
        return results

class SimpleGPXProcessor:
    """Simplified GPX processor focused on speed/acceleration"""
    
    def __init__(self, device):
        self.device = device
    
    def parse_gpx_file(self, gpx_path):
        """Parse single GPX file and extract speed/acceleration"""
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for pt in segment.points:
                        if pt.time:
                            points.append({
                                'timestamp': pt.time.replace(tzinfo=None),
                                'lat': pt.latitude,
                                'lon': pt.longitude,
                                'elevation': pt.elevation or 0
                            })
            
            if len(points) < 10:
                return None
            
            df = pd.DataFrame(points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate speeds and accelerations
            speeds = []
            accelerations = []
            
            for i in range(len(df)):
                if i == 0:
                    speeds.append(0)
                    accelerations.append(0)
                    continue
                
                # Calculate distance using Haversine formula
                lat1, lon1 = np.radians(df.iloc[i-1][['lat', 'lon']])
                lat2, lon2 = np.radians(df.iloc[i][['lat', 'lon']])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                distance = 2 * 3958.8 * np.arcsin(np.sqrt(a))  # miles
                
                # Calculate time difference
                time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
                if time_diff <= 0:
                    time_diff = 1.0
                
                # Speed in mph
                speed = (distance * 3600) / time_diff
                speeds.append(speed)
                
                # Acceleration
                if i == 1:
                    accelerations.append(0)
                else:
                    prev_speed = speeds[-2]
                    acceleration = (speed - prev_speed) / time_diff
                    accelerations.append(acceleration)
            
            # Smooth the data
            speeds = np.array(speeds)
            accelerations = np.array(accelerations)
            
            # Simple moving average smoothing
            window = min(5, len(speeds) // 4)
            if window > 1:
                speeds = np.convolve(speeds, np.ones(window)/window, mode='same')
                accelerations = np.convolve(accelerations, np.ones(window)/window, mode='same')
            
            # Sample at regular intervals to match video sampling
            target_samples = min(len(speeds), 300)  # Max 300 samples
            if len(speeds) > target_samples:
                indices = np.linspace(0, len(speeds)-1, target_samples, dtype=int)
                speeds = speeds[indices]
                accelerations = accelerations[indices]
            
            # Convert to GPU tensors
            features = {
                'speed': torch.tensor(speeds, device=self.device, dtype=torch.float32),
                'acceleration': torch.tensor(accelerations, device=self.device, dtype=torch.float32),
                'duration': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds(),
                'point_count': len(df),
                'sample_count': len(speeds)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error parsing GPX {gpx_path}: {e}")
            return None
    
    def process_gpx_files(self, gpx_paths):
        """Process all GPX files"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = {executor.submit(self.parse_gpx_file, path): path for path in gpx_paths}
            
            for future in tqdm(as_completed(futures), total=len(gpx_paths), desc="Processing GPX files"):
                path = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[path] = result
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
        
        return results

class SimpleGPUCorrelator:
    """Simple but effective GPU correlation"""
    
    def __init__(self, device):
        self.device = device
        torch.cuda.set_device(device)
    
    def correlate_signals(self, signal1, signal2):
        """Correlate two signals on GPU"""
        try:
            # Ensure signals are on GPU
            if signal1.device != self.device:
                signal1 = signal1.to(self.device)
            if signal2.device != self.device:
                signal2 = signal2.to(self.device)
            
            # Normalize signals
            signal1 = (signal1 - torch.mean(signal1)) / (torch.std(signal1) + 1e-8)
            signal2 = (signal2 - torch.mean(signal2)) / (torch.std(signal2) + 1e-8)
            
            # Make signals same length
            min_len = min(len(signal1), len(signal2))
            if min_len < 10:
                return 0.0, 0
            
            signal1 = signal1[:min_len]
            signal2 = signal2[:min_len]
            
            # Cross-correlation using convolution
            correlation = F.conv1d(
                signal1.flip(0).unsqueeze(0).unsqueeze(0),
                signal2.unsqueeze(0).unsqueeze(0),
                padding=len(signal2)-1
            ).squeeze()
            
            # Find best correlation and offset
            max_corr, max_idx = torch.max(torch.abs(correlation), dim=0)
            offset = max_idx.item() - (len(signal2) - 1)
            
            # Normalize correlation coefficient
            corr_coeff = max_corr.item() / min_len
            
            return corr_coeff, offset
            
        except Exception as e:
            logger.debug(f"Correlation failed: {e}")
            return 0.0, 0
    
    def find_best_matches(self, video_features, gpx_database, top_k=5):
        """Find best GPX matches for video"""
        if video_features is None:
            return []
        
        scores = []
        
        # Get video motion data
        video_motion = video_features['motion_magnitude']
        video_accel = video_features['acceleration']
        
        for gpx_path, gpx_features in gpx_database.items():
            if gpx_features is None:
                continue
            
            try:
                # Get GPX speed/acceleration data
                gpx_speed = gpx_features['speed']
                gpx_accel = gpx_features['acceleration']
                
                # Correlate motion magnitude with speed
                motion_speed_corr, motion_offset = self.correlate_signals(video_motion, gpx_speed)
                
                # Correlate accelerations
                accel_corr, accel_offset = self.correlate_signals(video_accel, gpx_accel)
                
                # Duration ratio check
                video_duration = video_features['duration']
                gpx_duration = gpx_features['duration']
                duration_ratio = min(video_duration, gpx_duration) / max(video_duration, gpx_duration)
                
                # Combined score
                combined_score = (0.6 * motion_speed_corr + 0.4 * accel_corr) * duration_ratio
                
                scores.append({
                    'path': gpx_path,
                    'score': combined_score,
                    'motion_speed_corr': motion_speed_corr,
                    'accel_corr': accel_corr,
                    'duration_ratio': duration_ratio,
                    'motion_offset': motion_offset,
                    'accel_offset': accel_offset
                })
                
            except Exception as e:
                logger.debug(f"Failed to correlate with {gpx_path}: {e}")
                continue
        
        # Sort by score and return top K
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def correlate_all(self, video_features_dict, gpx_database, top_k=5):
        """Correlate all videos with all GPX files"""
        results = {}
        
        for video_path, video_features in tqdm(video_features_dict.items(), desc="Correlating"):
            if video_features is None:
                results[video_path] = None
                continue
            
            matches = self.find_best_matches(video_features, gpx_database, top_k)
            
            if matches:
                # Format matches for compatibility
                formatted_matches = []
                for match in matches:
                    formatted_matches.append({
                        'path': match['path'],
                        'combined_score': match['score'],
                        'sub_scores': {
                            'motion_speed_correlation': match['motion_speed_corr'],
                            'acceleration_correlation': match['accel_corr'],
                            'duration_ratio': match['duration_ratio']
                        },
                        'duration_ratio': match['duration_ratio'],
                        'time_offset': match['motion_offset'],
                        'point_count': gpx_database[match['path']]['point_count'] if match['path'] in gpx_database else 0
                    })
                
                results[video_path] = {
                    'matches': formatted_matches,
                    'video_duration': video_features['duration']
                }
            else:
                results[video_path] = None
        
        return results

class SimpleAccuracyReporter:
    """Generate accuracy reports"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, results):
        """Generate detailed accuracy report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(results),
            'successful_correlations': 0,
            'failed_correlations': 0,
            'score_distribution': [],
            'confidence_levels': {
                'excellent': 0,    # score > 0.5
                'high': 0,         # 0.3 < score <= 0.5  
                'medium': 0,       # 0.2 < score <= 0.3
                'low': 0,          # 0.1 < score <= 0.2
                'very_low': 0      # score <= 0.1
            },
            'detailed_results': []
        }
        
        for video_path, result in results.items():
            if result is None or not result.get('matches'):
                report['failed_correlations'] += 1
                continue
            
            report['successful_correlations'] += 1
            best_match = result['matches'][0]
            score = best_match['combined_score']
            report['score_distribution'].append(score)
            
            # Classify confidence levels
            if score > 0.5:
                report['confidence_levels']['excellent'] += 1
            elif score > 0.3:
                report['confidence_levels']['high'] += 1
            elif score > 0.2:
                report['confidence_levels']['medium'] += 1
            elif score > 0.1:
                report['confidence_levels']['low'] += 1
            else:
                report['confidence_levels']['very_low'] += 1
            
            report['detailed_results'].append({
                'video': str(video_path),
                'best_match': str(best_match['path']),
                'score': score,
                'sub_scores': best_match.get('sub_scores', {}),
                'duration_ratio': best_match['duration_ratio'],
                'time_offset': best_match.get('time_offset', 0),
                'all_matches': [
                    {'gpx': str(m['path']), 'score': m['combined_score']}
                    for m in result['matches']
                ]
            })
        
        # Calculate statistics
        if report['score_distribution']:
            scores = np.array(report['score_distribution'])
            report['statistics'] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'median_score': float(np.median(scores))
            }
        
        # Save reports
        self._save_reports(report)
        return report
    
    def _save_reports(self, report):
        """Save all report formats"""
        # JSON report
        json_path = self.output_dir / 'simple_accuracy_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Text summary
        summary_path = self.output_dir / 'simple_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("SIMPLE GPU VIDEO-GPX CORRELATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Total Videos: {report['total_videos']}\n")
            f.write(f"Successful: {report['successful_correlations']}\n")
            f.write(f"Failed: {report['failed_correlations']}\n")
            
            if report['successful_correlations'] > 0:
                success_rate = (report['successful_correlations'] / report['total_videos']) * 100
                f.write(f"Success Rate: {success_rate:.1f}%\n\n")
            
            if 'statistics' in report:
                f.write("Score Statistics:\n")
                f.write(f"  Mean: {report['statistics']['mean_score']:.4f}\n")
                f.write(f"  Median: {report['statistics']['median_score']:.4f}\n")
                f.write(f"  Range: {report['statistics']['min_score']:.4f} - {report['statistics']['max_score']:.4f}\n\n")
            
            f.write("Confidence Distribution:\n")
            for level, count in report['confidence_levels'].items():
                percentage = (count / report['total_videos'] * 100) if report['total_videos'] > 0 else 0
                f.write(f"  {level.title()}: {count} ({percentage:.1f}%)\n")
        
        logger.info(f"Reports saved to {json_path} and {summary_path}")

def main():
    """Main execution with simple but effective approach"""
    parser = argparse.ArgumentParser(description="Simple GPU-Accelerated Video-GPX Correlation")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing videos and GPX files")
    parser.add_argument("-o", "--output", default="./simple_gpu_results", help="Output directory")
    parser.add_argument("-c", "--cache", default="./simple_gpu_cache", help="Cache directory")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0], help="GPU IDs to use")
    parser.add_argument("--top_k", type=int, default=5, help="Top K matches per video")
    parser.add_argument("--force", action='store_true', help="Force reprocessing")
    
    args = parser.parse_args()
    
    # Validate GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in args.gpu_ids if gpu_id < torch.cuda.device_count()]
    if not devices:
        devices = [torch.device('cuda:0')]
    
    logger.info(f"Using GPUs: {[d.index for d in devices]}")
    
    # Setup directories
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)
    
    # Find files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']
    video_files = []
    for pattern in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.directory, pattern)))
    
    gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
    gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
    
    logger.info(f"Found {len(video_files)} videos and {len(gpx_files)} GPX files")
    
    if not video_files or not gpx_files:
        raise ValueError("No videos or GPX files found!")
    
    # Initialize processors
    video_processor = SimpleGPUVideoProcessor(devices)
    gpx_processor = SimpleGPXProcessor(devices[0])
    correlator = SimpleGPUCorrelator(devices[0])
    
    # Process videos
    logger.info("Processing videos...")
    video_cache_path = cache_dir / "simple_video_features.pkl"
    
    if video_cache_path.exists() and not args.force:
        with open(video_cache_path, 'rb') as f:
            video_features = pickle.load(f)
        logger.info(f"Loaded cached video features for {len(video_features)} videos")
    else:
        video_features = video_processor.process_videos(video_files)
        with open(video_cache_path, 'wb') as f:
            pickle.dump(video_features, f)
        logger.info(f"Processed and cached {len(video_features)} videos")
    
    # Process GPX files
    logger.info("Processing GPX files...")
    gpx_cache_path = cache_dir / "simple_gpx_features.pkl"
    
    if gpx_cache_path.exists() and not args.force:
        with open(gpx_cache_path, 'rb') as f:
            gpx_database = pickle.load(f)
        logger.info(f"Loaded cached GPX features for {len(gpx_database)} files")
    else:
        gpx_database = gpx_processor.process_gpx_files(gpx_files)
        with open(gpx_cache_path, 'wb') as f:
            pickle.dump(gpx_database, f)
        logger.info(f"Processed and cached {len(gpx_database)} GPX files")
    
    # Perform correlation
    logger.info("Performing correlation analysis...")
    results = correlator.correlate_all(video_features, gpx_database, top_k=args.top_k)
    
    # Generate report
    reporter = SimpleAccuracyReporter(output_dir)
    report = reporter.generate_report(results)
    
    # Save results
    results_path = output_dir / "simple_correlations.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    success_rate = (report['successful_correlations'] / report['total_videos'] * 100) if report['total_videos'] > 0 else 0
    logger.info(f"Correlation complete! Success rate: {success_rate:.1f}%")
    
    if 'statistics' in report:
        logger.info(f"Mean correlation score: {report['statistics']['mean_score']:.4f}")
    
    with open(output_dir / 'simple_summary.txt', 'r') as f:
        print("\n" + f.read())

if __name__ == "__main__":
    main()
