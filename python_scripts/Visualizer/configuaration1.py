#!/usr/bin/env python3
"""
Configuration and Utility Scripts for Video Analysis Processor
============================================================

This file contains configuration management and utility functions
to help transition from the old process*.py files to the new system.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Default configuration template
DEFAULT_CONFIG = {
    "processing": {
        "yolo_model": "yolo11x.pt",
        "confidence_threshold": 0.3,
        "parallel_processing": True,
        "max_workers": 4,
        "batch_size": 10,
        "gpu_acceleration": True
    },
    "video_analysis": {
        "analyze_audio": True,
        "scene_complexity": True,
        "object_tracking": True,
        "stoplight_detection": True,
        "traffic_counting": True
    },
    "360_video": {
        "enable_360_processing": True,
        "equirectangular_detection": True,
        "region_based_analysis": True,
        "bearing_calculation": True
    },
    "output": {
        "csv_compression": False,
        "include_thumbnails": False,
        "separate_files_per_analysis": True,
        "consolidate_reports": True
    },
    "performance": {
        "memory_limit_gb": 8,
        "chunk_size_mb": 100,
        "cache_features": True,
        "cleanup_temp_files": True
    }
}

class ConfigManager:
    """Manage configuration for the video analysis processor"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "video_analysis_config.yaml"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
        
        return DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class LegacyConverter:
    """Convert old process*.py workflow to new system"""
    
    def __init__(self):
        self.old_scripts = [
            'process_groups.py',
            'stoplight.py', 
            'counter.py',
            'unified.py'
        ]
    
    def analyze_old_workflow(self, directory: str) -> Dict[str, Any]:
        """Analyze existing workflow to understand processing patterns"""
        analysis = {
            'found_scripts': [],
            'video_directories': [],
            'csv_outputs': [],
            'processing_patterns': {}
        }
        
        directory = Path(directory)
        
        # Find old scripts
        for script in self.old_scripts:
            script_path = directory / script
            if script_path.exists():
                analysis['found_scripts'].append(str(script_path))
        
        # Find video directories with merged_output.csv
        for item in directory.rglob('*'):
            if item.is_dir():
                merged_csv = item / 'merged_output.csv'
                if merged_csv.exists():
                    # Check for video files
                    video_files = list(item.glob('*.mp4')) + list(item.glob('*.MP4'))
                    if video_files:
                        analysis['video_directories'].append({
                            'path': str(item),
                            'merged_csv': str(merged_csv),
                            'video_files': [str(v) for v in video_files],
                            'video_count': len(video_files)
                        })
        
        # Find existing CSV outputs
        csv_patterns = ['*_tracking.csv', '*_stoplights.csv', '*_counts.csv', 'speed_table_*.csv']
        for pattern in csv_patterns:
            csv_files = list(directory.rglob(pattern))
            analysis['csv_outputs'].extend([str(f) for f in csv_files])
        
        return analysis
    
    def create_migration_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create migration plan from old to new system"""
        plan = {
            'migration_steps': [],
            'data_consolidation': [],
            'performance_improvements': [],
            'new_features': []
        }
        
        # Migration steps
        if analysis['found_scripts']:
            plan['migration_steps'].append({
                'step': 'backup_old_scripts',
                'description': 'Backup existing process scripts',
                'scripts': analysis['found_scripts']
            })
        
        if analysis['video_directories']:
            plan['migration_steps'].append({
                'step': 'convert_directory_structure',
                'description': 'Convert individual directory processing to batch processing',
                'directories': len(analysis['video_directories'])
            })
        
        # Data consolidation opportunities
        if len(analysis['csv_outputs']) > 50:
            plan['data_consolidation'].append({
                'type': 'csv_merger',
                'description': f'Consolidate {len(analysis["csv_outputs"])} CSV files',
                'benefit': 'Reduced file count, improved query performance'
            })
        
        # Performance improvements
        if len(analysis['video_directories']) > 10:
            plan['performance_improvements'].append({
                'improvement': 'parallel_processing',
                'description': 'Process multiple videos simultaneously',
                'estimated_speedup': f'{min(4, len(analysis["video_directories"]))}x'
            })
        
        plan['performance_improvements'].append({
            'improvement': '360_video_optimization',
            'description': 'Specialized 360° video processing',
            'benefit': 'Better accuracy for panoramic videos'
        })
        
        # New features
        plan['new_features'] = [
            'Automatic 360° vs flat video detection',
            'Audio analysis and noise level detection',
            'Scene complexity assessment',
            'Unified CSV output format',
            'Progress tracking and reporting',
            'Memory-efficient processing for large datasets'
        ]
        
        return plan
    
    def convert_merged_csv_to_matcher50_format(self, video_dirs: List[Dict]) -> str:
        """Convert old merged_output.csv files to matcher50 compatible format"""
        import pandas as pd
        import json
        
        matcher50_results = {}
        
        for video_dir in video_dirs:
            for video_file in video_dir['video_files']:
                # Create a synthetic match entry
                matcher50_results[video_file] = {
                    'matches': [{
                        'path': video_dir['merged_csv'],
                        'quality': 'synthetic_conversion',
                        'combined_score': 1.0,
                        'is_360_video': False,  # Will be detected automatically
                        'conversion_note': 'Converted from old workflow'
                    }]
                }
        
        # Save synthetic matcher50 results
        output_path = 'converted_matcher50_results.json'
        with open(output_path, 'w') as f:
            json.dump(matcher50_results, f, indent=2)
        
        return output_path


class BatchProcessor:
    """Utility for batch processing large video datasets"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def estimate_processing_time(self, video_paths: List[str]) -> Dict[str, Any]:
        """Estimate processing time for a batch of videos"""
        import cv2
        
        total_duration = 0
        total_size = 0
        video_count = len(video_paths)
        
        # Sample a few videos to estimate
        sample_size = min(5, len(video_paths))
        sample_videos = video_paths[:sample_size]
        
        sample_duration = 0
        sample_size_mb = 0
        
        for video_path in sample_videos:
            try:
                # Get video info
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Get file size
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                
                sample_duration += duration
                sample_size_mb += size_mb
                
            except Exception as e:
                logger.warning(f"Could not analyze {video_path}: {e}")
        
        if sample_size > 0:
            avg_duration = sample_duration / sample_size
            avg_size_mb = sample_size_mb / sample_size
            
            # Estimate total
            total_duration = avg_duration * video_count
            total_size = avg_size_mb * video_count
            
            # Processing time estimation (rough)
            # Assume ~0.1x real-time for processing (10 min video = 1 min processing)
            processing_time_hours = (total_duration * 0.1) / 3600
            
            # Adjust for parallel processing
            if self.config.get('processing.parallel_processing', False):
                workers = self.config.get('processing.max_workers', 4)
                processing_time_hours /= workers
        else:
            processing_time_hours = 0
            total_duration = 0
            total_size = 0
        
        return {
            'video_count': video_count,
            'total_duration_hours': total_duration / 3600,
            'total_size_gb': total_size / 1024,
            'estimated_processing_hours': processing_time_hours,
            'parallel_speedup': self.config.get('processing.max_workers', 1),
            'recommendation': self._get_processing_recommendation(total_size, processing_time_hours)
        }
    
    def _get_processing_recommendation(self, size_gb: float, time_hours: float) -> str:
        """Get processing recommendation based on dataset size"""
        if size_gb > 1000:  # > 1TB
            return "Large dataset detected. Consider processing in chunks and using high-end GPU."
        elif size_gb > 100:  # > 100GB
            return "Medium dataset. Enable parallel processing and monitor memory usage."
        elif time_hours > 24:  # > 1 day
            return "Long processing time expected. Consider overnight processing or cloud resources."
        else:
            return "Dataset size is manageable for local processing."
    
    def create_processing_chunks(self, video_paths: List[str], chunk_size: int = None) -> List[List[str]]:
        """Split video list into processing chunks"""
        if chunk_size is None:
            chunk_size = self.config.get('processing.batch_size', 10)
        
        chunks = []
        for i in range(0, len(video_paths), chunk_size):
            chunk = video_paths[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks


def create_sample_config():
    """Create a sample configuration file"""
    config_manager = ConfigManager("sample_config.yaml")
    config_manager.save_config()
    print("Sample configuration created: sample_config.yaml")


def analyze_dataset(directory: str):
    """Analyze existing dataset for migration planning"""
    converter = LegacyConverter()
    analysis = converter.analyze_old_workflow(directory)
    plan = converter.create_migration_plan(analysis)
    
    print("\n=== Dataset Analysis ===")
    print(f"Found {len(analysis['found_scripts'])} old processing scripts")
    print(f"Found {len(analysis['video_directories'])} video directories")
    print(f"Found {len(analysis['csv_outputs'])} existing CSV output files")
    
    print("\n=== Migration Plan ===")
    for step in plan['migration_steps']:
        print(f"- {step['step']}: {step['description']}")
    
    print("\n=== Performance Improvements ===")
    for improvement in plan['performance_improvements']:
        print(f"- {improvement['improvement']}: {improvement['description']}")
    
    print("\n=== New Features ===")
    for feature in plan['new_features']:
        print(f"- {feature}")
    
    # Convert if requested
    if analysis['video_directories']:
        convert = input("\nConvert old workflow to new format? (y/n): ")
        if convert.lower() == 'y':
            output_file = converter.convert_merged_csv_to_matcher50_format(analysis['video_directories'])
            print(f"Conversion complete! Use this file with the new processor: {output_file}")


def estimate_processing(video_directory: str):
    """Estimate processing time for a video directory"""
    video_paths = []
    
    directory = Path(video_directory)
    for ext in ['*.mp4', '*.MP4', '*.avi', '*.mov']:
        video_paths.extend(directory.rglob(ext))
    
    video_paths = [str(p) for p in video_paths]
    
    if not video_paths:
        print("No video files found in directory")
        return
    
    config = ConfigManager()
    processor = BatchProcessor(config)
    
    estimate = processor.estimate_processing_time(video_paths)
    
    print("\n=== Processing Estimate ===")
    print(f"Video count: {estimate['video_count']}")
    print(f"Total duration: {estimate['total_duration_hours']:.1f} hours")
    print(f"Total size: {estimate['total_size_gb']:.1f} GB")
    print(f"Estimated processing time: {estimate['estimated_processing_hours']:.1f} hours")
    print(f"Parallel speedup factor: {estimate['parallel_speedup']}x")
    print(f"Recommendation: {estimate['recommendation']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Analysis Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    parser_config = subparsers.add_parser('create-config', help='Create sample configuration')
    
    # Analyze dataset command
    parser_analyze = subparsers.add_parser('analyze', help='Analyze existing dataset')
    parser_analyze.add_argument('directory', help='Directory to analyze')
    
    # Estimate processing command
    parser_estimate = subparsers.add_parser('estimate', help='Estimate processing time')
    parser_estimate.add_argument('directory', help='Video directory to estimate')
    
    args = parser.parse_args()
    
    if args.command == 'create-config':
        create_sample_config()
    elif args.command == 'analyze':
        analyze_dataset(args.directory)
    elif args.command == 'estimate':
        estimate_processing(args.directory)
    else:
        parser.print_help()