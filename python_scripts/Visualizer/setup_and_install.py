#!/usr/bin/env python3
"""
Setup and Installation Script for Video Analysis Processor
=========================================================

This script handles the setup and installation of dependencies
for the comprehensive video analysis processor.
"""

import subprocess
import sys
import os
from pathlib import Path

# Requirements for the video analysis processor
REQUIREMENTS = """
# Core dependencies
opencv-python>=4.8.0
numpy>=1.21.0
pandas>=1.3.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# GPS and geospatial
gpxpy>=1.5.0
geopy>=2.3.0

# Data processing
scipy>=1.9.0
scikit-learn>=1.0.0
PyYAML>=6.0

# Progress and logging
tqdm>=4.64.0

# Optional: Advanced features
# deep-sort-realtime>=1.3.0  # For advanced tracking
# plotly>=5.0.0  # For visualization
# folium>=0.14.0  # For GPS visualization
"""

def create_requirements_file():
    """Create requirements.txt file"""
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS.strip())
    print("Created requirements.txt")

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Python dependencies: {e}")
        return False
    return True

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        print("✓ FFmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg not found")
        print("Please install FFmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("⚠ No GPU available - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed - cannot check GPU")
        return False

def download_yolo_model():
    """Download YOLO model if not present"""
    model_file = 'yolo11x.pt'
    if os.path.exists(model_file):
        print(f"✓ YOLO model {model_file} already exists")
        return True
    
    print(f"Downloading YOLO model {model_file}...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_file)  # This will download if not present
        print(f"✓ YOLO model {model_file} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download YOLO model: {e}")
        return False

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        'output',
        'output/object_tracking',
        'output/stoplight_detection', 
        'output/traffic_counting',
        'output/audio_analysis',
        'output/scene_complexity',
        'output/processing_reports',
        'cache',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")

def create_sample_scripts():
    """Create sample usage scripts"""
    
    # Sample processing script
    sample_script = '''#!/usr/bin/env python3
"""
Sample usage script for the video analysis processor
"""
import sys
from video_analysis_processor import VideoAnalysisProcessor
from config_utils import ConfigManager

def main():
    # Load configuration
    config = ConfigManager().config
    
    # Initialize processor
    processor = VideoAnalysisProcessor(config)
    
    # Example: Process matcher50 results
    results_file = "path/to/matcher50_results.json"
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    # Load and process
    results_data = processor.load_matcher50_results(results_file)
    processor.process_matcher50_results(results_data)

if __name__ == "__main__":
    main()
'''
    
    with open('sample_usage.py', 'w') as f:
        f.write(sample_script)
    
    # Sample configuration
    sample_config = '''# Video Analysis Processor Configuration
# =====================================

processing:
  yolo_model: "yolo11x.pt"
  confidence_threshold: 0.3
  parallel_processing: true
  max_workers: 4
  batch_size: 10
  gpu_acceleration: true

video_analysis:
  analyze_audio: true
  scene_complexity: true
  object_tracking: true
  stoplight_detection: true
  traffic_counting: true

360_video:
  enable_360_processing: true
  equirectangular_detection: true
  region_based_analysis: true
  bearing_calculation: true

output:
  csv_compression: false
  include_thumbnails: false
  separate_files_per_analysis: true
  consolidate_reports: true

performance:
  memory_limit_gb: 8
  chunk_size_mb: 100
  cache_features: true
  cleanup_temp_files: true
'''
    
    with open('config.yaml', 'w') as f:
        f.write(sample_config)
    
    print("✓ Sample scripts created")

def run_tests():
    """Run basic tests to verify installation"""
    print("\nRunning basic tests...")
    
    # Test imports
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import torch
        import ultralytics
        import librosa
        import gpxpy
        import geopy
        print("✓ All core imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test YOLO model loading
    try:
        from ultralytics import YOLO
        model = YOLO('yolo11x.pt')
        print("✓ YOLO model loads successfully")
    except Exception as e:
        print(f"✗ YOLO model loading failed: {e}")
        return False
    
    print("✓ All tests passed!")
    return True

def main():
    """Main setup function"""
    print("Video Analysis Processor Setup")
    print("=" * 50)
    
    # Create requirements file
    create_requirements_file()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed - could not install requirements")
        return
    
    # Check system dependencies
    ffmpeg_ok = check_ffmpeg()
    gpu_ok = check_gpu()
    
    # Download YOLO model
    yolo_ok = download_yolo_model()
    
    # Create directories
    create_directory_structure()
    
    # Create sample files
    create_sample_scripts()
    
    # Run tests
    tests_ok = run_tests()
    
    print("\nSetup Summary:")
    print("=" * 50)
    print(f"Python dependencies: ✓")
    print(f"FFmpeg: {'✓' if ffmpeg_ok else '✗'}")
    print(f"GPU support: {'✓' if gpu_ok else '⚠'}")
    print(f"YOLO model: {'✓' if yolo_ok else '✗'}")
    print(f"Tests: {'✓' if tests_ok else '✗'}")
    
    if ffmpeg_ok and yolo_ok and tests_ok:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run your matcher50.py to generate video-GPS matches")
        print("2. Use video_analysis_processor.py to process the results:")
        print("   python video_analysis_processor.py -i matcher50_results.json -o output/")
        print("3. Check the output/ directory for CSV results")
        print("\nFor help:")
        print("  python video_analysis_processor.py --help")
        print("  python config_utils.py analyze /path/to/your/videos")
    else:
        print("\n⚠ Setup completed with warnings - please address the issues above")

if __name__ == "__main__":
    main()

# Additional utility functions

def upgrade_packages():
    """Upgrade all packages to latest versions"""
    print("Upgrading packages...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '-r', 'requirements.txt'])

def clean_cache():
    """Clean cache and temporary files"""
    import shutil
    
    cache_dirs = ['cache', '__pycache__', '.pytest_cache']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleaned {cache_dir}")
    
    print("Cache cleaned")

def verify_installation():
    """Verify that everything is working correctly"""
    print("Verifying installation...")
    
    # Check all imports
    required_modules = [
        'cv2', 'numpy', 'pandas', 'torch', 'ultralytics',
        'librosa', 'gpxpy', 'geopy', 'scipy', 'sklearn', 'yaml', 'tqdm'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            missing.append(module)
    
    if missing:
        print(f"\nMissing modules: {missing}")
        print("Run: pip install " + " ".join(missing))
    else:
        print("\n✓ All modules available")
    
    return len(missing) == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup and maintenance utilities")
    parser.add_argument('--upgrade', action='store_true', help='Upgrade packages')
    parser.add_argument('--clean', action='store_true', help='Clean cache')
    parser.add_argument('--verify', action='store_true', help='Verify installation')
    
    args = parser.parse_args()
    
    if args.upgrade:
        upgrade_packages()
    elif args.clean:
        clean_cache()
    elif args.verify:
        verify_installation()
    else:
        main()