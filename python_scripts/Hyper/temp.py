#!/usr/bin/env python3
"""
Complete Integration Patch for matcher41.py
This script automatically integrates the GPU-optimized chunked processor

Usage: python integration_patch.py
"""

import os
import shutil
import re
import sys
from pathlib import Path

def create_backup(filepath):
    """Create a backup of the original file"""
    backup_path = filepath + ".pre_chunked_backup"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def add_chunked_imports(content):
    """Add imports needed for chunked processing"""
    
    # New imports to add
    new_imports = """
# GPU-Optimized Chunked Processing Imports
import threading
import queue
from collections import deque"""
    
    # Find a good place to insert imports (after existing imports)
    import_pattern = r'(from typing import Dict, List, Tuple, Optional, Any\s*\n)'
    
    if re.search(import_pattern, content):
        content = re.sub(import_pattern, r'\1' + new_imports + '\n', content)
        print("✅ Added chunked processing imports")
    else:
        # Fallback: add after the first import block
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif line.strip() == '' and insert_idx > 0:
                break
        
        if insert_idx > 0:
            lines.insert(insert_idx, new_imports)
            content = '\n'.join(lines)
            print("✅ Added chunked processing imports (fallback method)")
    
    return content

def enhance_processing_config(content):
    """Enhance ProcessingConfig for chunked processing"""
    
    enhanced_config = '''@dataclass
class ProcessingConfig:
    """Enhanced configuration for chunked GPU processing"""
    max_frames: int = 999999  # Unlimited frames with chunking
    target_size: Tuple[int, int] = (3840, 2160)  # 4K default
    sample_rate: float = 3.0
    parallel_videos: int = 2  # Optimized for dual GPU
    gpu_memory_fraction: float = 0.95
    motion_threshold: float = 0.01
    temporal_window: int = 10
    powersafe: bool = False
    save_interval: int = 5
    gpu_timeout: int = 60
    strict: bool = False
    strict_fail: bool = False
    memory_efficient: bool = True
    max_gpu_memory_gb: float = 15.0  # Per GPU
    enable_preprocessing: bool = True
    ram_cache_gb: float = 100.0  # Use up to 100GB RAM
    disk_cache_gb: float = 1000.0
    cache_dir: str = "~/penis/temp"
    replace_originals: bool = False
    skip_validation: bool = False
    no_quarantine: bool = False
    validation_only: bool = False
    max_ram_usage_gb: float = 100.0
    gpu_memory_safety_margin: float = 0.95
    enable_ram_fallback: bool = True
    dynamic_resolution_scaling: bool = False  # Disabled - use chunking instead
    
    # Chunked processing parameters
    enable_chunked_processing: bool = True
    chunk_frames: int = 60
    chunk_overlap: int = 5
    max_chunk_memory_gb: float = 4.0'''
    
    # Find and replace ProcessingConfig
    config_pattern = r'@dataclass\s*\nclass ProcessingConfig:.*?dynamic_resolution_scaling: bool = True'
    
    match = re.search(config_pattern, content, re.DOTALL)
    if match:
        content = content.replace(match.group(0), enhanced_config)
        print("✅ Enhanced ProcessingConfig with chunked processing parameters")
    else:
        print("⚠️ Could not find ProcessingConfig - will add it manually")
        # Add enhanced config after imports
        content = content.replace(
            "logger = setup_logging()",
            enhanced_config + "\n\nlogger = setup_logging()"
        )
    
    return content

def replace_video_processing_function(content):
    """Replace the main video processing function with chunked version"""
    
    # New chunked processing function
    new_processing_function = '''def process_video_parallel_enhanced(args) -> Tuple[str, Optional[Dict]]:
    """GPU-Optimized chunked video processing with unlimited resolution/frames"""
    video_path, gpu_manager, config, powersafe_manager = args
    
    # Mark as processing in power-safe mode
    if powersafe_manager:
        powersafe_manager.mark_video_processing(video_path)
    
    try:
        # Use chunked processing for unlimited resolution/frames
        if getattr(config, 'enable_chunked_processing', True):
            try:
                # Import chunked processor
                from gpu_optimized_chunked_processor import ChunkedVideoProcessor
                
                processor = ChunkedVideoProcessor(gpu_manager, config)
                features = processor.process_video_chunked(video_path)
                
                if features is not None:
                    features['processing_mode'] = 'GPU_CHUNKED_OPTIMIZED'
                    
                    if powersafe_manager:
                        powersafe_manager.mark_video_features_done(video_path)
                    
                    logger.info(f"🚀 Chunked processing successful: {Path(video_path).name}")
                    return video_path, features
                else:
                    logger.warning(f"Chunked processing failed, trying fallback: {Path(video_path).name}")
                    
            except ImportError as e:
                logger.error(f"❌ ChunkedVideoProcessor not available: {e}")
                logger.error("Please ensure gpu_optimized_chunked_processor.py exists")
                return video_path, None
            except Exception as e:
                logger.warning(f"Chunked processing error, trying fallback: {e}")
        
        # Fallback to original processing if chunked fails
        logger.info(f"Using fallback processing: {Path(video_path).name}")
        return original_process_video_parallel_enhanced(args)
        
    except Exception as e:
        error_msg = f"All video processing methods failed: {str(e)}"
        logger.error(f"{error_msg} for {Path(video_path).name}")
        if powersafe_manager:
            powersafe_manager.mark_video_failed(video_path, error_msg)
        return video_path, None'''
    
    # Find the existing function
    func_pattern = r'def process_video_parallel_enhanced\(args\) -> Tuple\[str, Optional\[Dict\]\]:.*?finally:\s*if gpu_id is not None:.*?pass'
    
    match = re.search(func_pattern, content, re.DOTALL)
    if match:
        original_function = match.group(0)
        
        # Rename original function to use as fallback
        renamed_original = original_function.replace(
            'def process_video_parallel_enhanced(',
            'def original_process_video_parallel_enhanced('
        )
        
        # Replace with new function + renamed original
        replacement = new_processing_function + '\n\n' + renamed_original
        content = content.replace(original_function, replacement)
        print("✅ Replaced video processing function with chunked version")
    else:
        print("⚠️ Could not find process_video_parallel_enhanced function")
        print("   Please manually integrate the chunked processor")
    
    return content

def update_argument_defaults(content):
    """Update argument parser defaults for high performance"""
    
    # Update max_frames default
    content = re.sub(
        r'parser\.add_argument\("--max_frames", type=int, default=150,',
        'parser.add_argument("--max_frames", type=int, default=999999,',
        content
    )
    
    # Update video_size default  
    content = re.sub(
        r'parser\.add_argument\("--video_size", nargs=2, type=int, default=\[480, 270\],',
        'parser.add_argument("--video_size", nargs=2, type=int, default=[3840, 2160],',
        content
    )
    
    # Update parallel_videos default
    content = re.sub(
        r'parser\.add_argument\("--parallel_videos", type=int, default=1,',
        'parser.add_argument("--parallel_videos", type=int, default=2,',
        content
    )
    
    # Update max_gpu_memory default
    content = re.sub(
        r'parser\.add_argument\("--max_gpu_memory", type=float, default=12\.0,',
        'parser.add_argument("--max_gpu_memory", type=float, default=15.0,',
        content
    )
    
    # Update ram_cache default
    content = re.sub(
        r'parser\.add_argument\("--ram_cache", type=float, default=32\.0,',
        'parser.add_argument("--ram_cache", type=float, default=100.0,',
        content
    )
    
    print("✅ Updated argument defaults for high-performance processing")
    return content

def add_performance_monitoring(content):
    """Add performance monitoring hooks"""
    
    monitoring_code = '''
def monitor_chunked_performance():
    """Monitor chunked processing performance"""
    try:
        import torch
        import psutil
        
        # GPU stats
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                if allocated > 0:  # Only show active GPUs
                    logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        
        # RAM stats
        ram = psutil.virtual_memory()
        ram_used_gb = (ram.total - ram.available) / 1024**3
        ram_total_gb = ram.total / 1024**3
        
        if ram_used_gb > 50:  # Only log if significant RAM usage
            logger.info(f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent:.1f}%)")
            
    except Exception as e:
        logger.debug(f"Performance monitoring error: {e}")
'''
    
    # Add the monitoring function before main()
    main_pattern = r'(def main\(\):)'
    content = re.sub(main_pattern, monitoring_code + r'\n\1', content)
    
    print("✅ Added performance monitoring")
    return content

def create_chunked_processor_stub():
    """Create a stub file for the chunked processor"""
    
    stub_content = '''#!/usr/bin/env python3
"""
GPU-Optimized Chunked Video Processor Stub

⚠️ IMPORTANT: This is just a stub file!

To complete the integration, you need to:
1. Copy the complete ChunkedVideoProcessor code from the artifact
2. Paste it into this file, replacing this stub content

The complete code should be around 500+ lines and include:
- ChunkConfig class
- GPUMemoryPool class  
- ChunkedVideoProcessor class
- OptimizedFeatureExtractor class
- Integration functions

Without the complete code, chunked processing will not work!
"""

print("❌ ERROR: ChunkedVideoProcessor stub - complete implementation needed!")
print("Please copy the full GPU-Optimized Chunked Video Processor code into this file")

class ChunkedVideoProcessor:
    def __init__(self, *args, **kwargs):
        raise ImportError("ChunkedVideoProcessor stub - needs complete implementation!")
'''
    
    stub_path = "gpu_optimized_chunked_processor.py"
    
    if not os.path.exists(stub_path):
        with open(stub_path, 'w') as f:
            f.write(stub_content)
        print(f"📄 Created stub file: {stub_path}")
        print("⚠️  You MUST copy the complete ChunkedVideoProcessor code into this file!")
        return True
    else:
        print(f"📄 File {stub_path} already exists - checking if it's complete...")
        
        # Check if it's the real implementation or stub
        with open(stub_path, 'r') as f:
            content = f.read()
        
        if "ChunkedVideoProcessor stub" in content:
            print("⚠️  Still contains stub - you need to copy the complete implementation!")
            return False
        elif "class ChunkedVideoProcessor:" in content and len(content) > 10000:
            print("✅ Complete ChunkedVideoProcessor implementation found!")
            return True
        else:
            print("⚠️  File exists but may be incomplete - please verify it contains the full implementation")
            return False

def verify_integration(content):
    """Verify the integration was successful"""
    
    checks = [
        ("Enhanced ProcessingConfig", "enable_chunked_processing: bool = True"),
        ("Chunked processing function", "GPU_CHUNKED_OPTIMIZED"),
        ("Original function renamed", "original_process_video_parallel_enhanced"),
        ("Performance monitoring", "monitor_chunked_performance"),
        ("High-performance defaults", "default=999999")
    ]
    
    print("\n🔍 Verifying integration:")
    all_good = True
    
    for check_name, check_pattern in checks:
        if check_pattern in content:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_good = False
    
    return all_good

def main():
    """Main integration function"""
    
    print("🚀 GPU-OPTIMIZED CHUNKED PROCESSOR INTEGRATION")
    print("=" * 60)
    
    # Check if matcher41.py exists
    script_path = "matcher41.py"
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found in current directory")
        print("Please run this script in the same directory as matcher41.py")
        return False
    
    print(f"📁 Found {script_path}")
    
    # Create backup
    backup_path = create_backup(script_path)
    
    # Read the original file
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📖 Read {len(content)} characters from {script_path}")
    except Exception as e:
        print(f"❌ Error reading {script_path}: {e}")
        return False
    
    # Apply all patches
    print("\n🔧 Applying patches...")
    
    try:
        content = add_chunked_imports(content)
        content = enhance_processing_config(content)
        content = replace_video_processing_function(content)
        content = update_argument_defaults(content)
        content = add_performance_monitoring(content)
        
        print("✅ All patches applied successfully")
        
    except Exception as e:
        print(f"❌ Error applying patches: {e}")
        return False
    
    # Write the patched file
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Wrote patched content to {script_path}")
    except Exception as e:
        print(f"❌ Error writing patched file: {e}")
        # Restore backup
        shutil.copy2(backup_path, script_path)
        print(f"🔄 Restored backup due to write error")
        return False
    
    # Create chunked processor stub
    processor_ready = create_chunked_processor_stub()
    
    # Verify integration
    integration_good = verify_integration(content)
    
    # Final status
    print("\n" + "=" * 60)
    if integration_good:
        print("✅ INTEGRATION SUCCESSFUL!")
        print(f"📁 Original file backed up to: {backup_path}")
        print(f"🚀 Enhanced {script_path} ready for chunked processing")
        
        if processor_ready:
            print("\n🎯 READY TO USE:")
            print("python matcher41.py -d /path/to/data --gpu_ids 0 1 --video_size 3840 2160")
        else:
            print("\n⚠️  MANUAL STEP REQUIRED:")
            print("1. Copy the complete ChunkedVideoProcessor code")
            print("2. Paste it into gpu_optimized_chunked_processor.py")
            print("3. Then run your enhanced script")
        
        print("\n💡 For maximum performance:")
        print("python matcher41.py -d /path/to/data \\")
        print("    --gpu_ids 0 1 \\")
        print("    --max_gpu_memory 15.0 \\") 
        print("    --parallel_videos 2 \\")
        print("    --max_frames 999999 \\")
        print("    --video_size 3840 2160 \\")
        print("    --ram_cache 100.0")
        
        return True
    else:
        print("❌ INTEGRATION FAILED!")
        print("Some patches were not applied correctly")
        print(f"Original file restored from: {backup_path}")
        # Restore backup
        shutil.copy2(backup_path, script_path)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
