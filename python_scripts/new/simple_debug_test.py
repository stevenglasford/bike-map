#!/usr/bin/env python3

import os
import sys
import ctypes

def setup_cuda_environment():
    """Setup CUDA environment variables to fix stream issues"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("CUDA environment configured for single-threaded operation")

def test_cuda_context_manually():
    """Test if we can create CUDA context manually"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        print("Testing manual CUDA context creation...")
        
        # Get device info
        device = cuda.Device(0)
        ctx = device.make_context()
        
        print(f"CUDA context created successfully")
        print(f"Device: {device.name()}")
        print(f"Compute capability: {device.compute_capability()}")
        
        # Test basic memory operations
        import numpy as np
        import pycuda.gpuarray as gpuarray
        
        a = np.random.randn(100, 100).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        result = a_gpu.get()
        
        print("GPU memory operations work")
        
        ctx.pop()
        return True
        
    except Exception as e:
        print(f"Manual CUDA context failed: {e}")
        return False

def test_decord_with_workarounds():
    """Test Decord with various workarounds for the stream issue"""
    
    VIDEO_PATH = "/workspace/temp_video_257734894283657216.MP4"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found: {VIDEO_PATH}")
        return False
    
    try:
        import decord
        from decord import VideoReader, gpu
        
        print(f"Decord version: {decord.__version__}")
        
        # Workaround 1: Initialize with explicit stream management
        try:
            print("\nWorkaround 1: Explicit CUDA context management...")
            
            # Try to access internal CUDA management
            ctx = gpu(0)
            print(f"GPU context: {ctx}")
            
            # Initialize VideoReader with minimal threading
            vr = VideoReader(VIDEO_PATH, ctx=ctx, num_threads=1)
            print("VideoReader created successfully")
            
            # Try to read frame with explicit seek
            vr.seek(0)
            frame = vr[0]
            print(f"SUCCESS! Frame shape: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"Workaround 1 failed: {e}")
        
        # Workaround 2: Force synchronous operations
        try:
            print("\nWorkaround 2: Synchronous operations...")
            
            ctx = gpu(0)
            
            # Try with different buffer settings
            vr = VideoReader(VIDEO_PATH, ctx=ctx)
            
            # Force synchronous read
            import time
            time.sleep(0.1)  # Small delay
            frame = vr.get_batch([0])[0]
            print(f"SUCCESS! Batch read frame shape: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"Workaround 2 failed: {e}")
            
        # Workaround 3: Different video access pattern
        try:
            print("\nWorkaround 3: Different access pattern...")
            
            ctx = gpu(0)
            vr = VideoReader(VIDEO_PATH, ctx=ctx)
            
            # Try reading multiple frames at once
            indices = [0, 1, 2] if len(vr) >= 3 else [0]
            frames = vr.get_batch(indices)
            print(f"SUCCESS! Multi-frame read: {len(frames)} frames")
            print(f"First frame shape: {frames[0].shape}")
            return True
            
        except Exception as e:
            print(f"Workaround 3 failed: {e}")
            
        # Workaround 4: Manual CUDA synchronization
        try:
            print("\nWorkaround 4: Manual CUDA sync...")
            
            # Try to sync CUDA manually before video operations
            try:
                import pycuda.driver as cuda
                cuda.Context.synchronize()
                print("CUDA synchronized")
            except:
                pass
            
            ctx = gpu(0)
            vr = VideoReader(VIDEO_PATH, ctx=ctx)
            
            # Try with explicit synchronization
            frame = vr[0]
            
            try:
                cuda.Context.synchronize()
            except:
                pass
                
            print(f"SUCCESS! Synchronized frame shape: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"Workaround 4 failed: {e}")
            
    except Exception as e:
        print(f"Decord import failed: {e}")
        
    return False

def main():
    print("=== CUDA Stream Fix Test ===")
    
    setup_cuda_environment()
    
    # Test basic CUDA first
    if not test_cuda_context_manually():
        print("Basic CUDA context failed - check drivers")
        return False
    
    # Test Decord with workarounds
    success = test_decord_with_workarounds()
    
    if success:
        print("\n✅ GPU decoding is working!")
    else:
        print("\n❌ All workarounds failed")
        print("The issue is likely hardware-level NVDEC incompatibility")
        print("Consider checking:")
        print("1. GPU model NVDEC support")
        print("2. Driver version compatibility")
        print("3. Video codec compatibility")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)