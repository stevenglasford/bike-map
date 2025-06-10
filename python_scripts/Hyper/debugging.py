"""
Debug script to analyze your current cached features
"""

import pickle
import numpy as np
import logging

def debug_cached_features(cache_dir):
    """Debug existing cached features to identify issues"""
    
    print("=== FEATURE DEBUGGING REPORT ===")
    
    # Load video features
    video_cache_path = f"{cache_dir}/ultra_video_features.pkl"
    try:
        with open(video_cache_path, 'rb') as f:
            video_features = pickle.load(f)
        
        print(f"\n📹 VIDEO FEATURES ANALYSIS:")
        print(f"Total videos: {len(video_features)}")
        
        valid_videos = 0
        for video_path, features in video_features.items():
            if features is None:
                print(f"❌ {video_path}: features is None")
                continue
            
            if not isinstance(features, dict):
                print(f"❌ {video_path}: features is not dict, got {type(features)}")
                continue
            
            print(f"\n🎬 {video_path}:")
            valid_features = 0
            
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        print(f"  ⚠️  {key}: empty array")
                    elif np.all(value == 0):
                        print(f"  ⚠️  {key}: all zeros (shape: {value.shape})")
                    elif not np.isfinite(value).all():
                        print(f"  ❌ {key}: contains inf/nan (shape: {value.shape})")
                    else:
                        print(f"  ✅ {key}: valid (shape: {value.shape}, mean: {np.mean(value):.4f})")
                        valid_features += 1
                else:
                    print(f"  ℹ️  {key}: {type(value)} = {value}")
            
            if valid_features > 0:
                valid_videos += 1
                print(f"  📊 Valid features: {valid_features}")
            else:
                print(f"  ❌ No valid array features found!")
        
        print(f"\n📈 VIDEO SUMMARY: {valid_videos}/{len(video_features)} videos have valid features")
        
    except Exception as e:
        print(f"❌ Error loading video features: {e}")
    
    # Load GPX features
    gpx_cache_path = f"{cache_dir}/ultra_gpx_features.pkl"
    try:
        with open(gpx_cache_path, 'rb') as f:
            gpx_database = pickle.load(f)
        
        print(f"\n🗺️  GPX FEATURES ANALYSIS:")
        print(f"Total GPX files: {len(gpx_database)}")
        
        valid_gpx = 0
        for gpx_path, gpx_data in gpx_database.items():
            if gpx_data is None:
                print(f"❌ {gpx_path}: gpx_data is None")
                continue
            
            if 'features' not in gpx_data:
                print(f"❌ {gpx_path}: no 'features' key")
                continue
            
            features = gpx_data['features']
            print(f"\n🧭 {gpx_path}:")
            
            valid_features = 0
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        print(f"  ⚠️  {key}: empty array")
                    elif np.all(value == 0):
                        print(f"  ⚠️  {key}: all zeros (shape: {value.shape})")
                    elif not np.isfinite(value).all():
                        print(f"  ❌ {key}: contains inf/nan (shape: {value.shape})")
                    else:
                        print(f"  ✅ {key}: valid (shape: {value.shape}, mean: {np.mean(value):.4f})")
                        valid_features += 1
                else:
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        print(f"  ✅ {key}: {value}")
                    else:
                        print(f"  ⚠️  {key}: {type(value)} = {value}")
            
            if valid_features > 0:
                valid_gpx += 1
                print(f"  📊 Valid features: {valid_features}")
            else:
                print(f"  ❌ No valid array features found!")
        
        print(f"\n📈 GPX SUMMARY: {valid_gpx}/{len(gpx_database)} GPX files have valid features")
        
    except Exception as e:
        print(f"❌ Error loading GPX features: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    else:
        cache_dir = "./gpu_cache"
    
    debug_cached_features(cache_dir)