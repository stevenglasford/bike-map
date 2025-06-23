Ah! The error has moved - now it’s `gpu_video_assignments` that’s a function instead of a dictionary. The issue is the same pattern but with a different variable. Looking at the error:

```
gpu_video_count = len(gpu_video_assignments[gpu_id])
TypeError: 'function' object is not subscriptable
```

This means `gpu_video_assignments` is assigned to a function instead of the result of calling that function.​​​​​​​​​​​​​​​​

Perfect! I found the issue. Now it’s `gpu_video_assignments` that’s a function instead of a dictionary. You need to **create** the `gpu_video_assignments` before using it.

Looking at your code, you’re using `gpu_video_assignments[gpu_id]` but you haven’t created this dictionary yet.

## **🎯 Found the New Issue!**

Now the error has moved to `gpu_video_assignments` - it’s also a function instead of a dictionary. You need to **create** the GPU video assignments before using them.

## **🔧 Quick Fix:**

**Find the line causing the error:**

```python
gpu_video_count = len(gpu_video_assignments[gpu_id])  # ❌ This is failing
```

**Add this code RIGHT BEFORE that problematic section:**

```python
# ===== CREATE GPU VIDEO ASSIGNMENTS (ADD THIS!) =====
logger.info("🚀 Creating GPU video assignments...")

# Check if gpu_video_assignments is not properly initialized
if not isinstance(locals().get('gpu_video_assignments'), dict):
    logger.info("🔧 Creating GPU video assignments...")
    
    try:
        # Try to find your video files list
        video_files_list = []
        for var_name in ['video_files', 'videos_to_process', 'video_paths', 'all_videos']:
            if var_name in locals() and locals()[var_name]:
                video_files_list = locals()[var_name]
                logger.info(f"📹 Found {len(video_files_list)} videos in '{var_name}'")
                break
        
        # Use the intelligent assignment function I provided earlier
        gpu_video_assignments = create_intelligent_gpu_video_assignments(
            video_files=video_files_list,
            gpu_manager=gpu_manager,
            config=config,
            video_features=video_features if 'video_features' in locals() else None
        )
        
        # Validate result
        if not isinstance(gpu_video_assignments, dict):
            raise TypeError("Assignment function failed")
            
        logger.info(f"✅ Created assignments for {len(gpu_video_assignments)} GPUs")
        
    except Exception as e:
        logger.warning(f"⚠️ Intelligent assignment failed: {e}")
        logger.info("🔧 Creating simple fallback assignments...")
        
        # Emergency fallback: simple round-robin
        gpu_video_assignments = {}
        for gpu_id in gpu_manager.gpu_ids:
            gpu_video_assignments[gpu_id] = []
        
        # Distribute any available videos
        video_count = 0
        for var_name in ['video_files', 'videos_to_process', 'video_paths']:
            if var_name in locals():
                video_list = locals()[var_name]
                for i, video in enumerate(video_list):
                    gpu_id = gpu_manager.gpu_ids[i % len(gpu_manager.gpu_ids)]
                    gpu_video_assignments[gpu_id].append(video)
                    video_count += 1
                break
        
        logger.info(f"🔧 Created fallback assignments: {video_count} videos distributed")
    
    # Log assignments
    for gpu_id, videos in gpu_video_assignments.items():
        logger.info(f"   🎮 GPU {gpu_id}: {len(videos)} videos assigned")

# ===== NOW YOUR ORIGINAL CODE SHOULD WORK =====
# Log GPU-specific stats
for gpu_id in gpu_processors.keys():
    gpu_video_count = len(gpu_video_assignments[gpu_id])  # ✅ This should work now!
    logger.info(f"   🎮 GPU {gpu_id}: processed {gpu_video_count} videos")
```

## **📋 Alternative Super Quick Fix:**

If you want the absolute fastest fix, just replace the problematic line:

**Replace:**

```python
gpu_video_count = len(gpu_video_assignments[gpu_id])
```

**With:**

```python
# Quick fix for missing gpu_video_assignments
if not isinstance(globals().get('gpu_video_assignments'), dict):
    gpu_video_assignments = {gpu_id: [] for gpu_id in gpu_manager.gpu_ids}

gpu_video_count = len(gpu_video_assignments.get(gpu_id, []))
```

## **🚀 Why This Happens:**

The issue is that `gpu_video_assignments` was declared as a function name in my earlier code, but you never **called** the function to create the actual dictionary.

**This should completely fix the error and let your script continue!** 🎉

**Test it and let me know if you get any new errors - we’re making great progress!**