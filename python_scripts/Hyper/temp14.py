Perfect! I can see exactly what’s needed. The code is trying to access `gpu_video_assignments[gpu_id]` but this variable isn’t defined. I’ll create an ultra-optimized GPU video assignment system specifically designed for your 2x RTX 5060 Ti setup.​​​​​​​​​​​​​​​​

Now I need to show you exactly how to integrate this into your existing code. The error is happening because `gpu_video_assignments` isn’t defined before it’s used. Here’s how to fix it:​​​​​​​​​​​​​​​​

Perfect! I’ve created an ultra-optimized GPU video assignment system specifically designed for your 2x RTX 5060 Ti setup. Here’s how to fix the error:

## **🚀 Step-by-Step Integration:**

### **Step 1: Add the GPU Assignment System**

Add the first artifact (UltraOptimizedGPUVideoAssigner class and all related functions) to your `matcher50.py` file, preferably after your existing imports and before your main correlation system function.

### **Step 2: Fix the Error Location**

In your `complete_turbo_video_gpx_correlation_system` function, find where you have `video_files` defined (probably where you scan for video files), and **RIGHT AFTER** that section, add this code:

```python
# ===== ADD THIS RIGHT AFTER VIDEO_FILES IS DEFINED =====
logger.info("🚀 Creating intelligent GPU video assignments...")

# Create GPU video assignments (THIS FIXES THE ERROR!)
gpu_video_assignments = create_intelligent_gpu_video_assignments(
    video_files=video_files,
    gpu_manager=gpu_manager,
    config=config,
    video_features=video_features if 'video_features' in locals() else None
)

# Validate and log assignments
total_assigned = sum(len(videos) for videos in gpu_video_assignments.values())
logger.info(f"📊 GPU Assignment Summary: {total_assigned} videos distributed")
for gpu_id in gpu_manager.gpu_ids:
    gpu_video_count = len(gpu_video_assignments[gpu_id])  # This line will now work!
    logger.info(f"   GPU {gpu_id}: {gpu_video_count} videos assigned")
```

### **Step 3: Test the Fix**

Run your script:

```bash
./runner.sh
```

## **🚀 What This System Does:**

1. **🎯 Intelligent Load Balancing**: Analyzes video complexity and distributes optimally across your 2 GPUs
1. **📊 Performance Monitoring**: Tracks GPU performance and rebalances if needed
1. **🧠 Memory Awareness**: Calculates optimal assignments based on your 15.5GB GPU memory
1. **🌐 360° Video Detection**: Automatically detects and handles 360° videos with higher complexity
1. **⚡ Turbo Mode Optimization**: Uses advanced algorithms in turbo mode for maximum speed

## **🔧 Key Optimizations for Your System:**

- **RTX 5060 Ti Detection**: Automatically profiles your GPUs’ compute capability (12.0)
- **Memory Optimization**: Uses up to 12 concurrent videos per GPU based on 15.5GB VRAM
- **Load Balancing**: Distributes based on video complexity, file size, and resolution
- **Performance Scaling**: Adapts batch sizes from 16-64 based on GPU memory

## **📊 Expected Performance:**

With this system, you should see:

- **Better GPU utilization** (both GPUs working optimally)
- **Faster processing** (videos assigned based on complexity)
- **Memory efficiency** (no GPU memory overflow)
- **Automatic rebalancing** if one GPU performs better

The error `name 'gpu_video_assignments' is not defined` should be completely resolved, and your script should continue to the actual correlation processing phase.

**Let me know when you’ve added this code and run the script - I’ll help with any new errors that come up!**