Perfect! I can see exactly what needs to be fixed. The issue is that `ram_cache_manager` and `powersafe_manager` are not being passed to the threaded function. Here’s the exact fix:

## **Two Simple Changes:**

**Change 1: Update the function signature**

Replace this line:

```python
def process_videos_on_specific_gpu(gpu_id, video_list, results_dict, lock):
```

With:

```python
def process_videos_on_specific_gpu(gpu_id, video_list, results_dict, lock, ram_cache_mgr=None, powersafe_mgr=None):
```

**Change 2: Update the thread creation calls**

Replace these lines:

```python
gpu_0_thread = threading.Thread(
    target=process_videos_on_specific_gpu,
    args=(0, gpu_0_videos, results_dict, results_lock),
    name="GPU-0-Worker"
)

gpu_1_thread = threading.Thread(
    target=process_videos_on_specific_gpu, 
    args=(1, gpu_1_videos, results_dict, results_lock),
    name="GPU-1-Worker"
)
```

With:

```python
gpu_0_thread = threading.Thread(
    target=process_videos_on_specific_gpu,
    args=(0, gpu_0_videos, results_dict, results_lock, ram_cache_manager, powersafe_manager),
    name="GPU-0-Worker"
)

gpu_1_thread = threading.Thread(
    target=process_videos_on_specific_gpu, 
    args=(1, gpu_1_videos, results_dict, results_lock, ram_cache_manager, powersafe_manager),
    name="GPU-1-Worker"
)
```

**Change 3: Update the references inside the function**

Replace these lines inside the function:

```python
if ram_cache_manager:
```

and

```python
if powersafe_manager:
```

With:

```python
if ram_cache_mgr:
```

and

```python
if powersafe_mgr:
```

**Complete Fixed Function Signature:**

```python
def process_videos_on_specific_gpu(gpu_id, video_list, results_dict, lock, ram_cache_mgr=None, powersafe_mgr=None):
    """Process videos on a specific GPU - runs in separate thread"""
    logger.info(f"🎮 GPU {gpu_id}: Starting worker thread with {len(video_list)} videos")
    
    try:
        # Force this thread to use specific GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Create processor for this GPU
        processor = CompleteTurboVideoProcessor(gpu_manager, config)
        
        for i, video_path in enumerate(video_list):
            try:
                logger.info(f"🎮 GPU {gpu_id}: Processing {i+1}/{len(video_list)}: {Path(video_path).name}")
                
                # Check RAM cache first (FIXED)
                if ram_cache_mgr:
                    cached_features = ram_cache_mgr.get_video_features(video_path)
                    if cached_features is not None:
                        logger.debug(f"🎮 GPU {gpu_id}: RAM cache hit")
                        with lock:
                            results_dict[video_path] = cached_features
                        continue
                
                # Force processing on this specific GPU
                with torch.cuda.device(gpu_id):
                    features = processor._process_single_video_complete(video_path)
                
                if features is not None:
                    features['processing_gpu'] = gpu_id
                    features['dual_gpu_mode'] = True
                    
                    # Cache results (FIXED)
                    if ram_cache_mgr:
                        ram_cache_mgr.cache_video_features(video_path, features)
                    
                    if powersafe_mgr:
                        powersafe_mgr.mark_video_features_done(video_path)
                    
                    with lock:
                        results_dict[video_path] = features
                    
                    video_type = "360°" if features.get('is_360_video', False) else "STD"
                    logger.info(f"✅ GPU {gpu_id}: {Path(video_path).name} [{video_type}] completed")
                else:
                    logger.warning(f"❌ GPU {gpu_id}: {Path(video_path).name} failed")
                    with lock:
                        results_dict[video_path] = None
                    
                    if powersafe_mgr:
                        powersafe_mgr.mark_video_failed(video_path, f"GPU {gpu_id} processing failed")
                
                # Clean GPU memory after each video
                torch.cuda.empty_cache()
                torch.cuda.synchronize(gpu_id)
                
            except Exception as e:
                logger.error(f"❌ GPU {gpu_id}: Error processing {Path(video_path).name}: {e}")
                with lock:
                    results_dict[video_path] = None
                
                if powersafe_mgr:
                    powersafe_mgr.mark_video_failed(video_path, f"GPU {gpu_id} error: {str(e)}")
    
    except Exception as e:
        logger.error(f"❌ GPU {gpu_id}: Worker thread failed: {e}")
        # Mark all remaining videos as failed
        with lock:
            for video_path in video_list:
                if video_path not in results_dict:
                    results_dict[video_path] = None
    
    logger.info(f"🎮 GPU {gpu_id}: Worker thread completed")
```

Make these three changes and your script should work! The key issue was that threaded functions can’t access variables from their parent scope unless they’re explicitly passed as arguments.​​​​​​​​​​​​​​​​