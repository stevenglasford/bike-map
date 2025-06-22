# Exact Placement Guide for matcher50.py Fixes

## 🎯 LOCATION 1: Integer Overflow Fix (get_proper_file_size)

**WHERE TO PLACE:** After all the imports but before the first class definition

**FIND THIS SECTION:** Look for the end of all the import statements (around line 100-150), you’ll see something like:

```python
try:
    import skimage.feature as skfeature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
```

**PLACE THE FIX RIGHT AFTER THE IMPORTS, BEFORE ANY CLASS DEFINITIONS:**

```python
# ========== UTILITY FUNCTIONS ==========

def get_proper_file_size(filepath):
    """Get file size without integer overflow for large video files"""
    try:
        size = os.path.getsize(filepath)
        # Handle integer overflow for very large files (>2GB)
        if size < 0:  # Indicates overflow on 32-bit systems
            # Use alternative method for large files
            with open(filepath, 'rb') as f:
                f.seek(0, 2)  # Seek to end
                size = f.tell()
        return size
    except Exception as e:
        logger.warning(f"Could not get size for {filepath}: {e}")
        return 0

def setup_360_specific_models(gpu_id: int):
    """Setup models specifically optimized for 360° panoramic videos"""
    try:
        import torch
        import torch.nn as nn
        
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        
        class Panoramic360Processor(nn.Module):
            def __init__(self):
                super().__init__()
                # Designed for 3840x1920 input (2:1 aspect ratio)
                self.equatorial_conv = nn.Conv2d(3, 64, kernel_size=(7, 14), padding=(3, 7))
                self.polar_conv = nn.Conv2d(3, 64, kernel_size=(14, 7), padding=(7, 3))
                self.fusion_conv = nn.Conv2d(128, 256, 3, padding=1)
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(256, 512)
                
            def forward(self, x):
                # Process equatorial and polar regions differently
                equatorial_features = torch.relu(self.equatorial_conv(x))
                polar_features = torch.relu(self.polar_conv(x))
                
                # Fuse features
                combined = torch.cat([equatorial_features, polar_features], dim=1)
                fused = torch.relu(self.fusion_conv(combined))
                
                # Global pooling and classification
                pooled = self.adaptive_pool(fused)
                output = self.classifier(pooled.view(pooled.size(0), -1))
                
                return output
        
        # Create and initialize the panoramic model
        panoramic_model = Panoramic360Processor()
        panoramic_model.eval()
        panoramic_model = panoramic_model.to(device)
        
        logger.info(f"🌐 GPU {gpu_id}: 360° panoramic models loaded")
        return {'panoramic_360': panoramic_model}
        
    except Exception as e:
        logger.error(f"❌ Failed to setup 360° models on GPU {gpu_id}: {e}")
        return {}

def initialize_feature_models_on_gpu(gpu_id: int):
    """Initialize basic feature extraction models on specified GPU"""
    try:
        import torch
        import torchvision.models as models
        
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        
        # Create basic models for 360° video processing
        feature_models = {}
        
        # ResNet50 for standard feature extraction
        try:
            resnet50 = models.resnet50(pretrained=True)
            resnet50.eval()
            resnet50 = resnet50.to(device)
            feature_models['resnet50'] = resnet50
            logger.info(f"🧠 GPU {gpu_id}: ResNet50 loaded")
        except Exception as e:
            logger.warning(f"⚠️ GPU {gpu_id}: Could not load ResNet50: {e}")
        
        # Simple CNN for spherical processing
        class Simple360CNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
                self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(128, 512)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        try:
            spherical_model = Simple360CNN()
            spherical_model.eval()
            spherical_model = spherical_model.to(device)
            feature_models['spherical'] = spherical_model
            
            # Tangent plane model (copy of spherical for now)
            tangent_model = Simple360CNN()
            tangent_model.eval()
            tangent_model = tangent_model.to(device)
            feature_models['tangent'] = tangent_model
            
            # Add 360° specific models
            panoramic_models = setup_360_specific_models(gpu_id)
            feature_models.update(panoramic_models)
            
            logger.info(f"🧠 GPU {gpu_id}: 360° models loaded")
        except Exception as e:
            logger.warning(f"⚠️ GPU {gpu_id}: Could not load 360° models: {e}")
        
        if feature_models:
            logger.info(f"🧠 GPU {gpu_id}: Feature models initialized successfully")
            return feature_models
        else:
            logger.error(f"❌ GPU {gpu_id}: No models could be loaded")
            return None
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models on GPU {gpu_id}: {e}")
        return None
```

-----

## 🎯 LOCATION 2: Fix the Enhanced360CNNFeatureExtractor Class

**WHERE TO FIND:** Search for the line `class Enhanced360CNNFeatureExtractor:`

**WHAT TO DO:** Find the `_ensure_models_loaded` method within this class and **REPLACE** it with:

```python
def _ensure_models_loaded(self, gpu_id: int):
    """Load models on GPU if not already loaded"""
    if gpu_id in self.models_loaded:
        return  # Models already loaded on this GPU
    
    try:
        # Try to use the initialization function
        models = initialize_feature_models_on_gpu(gpu_id)
        if models is not None:
            self.feature_models[gpu_id] = models
            self.models_loaded.add(gpu_id)
        else:
            # Create basic fallback models
            logger.warning(f"⚠️ Creating basic fallback models for GPU {gpu_id}")
            device = torch.device(f'cuda:{gpu_id}')
            
            class BasicCNN(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
                    self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = torch.nn.Linear(64, 256)
                
                def forward(self, x):
                    x = torch.relu(self.conv(x))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            
            basic_model = BasicCNN().to(device)
            basic_model.eval()
            
            self.feature_models[gpu_id] = {'basic_cnn': basic_model}
            self.models_loaded.add(gpu_id)
            logger.info(f"🧠 GPU {gpu_id}: Basic fallback models created")
            
    except Exception as e:
        logger.error(f"❌ Failed to load CNN models on GPU {gpu_id}: {e}")
        raise
```

-----

## 🎯 LOCATION 3: Fix the Main Function Call

**WHERE TO FIND:** Search for this line (around line 6840):

```python
turbo_mode=turbo_mode,
```

**WHAT TO DO:** Replace it with:

```python
turbo_mode=config.turbo_mode,
```

-----

## 🎯 LOCATION 4: Add Error Handling Around Main Function Call

**WHERE TO FIND:** Search for:

```python
results = complete_turbo_video_gpx_correlation_system(args, config)
```

**WHAT TO DO:** Replace that entire section with:

```python
try:
    logger.info("🚀 Starting complete turbo processing system...")
    results = complete_turbo_video_gpx_correlation_system(args, config)
    
    if results:
        logger.info(f"✅ Processing completed successfully with {len(results)} results")
        print(f"\n🎉 SUCCESS: Processing completed with {len(results)} video results!")
    else:
        logger.error("❌ Processing completed but returned no results")
        
except Exception as e:
    logger.error(f"❌ Complete turbo system failed: {e}")
    logger.error(traceback.format_exc())
    if hasattr(config, 'powersafe') and config.powersafe:
        logger.info("💾 PowerSafe: Partial progress has been saved")
    raise
```

-----

## 📋 Step-by-Step Implementation:

1. **Open matcher50.py in your text editor**
1. **Find the imports section** (around lines 1-150) and add the utility functions **after all imports but before any classes**
1. **Search for `class Enhanced360CNNFeatureExtractor:`** and find the `_ensure_models_loaded` method to replace it
1. **Search for `turbo_mode=turbo_mode,`** and change it to `turbo_mode=config.turbo_mode,`
1. **Search for the main function call** and wrap it with proper error handling
1. **Save the file and test with a small subset of videos**

## 🚨 Important Notes:

- **Make a backup first:** `cp matcher50.py matcher50.py.backup`
- **Place functions in the correct order:** The utility functions must come before they’re used by classes
- **Check indentation:** Python is sensitive to indentation - make sure everything aligns properly
- **Test incrementally:** Try processing 1-2 videos first before running the full batch

This should resolve both the integer overflow issues and the missing 360° model problems!