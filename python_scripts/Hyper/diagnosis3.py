#!/usr/bin/env python3
"""
Test TurboGPUBatchEngine directly to find the exact failure point
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock the classes we need to test TurboGPUBatchEngine
class MockGPUManager:
    def __init__(self):
        self.gpu_ids = [0, 1]

class MockConfig:
    def __init__(self):
        self.turbo_mode = True
        self.gpu_batch_size = 32
        self.vectorized_operations = True

def test_correlation_model_creation():
    """Test the correlation model creation that's failing"""
    print("=== TESTING CORRELATION MODEL CREATION ===")
    
    try:
        device = torch.device('cuda:0')
        print(f"Device: {device}")
        
        class TurboBatchCorrelationModel(nn.Module):
            def __init__(self):
                super().__init__()
                print("  Creating neural network parameters...")
                
                # Test each parameter creation individually
                try:
                    self.motion_weight = nn.Parameter(torch.tensor(0.25))
                    print("    ✅ motion_weight created")
                except Exception as e:
                    print(f"    ❌ motion_weight failed: {e}")
                    raise
                
                try:
                    self.temporal_weight = nn.Parameter(torch.tensor(0.20))
                    print("    ✅ temporal_weight created")
                except Exception as e:
                    print(f"    ❌ temporal_weight failed: {e}")
                    raise
                
                try:
                    self.statistical_weight = nn.Parameter(torch.tensor(0.15))
                    self.optical_flow_weight = nn.Parameter(torch.tensor(0.15))
                    self.cnn_weight = nn.Parameter(torch.tensor(0.15))
                    self.dtw_weight = nn.Parameter(torch.tensor(0.10))
                    print("    ✅ All weight parameters created")
                except Exception as e:
                    print(f"    ❌ Weight parameters failed: {e}")
                    raise
                
                try:
                    self.batch_norm = nn.BatchNorm1d(6)
                    print("    ✅ BatchNorm1d created")
                except Exception as e:
                    print(f"    ❌ BatchNorm1d failed: {e}")
                    raise
                    
                print("  ✅ Model __init__ complete")
            
            def forward(self, video_features_batch, gps_features_batch):
                print("    Forward pass called...")
                device = video_features_batch.device
                
                if gps_features_batch.device != device:
                    gps_features_batch = gps_features_batch.to(device, non_blocking=True)
                
                if device.type != 'cuda':
                    raise RuntimeError(f"Expected CUDA device, got {device}")
                
                batch_size = video_features_batch.shape[0]
                print(f"    Batch size: {batch_size}")
                
                # Test individual correlation methods
                try:
                    motion_corr = self._compute_motion_correlation_batch(video_features_batch, gps_features_batch)
                    print(f"    ✅ Motion correlation: {motion_corr.shape}")
                except Exception as e:
                    print(f"    ❌ Motion correlation failed: {e}")
                    raise
                
                # Create dummy correlations for testing
                all_corr = torch.randn(batch_size, 6, device=device)
                
                # Test batch norm
                try:
                    all_corr = self.batch_norm(all_corr)
                    print("    ✅ Batch normalization passed")
                except Exception as e:
                    print(f"    ❌ Batch normalization failed: {e}")
                    raise
                
                # Test weighted combination
                try:
                    weights = torch.stack([self.motion_weight, self.temporal_weight, self.statistical_weight,
                                         self.optical_flow_weight, self.cnn_weight, self.dtw_weight]).to(device, non_blocking=True)
                    weights = F.softmax(weights, dim=0)
                    combined_scores = torch.sum(all_corr * weights.unsqueeze(0), dim=1)
                    result = torch.sigmoid(combined_scores)
                    print(f"    ✅ Forward pass complete: {result.shape}")
                    return result
                except Exception as e:
                    print(f"    ❌ Forward pass failed: {e}")
                    raise
            
            def _compute_motion_correlation_batch(self, video_batch, gps_batch):
                # Simplified version
                video_motion = torch.mean(video_batch, dim=-1)
                gps_motion = torch.mean(gps_batch, dim=-1)
                
                video_motion = F.normalize(video_motion, dim=-1, eps=1e-8)
                gps_motion = F.normalize(gps_motion, dim=-1, eps=1e-8)
                
                correlation = F.cosine_similarity(video_motion, gps_motion, dim=-1)
                return torch.abs(correlation)
        
        print("Creating model...")
        model = TurboBatchCorrelationModel()
        print("✅ Model created successfully!")
        
        print("Moving model to GPU...")
        model = model.to(device, non_blocking=True)
        print("✅ Model moved to GPU!")
        
        print("Testing forward pass...")
        # Test with dummy data
        batch_size = 4
        feature_dim = 128
        sequence_length = 10
        
        video_features = torch.randn(batch_size, feature_dim, sequence_length, device=device)
        gps_features = torch.randn(batch_size, feature_dim, sequence_length, device=device)
        
        with torch.no_grad():
            output = model(video_features, gps_features)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Correlation model creation failed: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None

def test_turbo_gpu_batch_engine():
    """Test the full TurboGPUBatchEngine"""
    print("\n=== TESTING TURBO GPU BATCH ENGINE ===")
    
    try:
        gpu_manager = MockGPUManager()
        config = MockConfig()
        
        print(f"GPU Manager GPU IDs: {gpu_manager.gpu_ids}")
        print(f"Config: turbo_mode={config.turbo_mode}, gpu_batch_size={config.gpu_batch_size}")
        
        # Test the constructor step by step
        correlation_models = {}
        
        for gpu_id in gpu_manager.gpu_ids:
            print(f"\n--- Testing GPU {gpu_id} ---")
            
            try:
                device = torch.device(f'cuda:{gpu_id}')
                print(f"Device created: {device}")
                
                model = test_correlation_model_creation()
                if model is None:
                    print(f"❌ GPU {gpu_id}: Model creation failed")
                    break
                    
                correlation_models[gpu_id] = model
                print(f"✅ GPU {gpu_id}: Model stored successfully")
                
            except Exception as e:
                print(f"❌ GPU {gpu_id} failed: {e}")
                break
        
        if len(correlation_models) == len(gpu_manager.gpu_ids):
            print(f"\n✅ TurboGPUBatchEngine would initialize successfully!")
            print(f"   Models created for all GPUs: {list(correlation_models.keys())}")
            return True
        else:
            print(f"\n❌ TurboGPUBatchEngine would fail!")
            print(f"   Only {len(correlation_models)} of {len(gpu_manager.gpu_ids)} GPUs succeeded")
            return False
            
    except Exception as e:
        print(f"❌ TurboGPUBatchEngine test failed: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=== DIRECT TURBOGPUBATCHENGINE TEST ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        success = test_turbo_gpu_batch_engine()
        print(f"\n{'='*50}")
        if success:
            print("🚀 DIAGNOSIS: TurboGPUBatchEngine should work!")
            print("🔍 NEXT STEP: Check matcher50.py for missing imports or different class definitions")
        else:
            print("❌ DIAGNOSIS: TurboGPUBatchEngine has issues!")
            print("🔧 NEXT STEP: Fix the issues shown above")
    else:
        print("❌ CUDA not available!")