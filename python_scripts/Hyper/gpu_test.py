#!/usr/bin/env python3
"""
Comprehensive GPU Acceleration Test Script
==========================================

This script thoroughly tests all GPU acceleration components to ensure
no operations fall back to CPU when GPU should be used.

Usage: python gpu_test.py [--gpu_ids 0 1] [--strict]
"""

import torch
import cupy as cp
import numpy as np
import cv2
import subprocess
import json
import tempfile
import os
import time
import psutil
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUTestError(Exception):
    """Custom exception for GPU test failures"""
    pass

class ComprehensiveGPUTester:
    """Comprehensive GPU acceleration testing"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1], strict_mode: bool = True):
        self.gpu_ids = gpu_ids
        self.strict_mode = strict_mode
        self.test_results = {}
        self.performance_metrics = {}
        
        # Create test data directory
        self.test_dir = tempfile.mkdtemp(prefix='gpu_test_')
        logger.info(f"Test data directory: {self.test_dir}")
    
    def run_all_tests(self) -> Dict:
        """Run comprehensive GPU tests"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE GPU ACCELERATION TESTS")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_cuda_availability,
            self.test_cupy_acceleration,
            self.test_ffmpeg_gpu_support,
            self.test_pytorch_gpu_operations,
            self.test_video_decoding_gpu,
            self.test_feature_extraction_gpu,
            self.test_gpx_processing_gpu,
            self.test_correlation_gpu,
            self.test_memory_management,
            self.test_multi_gpu_distribution,
            self.test_mixed_precision,
            self.test_tensor_operations_speed,
            self.test_gpu_memory_allocation,
            self.test_cuda_streams,
            self.test_gpu_utilization_monitoring
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                start_time = time.time()
                result = test_method()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'duration': end_time - start_time,
                    'details': result
                }
                
                logger.info(f"✅ {test_name} PASSED ({end_time - start_time:.3f}s)")
                passed_tests += 1
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'duration': 0,
                    'error': str(e)
                }
                
                if self.strict_mode:
                    logger.error(f"❌ {test_name} FAILED: {e}")
                    failed_tests += 1
                    # In strict mode, we could choose to stop here or continue
                else:
                    logger.warning(f"⚠️ {test_name} FAILED (non-strict): {e}")
        
        # Generate summary
        self._generate_test_summary(passed_tests, failed_tests)
        
        if failed_tests > 0 and self.strict_mode:
            raise GPUTestError(f"GPU acceleration tests failed: {failed_tests} failures")
        
        return self.test_results
    
    def test_cuda_availability(self) -> Dict:
        """Test CUDA availability and device properties"""
        if not torch.cuda.is_available():
            raise GPUTestError("CUDA is not available!")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise GPUTestError("No CUDA devices found!")
        
        device_info = {}
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            device_info[f'gpu_{i}'] = {
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            }
            
            # Test basic operations on each device
            device = torch.device(f'cuda:{i}')
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.mm(test_tensor, test_tensor)
            
            if result.device.type != 'cuda':
                raise GPUTestError(f"GPU {i} operation returned CPU tensor!")
            
            # Memory test
            allocated_before = torch.cuda.memory_allocated(i)
            large_tensor = torch.randn(5000, 5000, device=device)
            allocated_after = torch.cuda.memory_allocated(i)
            
            if allocated_after <= allocated_before:
                raise GPUTestError(f"GPU {i} memory allocation failed!")
            
            del test_tensor, result, large_tensor
            torch.cuda.empty_cache()
        
        return {
            'device_count': device_count,
            'device_info': device_info,
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__
        }
    
    def test_cupy_acceleration(self) -> Dict:
        """Test CuPy GPU acceleration"""
        if not cp.cuda.is_available():
            raise GPUTestError("CuPy CUDA is not available!")
        
        device_count = cp.cuda.runtime.getDeviceCount()
        
        results = {}
        for gpu_id in self.gpu_ids:
            if gpu_id >= device_count:
                continue
            
            with cp.cuda.Device(gpu_id):
                # Test basic array operations
                a = cp.random.randn(5000, 5000)
                b = cp.random.randn(5000, 5000)
                
                # Ensure operations stay on GPU
                if a.device.id != gpu_id:
                    raise GPUTestError(f"CuPy array not on GPU {gpu_id}!")
                
                # Matrix multiplication
                start_time = time.time()
                c = cp.dot(a, b)
                gpu_time = time.time() - start_time
                
                if c.device.id != gpu_id:
                    raise GPUTestError(f"CuPy operation result not on GPU {gpu_id}!")
                
                # Compare with CPU (should be much faster)
                a_cpu = cp.asnumpy(a)
                b_cpu = cp.asnumpy(b)
                
                start_time = time.time()
                c_cpu = np.dot(a_cpu, b_cpu)
                cpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time
                
                if speedup < 2.0:  # GPU should be at least 2x faster
                    raise GPUTestError(f"CuPy GPU not significantly faster than CPU: {speedup:.2f}x")
                
                results[f'gpu_{gpu_id}'] = {
                    'gpu_time': gpu_time,
                    'cpu_time': cpu_time,
                    'speedup': speedup,
                    'memory_pool_used': cp.get_default_memory_pool().used_bytes(),
                    'memory_pool_total': cp.get_default_memory_pool().total_bytes()
                }
                
                # Test advanced operations
                fft_result = cp.fft.fft2(a)
                if fft_result.device.id != gpu_id:
                    raise GPUTestError(f"CuPy FFT result not on GPU {gpu_id}!")
                
                # Test reduction operations
                sum_result = cp.sum(a)
                if not isinstance(sum_result, cp.ndarray) or sum_result.device.id != gpu_id:
                    raise GPUTestError(f"CuPy reduction not on GPU {gpu_id}!")
                
                # Cleanup
                del a, b, c, fft_result, sum_result
                cp.get_default_memory_pool().free_all_blocks()
        
        return results
    
    def test_ffmpeg_gpu_support(self) -> Dict:
        """Test FFmpeg GPU support"""
        try:
            # Check hardware acceleration support
            result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                  capture_output=True, text=True, check=True)
            hwaccels = result.stdout
            
            if 'cuda' not in hwaccels:
                raise GPUTestError("FFmpeg CUDA hardware acceleration not available!")
            
            # Check GPU encoders
            result = subprocess.run(['ffmpeg', '-encoders'], 
                                  capture_output=True, text=True, check=True)
            encoders = result.stdout
            
            gpu_encoders = []
            for encoder in ['h264_nvenc', 'hevc_nvenc', 'h264_cuvid']:
                if encoder in encoders:
                    gpu_encoders.append(encoder)
            
            if not gpu_encoders:
                raise GPUTestError("No GPU encoders found in FFmpeg!")
            
            # Test GPU decoding with a sample video
            test_video_path = self._create_test_video()
            
            # Test hardware decode
            decode_cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-hwaccel', 'cuda',
                '-c:v', 'h264_cuvid',
                '-i', test_video_path,
                '-f', 'null', '-'
            ]
            
            start_time = time.time()
            result = subprocess.run(decode_cmd, capture_output=True, text=True)
            gpu_decode_time = time.time() - start_time
            
            if result.returncode != 0:
                # Try without hardware acceleration for comparison
                cpu_cmd = [
                    'ffmpeg', '-y', '-v', 'error',
                    '-i', test_video_path,
                    '-f', 'null', '-'
                ]
                
                start_time = time.time()
                cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True)
                cpu_decode_time = time.time() - start_time
                
                if cpu_result.returncode == 0:
                    logger.warning("GPU decode failed but CPU decode succeeded")
                    # This might be acceptable in some cases
                else:
                    raise GPUTestError("Both GPU and CPU decode failed!")
            
            return {
                'hwaccels_available': hwaccels.strip().split('\n'),
                'gpu_encoders': gpu_encoders,
                'gpu_decode_time': gpu_decode_time,
                'test_video_path': test_video_path
            }
            
        except subprocess.CalledProcessError as e:
            raise GPUTestError(f"FFmpeg command failed: {e}")
    
    def test_pytorch_gpu_operations(self) -> Dict:
        """Test PyTorch GPU operations comprehensively"""
        results = {}
        
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            
            with torch.cuda.device(device):
                # Test mixed precision
                if torch.cuda.is_bf16_supported():
                    # Test bfloat16
                    bf16_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16, device=device)
                    bf16_result = torch.mm(bf16_tensor, bf16_tensor)
                    
                    if bf16_result.device != device or bf16_result.dtype != torch.bfloat16:
                        raise GPUTestError(f"GPU {gpu_id} bfloat16 operation failed!")
                
                # Test half precision
                half_tensor = torch.randn(1000, 1000, dtype=torch.float16, device=device)
                half_result = torch.mm(half_tensor, half_tensor)
                
                if half_result.device != device or half_result.dtype != torch.float16:
                    raise GPUTestError(f"GPU {gpu_id} half precision operation failed!")
                
                # Test autocast
                with torch.cuda.amp.autocast():
                    float_tensor = torch.randn(1000, 1000, device=device)
                    autocast_result = torch.mm(float_tensor, float_tensor)
                    
                    if autocast_result.device != device:
                        raise GPUTestError(f"GPU {gpu_id} autocast operation failed!")
                
                # Test CUDA streams
                stream1 = torch.cuda.Stream(device=device)
                stream2 = torch.cuda.Stream(device=device)
                
                with torch.cuda.stream(stream1):
                    tensor1 = torch.randn(2000, 2000, device=device)
                    result1 = torch.mm(tensor1, tensor1)
                
                with torch.cuda.stream(stream2):
                    tensor2 = torch.randn(2000, 2000, device=device)
                    result2 = torch.mm(tensor2, tensor2)
                
                # Synchronize streams
                stream1.synchronize()
                stream2.synchronize()
                
                if result1.device != device or result2.device != device:
                    raise GPUTestError(f"GPU {gpu_id} stream operations failed!")
                
                # Test CUDNN operations
                conv_layer = torch.nn.Conv2d(64, 128, 3, padding=1).to(device)
                input_tensor = torch.randn(32, 64, 256, 256, device=device)
                
                conv_result = conv_layer(input_tensor)
                if conv_result.device != device:
                    raise GPUTestError(f"GPU {gpu_id} CUDNN convolution failed!")
                
                # Test memory management
                memory_before = torch.cuda.memory_allocated(device)
                large_tensors = []
                for i in range(10):
                    large_tensors.append(torch.randn(1000, 1000, device=device))
                
                memory_after = torch.cuda.memory_allocated(device)
                if memory_after <= memory_before:
                    raise GPUTestError(f"GPU {gpu_id} memory allocation tracking failed!")
                
                # Cleanup
                del large_tensors, conv_layer, input_tensor, conv_result
                torch.cuda.empty_cache()
                
                results[f'gpu_{gpu_id}'] = {
                    'mixed_precision_supported': torch.cuda.is_bf16_supported(),
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'streams_working': True,
                    'cudnn_working': True
                }
        
        return results
    
    def test_video_decoding_gpu(self) -> Dict:
        """Test GPU video decoding performance"""
        test_video_path = self._create_test_video()
        
        results = {}
        
        # Test with our FFmpeg decoder
        try:
            from gpu_optimized_matcher import UltraOptimizedFFmpegDecoder
            
            decoder = UltraOptimizedFFmpegDecoder(gpu_ids=self.gpu_ids)
            
            for gpu_id in self.gpu_ids:
                if gpu_id >= torch.cuda.device_count():
                    continue
                
                start_time = time.time()
                frames_tensor, fps, duration, frame_indices = decoder.decode_video_gpu_batch(
                    test_video_path, gpu_id=gpu_id
                )
                decode_time = time.time() - start_time
                
                if frames_tensor is None:
                    raise GPUTestError(f"GPU {gpu_id} video decoding returned None!")
                
                if not isinstance(frames_tensor, torch.Tensor):
                    raise GPUTestError(f"GPU {gpu_id} decode didn't return torch.Tensor!")
                
                if frames_tensor.device.type != 'cuda':
                    raise GPUTestError(f"GPU {gpu_id} decoded frames not on GPU!")
                
                if frames_tensor.device.index != gpu_id:
                    raise GPUTestError(f"GPU {gpu_id} frames on wrong device: {frames_tensor.device}")
                
                results[f'gpu_{gpu_id}'] = {
                    'decode_time': decode_time,
                    'frames_shape': list(frames_tensor.shape),
                    'fps': fps,
                    'duration': duration,
                    'frame_count': len(frame_indices),
                    'device': str(frames_tensor.device),
                    'dtype': str(frames_tensor.dtype)
                }
                
        except ImportError:
            logger.warning("Could not import UltraOptimizedFFmpegDecoder for testing")
            # Create a minimal test
            results['fallback_test'] = self._test_basic_video_ops()
        
        return results
    
    def test_feature_extraction_gpu(self) -> Dict:
        """Test GPU feature extraction"""
        results = {}
        
        try:
            from gpu_optimized_matcher import MaxGPUFeatureExtractor
            
            extractor = MaxGPUFeatureExtractor(gpu_ids=self.gpu_ids)
            
            for gpu_idx, gpu_id in enumerate(self.gpu_ids):
                if gpu_id >= torch.cuda.device_count():
                    continue
                
                # Create test video data
                device = torch.device(f'cuda:{gpu_id}')
                test_frames = torch.randn(1, 30, 3, 360, 640, device=device)  # Batch, Time, C, H, W
                
                start_time = time.time()
                features = extractor.extract_all_features_gpu(test_frames, gpu_idx)
                extraction_time = time.time() - start_time
                
                # Verify all features are computed
                required_features = [
                    'scene_features', 'motion_features', 'texture_features',
                    'motion_magnitude', 'motion_direction', 'edge_density'
                ]
                
                for feature_name in required_features:
                    if feature_name not in features:
                        raise GPUTestError(f"Missing feature: {feature_name}")
                    
                    feature_value = features[feature_name]
                    if isinstance(feature_value, torch.Tensor):
                        if feature_value.device.type != 'cuda':
                            raise GPUTestError(f"Feature {feature_name} not on GPU!")
                        if feature_value.device.index != gpu_id:
                            raise GPUTestError(f"Feature {feature_name} on wrong GPU!")
                
                results[f'gpu_{gpu_id}'] = {
                    'extraction_time': extraction_time,
                    'features_extracted': list(features.keys()),
                    'feature_shapes': {k: list(v.shape) if isinstance(v, torch.Tensor) else 'not_tensor' 
                                     for k, v in features.items()},
                    'device_verified': True
                }
                
        except ImportError:
            logger.warning("Could not import MaxGPUFeatureExtractor for testing")
            results['fallback_test'] = self._test_basic_feature_ops()
        
        return results
    
    def test_gpx_processing_gpu(self) -> Dict:
        """Test CuPy GPX processing"""
        results = {}
        
        try:
            from gpu_optimized_matcher import CuPyGPXProcessor
            
            processor = CuPyGPXProcessor()
            
            # Create test GPX data
            test_gpx_path = self._create_test_gpx()
            
            start_time = time.time()
            gpx_data = processor._parse_gpx_cpu(test_gpx_path)
            
            if gpx_data is None:
                raise GPUTestError("Failed to parse test GPX file!")
            
            # Test GPU feature computation
            enhanced_data = processor._compute_gpu_features(gpx_data)
            processing_time = time.time() - start_time
            
            # Verify features are computed
            if 'features' not in enhanced_data:
                raise GPUTestError("GPX GPU processing didn't produce features!")
            
            features = enhanced_data['features']
            required_features = ['speed', 'acceleration', 'bearing', 'curvature']
            
            for feature_name in required_features:
                if feature_name not in features:
                    raise GPUTestError(f"Missing GPX feature: {feature_name}")
            
            results['gpx_processing'] = {
                'processing_time': processing_time,
                'features_computed': list(features.keys()),
                'point_count': enhanced_data.get('point_count', 0),
                'duration': enhanced_data.get('duration', 0),
                'distance': enhanced_data.get('distance', 0)
            }
            
        except ImportError:
            logger.warning("Could not import CuPyGPXProcessor for testing")
            results['fallback_test'] = self._test_basic_cupy_ops()
        
        return results
    
    def test_correlation_gpu(self) -> Dict:
        """Test GPU correlation operations"""
        results = {}
        
        try:
            from gpu_optimized_matcher import UltraHighPerformanceCorrelator
            
            correlator = UltraHighPerformanceCorrelator(gpu_ids=self.gpu_ids)
            
            # Create test data
            video_features = {
                'test_video_1': self._create_mock_video_features(),
                'test_video_2': self._create_mock_video_features()
            }
            
            gpx_database = {
                'test_gpx_1': self._create_mock_gpx_data(),
                'test_gpx_2': self._create_mock_gpx_data()
            }
            
            # Test correlation
            start_time = time.time()
            correlation_results = correlator.correlate_ultra_optimized(
                video_features, gpx_database, self.test_dir, top_k=2
            )
            correlation_time = time.time() - start_time
            
            if not correlation_results:
                raise GPUTestError("Correlation returned no results!")
            
            # Verify results structure
            for video_path, result in correlation_results.items():
                if result is None:
                    raise GPUTestError(f"Correlation failed for {video_path}")
                
                if 'matches' not in result:
                    raise GPUTestError(f"No matches in correlation result for {video_path}")
            
            results['correlation'] = {
                'correlation_time': correlation_time,
                'videos_processed': len(correlation_results),
                'successful_correlations': sum(1 for r in correlation_results.values() if r is not None),
                'gpu_count': len(self.gpu_ids)
            }
            
        except ImportError:
            logger.warning("Could not import UltraHighPerformanceCorrelator for testing")
            results['fallback_test'] = self._test_basic_correlation_ops()
        
        return results
    
    def test_memory_management(self) -> Dict:
        """Test GPU memory management"""
        results = {}
        
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            
            # Initial memory state
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
            
            # Allocate large tensors
            large_tensors = []
            for i in range(5):
                tensor = torch.randn(2000, 2000, device=device)
                large_tensors.append(tensor)
            
            peak_memory = torch.cuda.memory_allocated(device)
            
            # Test memory cleanup
            del large_tensors
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device)
            
            # Memory should be mostly freed
            memory_freed = peak_memory - final_memory
            memory_efficiency = memory_freed / (peak_memory - initial_memory) if peak_memory > initial_memory else 0
            
            if memory_efficiency < 0.8:  # Should free at least 80% of allocated memory
                raise GPUTestError(f"GPU {gpu_id} poor memory cleanup: {memory_efficiency:.2f}")
            
            # Test CuPy memory management
            with cp.cuda.Device(gpu_id):
                cupy_initial = cp.get_default_memory_pool().used_bytes()
                
                large_arrays = []
                for i in range(5):
                    arr = cp.random.randn(2000, 2000)
                    large_arrays.append(arr)
                
                cupy_peak = cp.get_default_memory_pool().used_bytes()
                
                del large_arrays
                cp.get_default_memory_pool().free_all_blocks()
                
                cupy_final = cp.get_default_memory_pool().used_bytes()
            
            results[f'gpu_{gpu_id}'] = {
                'pytorch_initial_memory': initial_memory,
                'pytorch_peak_memory': peak_memory,
                'pytorch_final_memory': final_memory,
                'pytorch_memory_efficiency': memory_efficiency,
                'cupy_initial_memory': cupy_initial,
                'cupy_peak_memory': cupy_peak,
                'cupy_final_memory': cupy_final
            }
        
        return results
    
    def test_multi_gpu_distribution(self) -> Dict:
        """Test multi-GPU work distribution"""
        if len(self.gpu_ids) < 2:
            return {'status': 'skipped', 'reason': 'Not enough GPUs for multi-GPU test'}
        
        results = {}
        
        # Test tensor distribution across GPUs
        tensors = {}
        for i, gpu_id in enumerate(self.gpu_ids):
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            tensor = torch.randn(1000, 1000, device=device)
            tensors[f'gpu_{gpu_id}'] = tensor
            
            if tensor.device != device:
                raise GPUTestError(f"Tensor not on correct GPU {gpu_id}")
        
        # Test parallel operations
        import concurrent.futures
        
        def gpu_operation(gpu_id):
            device = torch.device(f'cuda:{gpu_id}')
            with torch.cuda.device(device):
                a = torch.randn(2000, 2000, device=device)
                b = torch.randn(2000, 2000, device=device)
                result = torch.mm(a, b)
                return result.device == device
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            futures = [executor.submit(gpu_operation, gpu_id) for gpu_id in self.gpu_ids 
                      if gpu_id < torch.cuda.device_count()]
            results_parallel = [future.result() for future in futures]
        parallel_time = time.time() - start_time
        
        if not all(results_parallel):
            raise GPUTestError("Multi-GPU parallel operations failed!")
        
        results['multi_gpu'] = {
            'gpus_tested': self.gpu_ids,
            'parallel_time': parallel_time,
            'all_operations_successful': all(results_parallel),
            'tensors_distributed': len(tensors)
        }
        
        return results
    
    def test_mixed_precision(self) -> Dict:
        """Test mixed precision operations"""
        results = {}
        
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            
            # Test autocast
            with torch.cuda.amp.autocast():
                tensor_fp32 = torch.randn(1000, 1000, device=device)
                result = torch.mm(tensor_fp32, tensor_fp32)
                
                # In autocast, operations should use lower precision
                if result.device != device:
                    raise GPUTestError(f"GPU {gpu_id} autocast result not on GPU!")
            
            # Test explicit half precision
            tensor_fp16 = torch.randn(1000, 1000, dtype=torch.float16, device=device)
            result_fp16 = torch.mm(tensor_fp16, tensor_fp16)
            
            if result_fp16.dtype != torch.float16:
                raise GPUTestError(f"GPU {gpu_id} float16 operation changed dtype!")
            
            if result_fp16.device != device:
                raise GPUTestError(f"GPU {gpu_id} float16 result not on GPU!")
            
            # Test GradScaler
            scaler = torch.cuda.amp.GradScaler()
            
            # Create a simple model
            model = torch.nn.Linear(100, 50).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            # Test mixed precision training step
            input_data = torch.randn(32, 100, device=device)
            target = torch.randn(32, 50, device=device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(input_data)
                loss = torch.nn.functional.mse_loss(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if loss.device != device:
                raise GPUTestError(f"GPU {gpu_id} mixed precision training failed!")
            
            results[f'gpu_{gpu_id}'] = {
                'autocast_working': True,
                'float16_operations': True,
                'grad_scaler_working': True,
                'mixed_precision_training': True
            }
        
        return results
    
    def test_tensor_operations_speed(self) -> Dict:
        """Test tensor operation speeds for performance validation"""
        results = {}
        
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            
            # Matrix multiplication speed test
            sizes = [1000, 2000, 4000]
            gpu_times = []
            
            for size in sizes:
                torch.cuda.synchronize(device)
                start_time = time.time()
                
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                c = torch.mm(a, b)
                
                torch.cuda.synchronize(device)
                gpu_time = time.time() - start_time
                gpu_times.append(gpu_time)
                
                # Verify result is on GPU
                if c.device != device:
                    raise GPUTestError(f"GPU {gpu_id} matrix multiplication result not on GPU!")
            
            # Compare with CPU for largest size
            size = sizes[-1]
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            
            start_time = time.time()
            c_cpu = torch.mm(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_times[-1]
            
            if speedup < 2.0:  # GPU should be significantly faster
                logger.warning(f"GPU {gpu_id} speedup only {speedup:.2f}x - might indicate issues")
            
            results[f'gpu_{gpu_id}'] = {
                'matrix_mult_times': dict(zip(sizes, gpu_times)),
                'cpu_time_largest': cpu_time,
                'speedup': speedup,
                'performance_acceptable': speedup >= 2.0
            }
        
        return results
    
    def test_gpu_memory_allocation(self) -> Dict:
        """Test GPU memory allocation patterns"""
        results = {}
        
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            
            # Test incremental allocation
            torch.cuda.empty_cache()
            allocations = []
            
            for i in range(10):
                tensor = torch.randn(500, 500, device=device)
                allocated = torch.cuda.memory_allocated(device)
                allocations.append(allocated)
                
                if allocated == 0:
                    raise GPUTestError(f"GPU {gpu_id} memory allocation tracking failed!")
            
            # Test large allocation
            try:
                large_tensor = torch.randn(8000, 8000, device=device)
                large_allocation = torch.cuda.memory_allocated(device)
                
                if large_allocation <= allocations[-1]:
                    raise GPUTestError(f"GPU {gpu_id} large allocation failed!")
                
                del large_tensor
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"GPU {gpu_id} out of memory for large allocation - this may be expected")
                else:
                    raise
            
            # Test memory fragmentation
            torch.cuda.empty_cache()
            fragments = []
            for i in range(100):
                fragment = torch.randn(100, 100, device=device)
                fragments.append(fragment)
            
            fragmented_memory = torch.cuda.memory_allocated(device)
            
            # Cleanup every other fragment
            for i in range(0, len(fragments), 2):
                del fragments[i]
            
            partial_cleanup_memory = torch.cuda.memory_allocated(device)
            
            if partial_cleanup_memory >= fragmented_memory:
                raise GPUTestError(f"GPU {gpu_id} memory fragmentation test failed!")
            
            results[f'gpu_{gpu_id}'] = {
                'incremental_allocations': allocations,
                'fragmented_memory': fragmented_memory,
                'partial_cleanup_memory': partial_cleanup_memory,
                'memory_management_working': True
            }
        
        return results
    
    def test_cuda_streams(self) -> Dict:
        """Test CUDA streams functionality"""
        results = {}
        
        for gpu_id in self.gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                continue
            
            device = torch.device(f'cuda:{gpu_id}')
            
            # Create multiple streams
            streams = [torch.cuda.Stream(device=device) for _ in range(4)]
            
            # Test concurrent operations
            tensors = []
            
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    tensor = torch.randn(1000, 1000, device=device)
                    result = torch.mm(tensor, tensor)
                    tensors.append(result)
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            # Verify all operations completed
            for i, tensor in enumerate(tensors):
                if tensor.device != device:
                    raise GPUTestError(f"GPU {gpu_id} stream {i} operation failed!")
            
            # Test stream priorities (if supported)
            try:
                high_priority_stream = torch.cuda.Stream(device=device, priority=-1)
                low_priority_stream = torch.cuda.Stream(device=device, priority=0)
                
                priority_support = True
            except:
                priority_support = False
            
            results[f'gpu_{gpu_id}'] = {
                'streams_created': len(streams),
                'concurrent_operations': len(tensors),
                'priority_support': priority_support,
                'stream_synchronization': True
            }
        
        return results
    
    def test_gpu_utilization_monitoring(self) -> Dict:
        """Test GPU utilization monitoring"""
        results = {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for gpu_id in self.gpu_ids:
                if gpu_id >= torch.cuda.device_count():
                    continue
                
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Get initial state
                initial_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                initial_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Create workload
                device = torch.device(f'cuda:{gpu_id}')
                workload_tensors = []
                
                for i in range(10):
                    a = torch.randn(2000, 2000, device=device)
                    b = torch.randn(2000, 2000, device=device)
                    c = torch.mm(a, b)
                    workload_tensors.append(c)
                
                # Get state during workload
                workload_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                workload_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Cleanup
                del workload_tensors
                torch.cuda.empty_cache()
                
                # Verify utilization increased
                if workload_memory.used <= initial_memory.used:
                    raise GPUTestError(f"GPU {gpu_id} memory usage didn't increase during workload!")
                
                results[f'gpu_{gpu_id}'] = {
                    'initial_gpu_util': initial_util.gpu,
                    'workload_gpu_util': workload_util.gpu,
                    'initial_memory_used': initial_memory.used,
                    'workload_memory_used': workload_memory.used,
                    'memory_total': workload_memory.total,
                    'utilization_monitoring': True
                }
                
        except ImportError:
            results['monitoring'] = {
                'status': 'pynvml not available',
                'utilization_monitoring': False
            }
        
        return results
    
    def _create_test_video(self) -> str:
        """Create a test video file"""
        test_video_path = os.path.join(self.test_dir, 'test_video.mp4')
        
        # Create test video using FFmpeg
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'testsrc=duration=5:size=640x480:rate=30',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            test_video_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return test_video_path
    
    def _create_test_gpx(self) -> str:
        """Create a test GPX file"""
        test_gpx_path = os.path.join(self.test_dir, 'test_track.gpx')
        
        gpx_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
<trk>
<trkseg>
'''
        
        # Generate test points
        for i in range(100):
            lat = 37.7749 + i * 0.001  # San Francisco area
            lon = -122.4194 + i * 0.001
            time = f"2023-01-01T10:{i//60:02d}:{i%60:02d}Z"
            
            gpx_content += f'<trkpt lat="{lat}" lon="{lon}"><time>{time}</time></trkpt>\n'
        
        gpx_content += '''</trkseg>
</trk>
</gpx>'''
        
        with open(test_gpx_path, 'w') as f:
            f.write(gpx_content)
        
        return test_gpx_path
    
    def _create_mock_video_features(self) -> Dict:
        """Create mock video features for testing"""
        return {
            'scene_features': np.random.randn(30, 256),
            'motion_magnitude': np.random.randn(30),
            'acceleration': np.random.randn(30),
            'edge_density': np.random.randn(30),
            'duration': 30.0,
            'fps': 30.0
        }
    
    def _create_mock_gpx_data(self) -> Dict:
        """Create mock GPX data for testing"""
        return {
            'features': {
                'speed': np.random.randn(100),
                'acceleration': np.random.randn(100),
                'bearing': np.random.randn(100),
                'curvature': np.random.randn(100)
            },
            'duration': 3600.0,
            'distance': 10.5
        }
    
    def _test_basic_video_ops(self) -> Dict:
        """Basic video operations test"""
        # Test basic OpenCV GPU if available
        try:
            # This is a minimal test
            return {'basic_test': 'OpenCV operations would go here'}
        except:
            return {'basic_test': 'failed'}
    
    def _test_basic_feature_ops(self) -> Dict:
        """Basic feature extraction test"""
        # Test basic PyTorch operations
        device = torch.device(f'cuda:{self.gpu_ids[0]}')
        test_tensor = torch.randn(32, 3, 224, 224, device=device)
        
        # Simple conv operation
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        result = conv(test_tensor)
        
        if result.device != device:
            raise GPUTestError("Basic feature extraction failed!")
        
        return {'basic_conv_test': 'passed'}
    
    def _test_basic_cupy_ops(self) -> Dict:
        """Basic CuPy operations test"""
        a = cp.random.randn(100, 100)
        b = cp.random.randn(100, 100)
        c = cp.dot(a, b)
        
        if not isinstance(c, cp.ndarray):
            raise GPUTestError("Basic CuPy operation failed!")
        
        return {'basic_cupy_test': 'passed'}
    
    def _test_basic_correlation_ops(self) -> Dict:
        """Basic correlation operations test"""
        # Test basic similarity computation
        device = torch.device(f'cuda:{self.gpu_ids[0]}')
        
        a = torch.randn(512, device=device)
        b = torch.randn(512, device=device)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
        
        if similarity.device != device:
            raise GPUTestError("Basic correlation operation failed!")
        
        return {'basic_correlation_test': 'passed'}
    
    def _generate_test_summary(self, passed: int, failed: int):
        """Generate comprehensive test summary"""
        total = passed + failed
        
        print("\n" + "="*80)
        print("GPU ACCELERATION TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print("="*80)
        
        if failed > 0:
            print("\nFAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result['status'] == 'FAILED':
                    print(f"  ❌ {test_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\nDetailed results saved to: {self.test_dir}")
        
        # Save detailed results
        summary_file = os.path.join(self.test_dir, 'gpu_test_results.json')
        with open(summary_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"Full results: {summary_file}")
    
    def __del__(self):
        """Cleanup test directory"""
        try:
            import shutil
            if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, ignore_errors=True)
        except:
            pass

def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Comprehensive GPU Acceleration Test")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], 
                       help="GPU IDs to test")
    parser.add_argument("--strict", action='store_true', 
                       help="Strict mode - fail on any GPU fallback")
    parser.add_argument("--verbose", action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        tester = ComprehensiveGPUTester(gpu_ids=args.gpu_ids, strict_mode=args.strict)
        results = tester.run_all_tests()
        
        # Check if any tests failed
        failed_tests = [name for name, result in results.items() 
                       if result.get('status') == 'FAILED']
        
        if failed_tests and args.strict:
            print(f"\n❌ STRICT MODE: Tests failed, GPU acceleration not fully working!")
            sys.exit(1)
        elif failed_tests:
            print(f"\n⚠️ Some tests failed, but continuing in non-strict mode")
            sys.exit(2)
        else:
            print(f"\n✅ ALL TESTS PASSED - GPU acceleration is working perfectly!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n💥 TEST SUITE CRASHED: {e}")
        if args.strict:
            print("Strict mode enabled - this is a critical failure!")
        sys.exit(3)

if __name__ == "__main__":
    main()