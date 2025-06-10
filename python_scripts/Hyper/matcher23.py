#!/usr/bin/env python3
"""
Production-Ready Ultra-Optimized Multi-GPU Video-GPX Correlation Script

Features:
- Strict GPU acceleration with comprehensive validation
- Multiple similarity computation methods for robust matching
- H.264 4:4:4 automatic conversion support
- Comprehensive feature validation and error handling
- Detailed reporting and analytics
- Production-quality logging and monitoring
- Memory-optimized processing for large datasets
- Temporal correlation analysis for improved accuracy

Author: AI Assistant
Version: 2.0.0
License: MIT

Usage:
    python video_gpx_correlator.py -d /path/to/data --gpu_ids 0 1
    python video_gpx_correlator.py -d /path/to/data --force --debug
    python video_gpx_correlator.py --test_system  # Test GPU setup
"""

import cv2
import numpy as np
import gpxpy
import pandas as pd
import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from scipy.signal import find_peaks
from datetime import timedelta, datetime
import argparse
import os
import glob
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict, deque
import time
import warnings
import logging
from tqdm import tqdm
import gc
import asyncio
import aiofiles
from threading import Lock
import queue
import tempfile
import shutil
import sys

# Optional imports with fallbacks
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Setup production logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup comprehensive logging for production use"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    return logger

# Initialize logger
logger = setup_logging()

class ProductionGPUValidator:
    """Production-grade GPU validation and enforcement"""
    
    @staticmethod
    def validate_system_requirements():
        """Comprehensive system validation for production deployment"""
        validation_results = {
            'cuda_available': False,
            'cupy_available': False,
            'ffmpeg_gpu': False,
            'gpu_count': 0,
            'gpu_memory': [],
            'system_memory': 0,
            'issues': []
        }
        
        try:
            # CUDA validation
            if not torch.cuda.is_available():
                validation_results['issues'].append("CUDA not available")
            else:
                validation_results['cuda_available'] = True
                validation_results['gpu_count'] = torch.cuda.device_count()
                
                # Check GPU memory for each device
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    validation_results['gpu_memory'].append({
                        'device': i,
                        'name': props.name,
                        'memory_gb': memory_gb
                    })
                    
                    if memory_gb < 4:  # Minimum 4GB recommended
                        validation_results['issues'].append(f"GPU {i} has only {memory_gb:.1f}GB memory")
        
        except Exception as e:
            validation_results['issues'].append(f"CUDA validation failed: {e}")
        
        try:
            # CuPy validation
            if not cp.cuda.is_available():
                validation_results['issues'].append("CuPy CUDA not available")
            else:
                validation_results['cupy_available'] = True
                # Test basic CuPy operations
                test_array = cp.random.randn(100, 100)
                result = cp.dot(test_array, test_array)
                del test_array, result
                cp.get_default_memory_pool().free_all_blocks()
        
        except Exception as e:
            validation_results['issues'].append(f"CuPy validation failed: {e}")
        
        try:
            # FFmpeg GPU validation
            result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                  capture_output=True, text=True, check=True, timeout=10)
            if 'cuda' in result.stdout:
                validation_results['ffmpeg_gpu'] = True
            else:
                validation_results['issues'].append("FFmpeg CUDA support not found")
        
        except Exception as e:
            validation_results['issues'].append(f"FFmpeg validation failed: {e}")
        
        # System memory check
        if PSUTIL_AVAILABLE:
            try:
                memory_info = psutil.virtual_memory()
                validation_results['system_memory'] = memory_info.total / (1024**3)
                
                if validation_results['system_memory'] < 16:  # Minimum 16GB recommended
                    validation_results['issues'].append(f"System has only {validation_results['system_memory']:.1f}GB RAM")
            
            except Exception as e:
                validation_results['issues'].append(f"Memory validation failed: {e}")
        
        return validation_results
    
    @staticmethod
    def enforce_gpu_requirements(gpu_ids):
        """Enforce GPU requirements for production use"""
        validation = ProductionGPUValidator.validate_system_requirements()
        
        if not validation['cuda_available']:
            raise RuntimeError("PRODUCTION ERROR: CUDA is required but not available")
        
        if not validation['cupy_available']:
            raise RuntimeError("PRODUCTION ERROR: CuPy CUDA is required but not available")
        
        if not validation['ffmpeg_gpu']:
            raise RuntimeError("PRODUCTION ERROR: FFmpeg GPU support is required but not available")
        
        # Validate requested GPU IDs
        available_gpus = validation['gpu_count']
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise RuntimeError(f"PRODUCTION ERROR: GPU {gpu_id} requested but only {available_gpus} GPUs available")
        
        # Log system info
        logger.info("=== PRODUCTION GPU VALIDATION PASSED ===")
        logger.info(f"CUDA Available: ✓")
        logger.info(f"CuPy Available: ✓")
        logger.info(f"FFmpeg GPU: ✓")
        logger.info(f"Available GPUs: {available_gpus}")
        logger.info(f"System Memory: {validation['system_memory']:.1f}GB")
        
        for gpu_info in validation['gpu_memory']:
            logger.info(f"GPU {gpu_info['device']}: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
        
        if validation['issues']:
            logger.warning("System issues detected:")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
        
        return validation

class ProductionFeatureValidator:
    """Production-grade feature validation with comprehensive diagnostics"""
    
    @staticmethod
    def validate_video_features(features, video_path):
        """Comprehensive video feature validation"""
        if not isinstance(features, dict):
            return False, [f"Features must be dict, got {type(features)}"]
        
        if not features:
            return False, ["Features dictionary is empty"]
        
        issues = []
        valid_features = 0
        total_features = 0
        
        # Required features for video
        required_features = {
            'scene_features': {'type': np.ndarray, 'min_size': 1},
            'motion_magnitude': {'type': np.ndarray, 'min_size': 1},
            'color_variance': {'type': np.ndarray, 'min_size': 1}
        }
        
        # Check required features
        for feature_name, requirements in required_features.items():
            total_features += 1
            
            if feature_name not in features:
                issues.append(f"Missing required feature: {feature_name}")
                continue
            
            value = features[feature_name]
            
            if not isinstance(value, requirements['type']):
                issues.append(f"{feature_name}: Expected {requirements['type']}, got {type(value)}")
                continue
            
            if isinstance(value, np.ndarray):
                if value.size < requirements['min_size']:
                    issues.append(f"{feature_name}: Array too small (size: {value.size})")
                    continue
                
                if np.all(value == 0):
                    issues.append(f"{feature_name}: All values are zero")
                    continue
                
                if not np.isfinite(value).all():
                    nan_count = np.isnan(value).sum()
                    inf_count = np.isinf(value).sum()
                    issues.append(f"{feature_name}: Contains {nan_count} NaN and {inf_count} Inf values")
                    continue
                
                valid_features += 1
                logger.debug(f"{feature_name}: ✓ (shape={value.shape}, mean={np.mean(value):.4f})")
        
        # Check optional features
        optional_features = ['motion_features', 'texture_features', 'edge_density', 
                           'temporal_gradient', 'color_histograms']
        
        for feature_name in optional_features:
            if feature_name in features:
                total_features += 1
                value = features[feature_name]
                
                if isinstance(value, np.ndarray) and value.size > 0:
                    if np.isfinite(value).all() and not np.all(value == 0):
                        valid_features += 1
                        logger.debug(f"{feature_name}: ✓ (optional)")
        
        # Overall validation
        if valid_features == 0:
            issues.append("No valid features found")
            return False, issues
        
        success_rate = valid_features / total_features
        if success_rate < 0.5:  # At least 50% features should be valid
            issues.append(f"Low feature success rate: {success_rate:.1%}")
        
        logger.debug(f"Video {Path(video_path).name}: {valid_features}/{total_features} features valid")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_gpx_features(features, gpx_path):
        """Comprehensive GPX feature validation"""
        if not isinstance(features, dict):
            return False, [f"Features must be dict, got {type(features)}"]
        
        if not features:
            return False, ["Features dictionary is empty"]
        
        issues = []
        valid_features = 0
        total_features = 0
        
        # Required motion features
        required_features = {
            'speed': {'type': np.ndarray, 'min_size': 2},
            'bearing_change': {'type': np.ndarray, 'min_size': 1}
        }
        
        for feature_name, requirements in required_features.items():
            total_features += 1
            
            if feature_name not in features:
                issues.append(f"Missing required feature: {feature_name}")
                continue
            
            value = features[feature_name]
            
            if not isinstance(value, requirements['type']):
                issues.append(f"{feature_name}: Expected {requirements['type']}, got {type(value)}")
                continue
            
            if value.size < requirements['min_size']:
                issues.append(f"{feature_name}: Array too small (size: {value.size})")
                continue
            
            if np.all(value == 0):
                issues.append(f"{feature_name}: All values are zero")
                continue
            
            if not np.isfinite(value).all():
                issues.append(f"{feature_name}: Contains invalid values")
                continue
            
            valid_features += 1
            logger.debug(f"{feature_name}: ✓ (shape={value.shape}, mean={np.mean(value):.4f})")
        
        # Check statistical features
        stat_features = ['speed_stats', 'bearing_stats', 'elevation_stats', 'distance_stats']
        for feature_name in stat_features:
            if feature_name in features:
                total_features += 1
                value = features[feature_name]
                
                if isinstance(value, np.ndarray) and value.size > 0:
                    if np.isfinite(value).all():
                        valid_features += 1
        
        if valid_features == 0:
            issues.append("No valid features found")
            return False, issues
        
        logger.debug(f"GPX {Path(gpx_path).name}: {valid_features}/{total_features} features valid")
        
        return len(issues) == 0, issues

class ProductionSimilarityEngine:
    """Production-grade similarity computation engine"""
    
    def __init__(self):
        self.weights = {
            'motion_dynamics': 0.35,
            'statistical_profile': 0.25,
            'temporal_patterns': 0.25,
            'feature_correlation': 0.15
        }
        
        # Similarity thresholds for quality assessment
        self.thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def compute_comprehensive_similarity(self, video_features, gpx_features, video_path=None, gpx_path=None):
        """Compute multi-faceted similarity with production-grade error handling"""
        
        try:
            similarities = {}
            
            # 1. Motion Dynamics Similarity
            similarities['motion_dynamics'] = self._analyze_motion_dynamics(video_features, gpx_features)
            
            # 2. Statistical Profile Similarity  
            similarities['statistical_profile'] = self._analyze_statistical_profile(video_features, gpx_features)
            
            # 3. Temporal Pattern Similarity
            similarities['temporal_patterns'] = self._analyze_temporal_patterns(video_features, gpx_features)
            
            # 4. Feature Correlation Similarity
            similarities['feature_correlation'] = self._analyze_feature_correlation(video_features, gpx_features)
            
            # Compute weighted combined score
            combined_score = sum(
                similarities[key] * self.weights[key] 
                for key in similarities.keys()
            )
            
            similarities['combined'] = float(np.clip(combined_score, 0.0, 1.0))
            
            # Quality assessment
            similarities['quality'] = self._assess_quality(similarities['combined'])
            
            # Log detailed results for debugging
            if similarities['combined'] > 0.1:  # Only log meaningful similarities
                logger.debug(f"Similarity computed: {similarities['combined']:.3f} "
                           f"(motion: {similarities['motion_dynamics']:.3f}, "
                           f"stats: {similarities['statistical_profile']:.3f}, "
                           f"temporal: {similarities['temporal_patterns']:.3f}, "
                           f"features: {similarities['feature_correlation']:.3f})")
            
            return similarities
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return {
                'motion_dynamics': 0.0,
                'statistical_profile': 0.0,
                'temporal_patterns': 0.0,
                'feature_correlation': 0.0,
                'combined': 0.0,
                'quality': 'failed'
            }
    
    def _analyze_motion_dynamics(self, video_features, gpx_features):
        """Analyze motion dynamics between video and GPX"""
        try:
            video_motion = self._extract_motion_signature(video_features, 'video')
            gpx_motion = self._extract_motion_signature(gpx_features, 'gpx')
            
            if video_motion is None or gpx_motion is None:
                return 0.0
            
            # Normalize to same length
            min_len = min(len(video_motion), len(gpx_motion))
            if min_len < 3:  # Need minimum data for reliable comparison
                return 0.0
            
            video_motion = video_motion[:min_len]
            gpx_motion = gpx_motion[:min_len]
            
            # Normalize features to [0, 1] range
            video_motion = self._robust_normalize(video_motion)
            gpx_motion = self._robust_normalize(gpx_motion)
            
            # Compute multiple similarity metrics
            correlation = self._safe_correlation(video_motion, gpx_motion)
            cosine_sim = self._safe_cosine_similarity(video_motion, gpx_motion)
            euclidean_sim = self._safe_euclidean_similarity(video_motion, gpx_motion)
            
            # Weighted combination
            motion_similarity = 0.5 * correlation + 0.3 * cosine_sim + 0.2 * euclidean_sim
            
            return float(np.clip(motion_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Motion dynamics analysis failed: {e}")
            return 0.0
    
    def _extract_motion_signature(self, features, source_type):
        """Extract motion signature from features"""
        signature = []
        
        try:
            if source_type == 'video':
                # Video motion features
                motion_keys = ['motion_magnitude', 'acceleration', 'jerk', 'rotation']
                for key in motion_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                signature.extend([
                                    np.mean(values),
                                    np.std(values),
                                    np.max(values) if values.size > 0 else 0
                                ])
                
                # Edge dynamics (proxy for motion)
                if 'edge_density' in features:
                    values = features['edge_density']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        if np.isfinite(values).all():
                            signature.extend([
                                np.mean(values),
                                np.std(values)
                            ])
                            
            elif source_type == 'gpx':
                # GPX motion features
                motion_keys = ['speed', 'acceleration', 'jerk', 'bearing_change', 'curvature']
                for key in motion_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                signature.extend([
                                    np.mean(values),
                                    np.std(values),
                                    np.max(values) if values.size > 0 else 0
                                ])
            
            return np.array(signature) if signature else None
            
        except Exception as e:
            logger.debug(f"Motion signature extraction failed for {source_type}: {e}")
            return None
    
    def _analyze_statistical_profile(self, video_features, gpx_features):
        """Analyze statistical profiles"""
        try:
            video_stats = self._extract_statistical_profile(video_features, 'video')
            gpx_stats = self._extract_statistical_profile(gpx_features, 'gpx')
            
            if video_stats is None or gpx_stats is None:
                return 0.0
            
            # Ensure same length
            min_len = min(len(video_stats), len(gpx_stats))
            if min_len < 2:
                return 0.0
                
            video_stats = video_stats[:min_len]
            gpx_stats = gpx_stats[:min_len]
            
            # Normalize
            video_stats = self._robust_normalize(video_stats)
            gpx_stats = self._robust_normalize(gpx_stats)
            
            # Compute similarity
            distance = np.linalg.norm(video_stats - gpx_stats)
            similarity = 1.0 / (1.0 + distance)
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Statistical profile analysis failed: {e}")
            return 0.0
    
    def _extract_statistical_profile(self, features, source_type):
        """Extract statistical profile from features"""
        profile = []
        
        try:
            if source_type == 'video':
                # Video statistical features
                for key in ['motion_magnitude', 'color_variance', 'edge_density']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile.extend([
                                    np.mean(values),
                                    np.std(values),
                                    np.median(values)
                                ])
                                
            elif source_type == 'gpx':
                # GPX statistical features  
                for key in ['speed', 'bearing_change', 'elevation_change_rate']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile.extend([
                                    np.mean(values),
                                    np.std(values),
                                    np.median(values)
                                ])
                
                # Include pre-computed stats
                for key in ['speed_stats', 'bearing_stats']:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                profile.extend(values.flatten()[:4])  # Limit size
            
            return np.array(profile) if profile else None
            
        except Exception as e:
            logger.debug(f"Statistical profile extraction failed for {source_type}: {e}")
            return None
    
    def _analyze_temporal_patterns(self, video_features, gpx_features):
        """Analyze temporal patterns"""
        try:
            video_temporal = self._extract_temporal_signature(video_features, 'video')
            gpx_temporal = self._extract_temporal_signature(gpx_features, 'gpx')
            
            if video_temporal is None or gpx_temporal is None:
                return 0.0
            
            # Cross-correlation analysis
            correlation = self._safe_correlation(video_temporal, gpx_temporal)
            
            return float(np.clip((correlation + 1) / 2, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Temporal pattern analysis failed: {e}")
            return 0.0
    
    def _extract_temporal_signature(self, features, source_type):
        """Extract temporal signature"""
        try:
            if source_type == 'video':
                if 'temporal_gradient' in features:
                    values = features['temporal_gradient']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return values
                
                # Fallback: use motion magnitude temporal pattern
                if 'motion_magnitude' in features:
                    values = features['motion_magnitude']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return np.diff(values)  # Temporal changes
                            
            elif source_type == 'gpx':
                if 'speed' in features:
                    values = features['speed']
                    if isinstance(values, np.ndarray) and values.size > 5:
                        if np.isfinite(values).all():
                            return np.diff(values)  # Speed changes over time
            
            return None
            
        except Exception as e:
            logger.debug(f"Temporal signature extraction failed for {source_type}: {e}")
            return None
    
    def _analyze_feature_correlation(self, video_features, gpx_features):
        """Analyze direct feature correlation"""
        try:
            video_vector = self._create_robust_feature_vector(video_features, 'video')
            gpx_vector = self._create_robust_feature_vector(gpx_features, 'gpx')
            
            if video_vector is None or gpx_vector is None:
                return 0.0
            
            # Ensure same length
            min_len = min(len(video_vector), len(gpx_vector))
            if min_len < 3:
                return 0.0
                
            video_vector = video_vector[:min_len]
            gpx_vector = gpx_vector[:min_len]
            
            # Normalize
            video_vector = self._robust_normalize(video_vector)
            gpx_vector = self._robust_normalize(gpx_vector)
            
            # Cosine similarity
            similarity = self._safe_cosine_similarity(video_vector, gpx_vector)
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Feature correlation analysis failed: {e}")
            return 0.0
    
    def _create_robust_feature_vector(self, features, source_type):
        """Create robust feature vector with validation"""
        components = []
        
        try:
            if source_type == 'video':
                # Core video features
                feature_keys = ['motion_magnitude', 'color_variance', 'edge_density']
                for key in feature_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all() and not np.all(values == 0):
                                components.extend([
                                    np.mean(values),
                                    np.std(values),
                                    np.percentile(values, 25),
                                    np.percentile(values, 75)
                                ])
                
                # Scene features (if available)
                if 'scene_features' in features:
                    values = features['scene_features']
                    if isinstance(values, np.ndarray) and values.size > 0:
                        if values.ndim == 2:
                            values = np.mean(values, axis=0)
                        if np.isfinite(values).all():
                            components.extend(values.flatten()[:16])  # Limit size
                            
            elif source_type == 'gpx':
                # Core GPX features
                feature_keys = ['speed', 'acceleration', 'bearing_change', 'curvature']
                for key in feature_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all() and not np.all(values == 0):
                                components.extend([
                                    np.mean(values),
                                    np.std(values),
                                    np.percentile(values, 25),
                                    np.percentile(values, 75)
                                ])
                
                # Statistical summaries
                stat_keys = ['speed_stats', 'bearing_stats', 'elevation_stats']
                for key in stat_keys:
                    if key in features:
                        values = features[key]
                        if isinstance(values, np.ndarray) and values.size > 0:
                            if np.isfinite(values).all():
                                components.extend(values.flatten()[:6])
            
            if not components:
                return None
            
            vector = np.array(components)
            
            # Remove any remaining invalid values
            vector = vector[np.isfinite(vector)]
            
            return vector if len(vector) > 0 else None
            
        except Exception as e:
            logger.debug(f"Feature vector creation failed for {source_type}: {e}")
            return None
    
    def _robust_normalize(self, vector):
        """Robust normalization with outlier handling"""
        try:
            if len(vector) == 0:
                return vector
            
            # Remove outliers using IQR method
            q25, q75 = np.percentile(vector, [25, 75])
            iqr = q75 - q25
            
            if iqr > 1e-8:  # Avoid division by zero
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                vector = np.clip(vector, lower_bound, upper_bound)
            
            # Normalize to [0, 1]
            min_val, max_val = np.min(vector), np.max(vector)
            if max_val - min_val > 1e-8:
                vector = (vector - min_val) / (max_val - min_val)
            else:
                vector = np.zeros_like(vector)
            
            return vector
            
        except Exception as e:
            logger.debug(f"Normalization failed: {e}")
            return vector
    
    def _safe_correlation(self, x, y):
        """Safe correlation computation with error handling"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            correlation = np.corrcoef(x, y)[0, 1]
            
            if np.isnan(correlation) or np.isinf(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception:
            return 0.0
    
    def _safe_cosine_similarity(self, x, y):
        """Safe cosine similarity computation"""
        try:
            if len(x) != len(y) or len(x) == 0:
                return 0.0
            
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            
            if norm_x < 1e-8 or norm_y < 1e-8:
                return 0.0
            
            similarity = dot_product / (norm_x * norm_y)
            
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.0
            
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _safe_euclidean_similarity(self, x, y):
        """Safe Euclidean similarity computation"""
        try:
            if len(x) != len(y) or len(x) == 0:
                return 0.0
            
            distance = np.linalg.norm(x - y)
            similarity = 1.0 / (1.0 + distance)
            
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.0
            
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def _assess_quality(self, score):
        """Assess similarity quality"""
        if score >= self.thresholds['excellent']:
            return 'excellent'
        elif score >= self.thresholds['good']:
            return 'good'
        elif score >= self.thresholds['fair']:
            return 'fair'
        elif score >= self.thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'

class OptimizedFFmpegDecoder:
    """Production-optimized FFmpeg decoder with H.264 4:4:4 support"""
    
    def __init__(self, gpu_ids=[0, 1], conversion_settings=None):
        ProductionGPUValidator.enforce_gpu_requirements(gpu_ids)
        
        self.gpu_ids = gpu_ids
        self.temp_dirs = {}
        self.conversion_settings = conversion_settings or {
            'disable_444_conversion': False,
            'force_disk_conversion': False,
            'keep_444_backup': False
        }
        
        # Initialize temp directories
        for gpu_id in gpu_ids:
            self.temp_dirs[gpu_id] = tempfile.mkdtemp(prefix=f'production_gpu_{gpu_id}_')
        
        logger.info(f"Production FFmpeg decoder initialized for GPUs: {gpu_ids}")
    
    def decode_video_optimized(self, video_path, sample_rate=2.0, target_size=(640, 360), gpu_id=0):
        """Production-optimized video decoding with comprehensive error handling"""
        
        if gpu_id not in self.temp_dirs:
            raise ValueError(f"GPU {gpu_id} not initialized")
        
        try:
            # Check and convert H.264 4:4:4 if needed
            working_video_path = self._handle_h264_444_conversion(video_path, gpu_id)
            
            # Get video information
            video_info = self._get_video_info(working_video_path)
            if not video_info:
                raise RuntimeError(f"Could not get video info for {working_video_path}")
            
            # Adjust parameters based on video properties
            adjusted_params = self._adjust_decode_parameters(video_info, target_size, sample_rate)
            
            # Decode frames
            frames_tensor = self._decode_frames_gpu(
                working_video_path, 
                adjusted_params, 
                gpu_id
            )
            
            if frames_tensor is None:
                raise RuntimeError("Frame decoding failed")
            
            # Validate output
            if not self._validate_decoded_frames(frames_tensor):
                raise RuntimeError("Decoded frames validation failed")
            
            return frames_tensor, video_info['fps'], video_info['duration'], None
            
        except Exception as e:
            logger.error(f"Video decoding failed for {video_path}: {e}")
            raise
    
    def _handle_h264_444_conversion(self, video_path, gpu_id):
        """Handle H.264 4:4:4 conversion if needed"""
        if self.conversion_settings.get('disable_444_conversion', False):
            return video_path
        
        # Check if conversion is needed
        is_444, stream_info = self._check_h264_444_format(video_path)
        
        if not is_444:
            return video_path
        
        logger.info(f"Converting H.264 4:4:4 video: {video_path}")
        
        try:
            converted_path = self._convert_h264_444_to_420(video_path, gpu_id)
            logger.info(f"H.264 4:4:4 conversion successful: {converted_path}")
            return converted_path
            
        except Exception as e:
            logger.warning(f"H.264 4:4:4 conversion failed: {e}, using original")
            return video_path
    
    def _check_h264_444_format(self, video_path):
        """Check if video is H.264 4:4:4 format"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            info = json.loads(result.stdout)
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    codec_name = stream.get('codec_name', '').lower()
                    pix_fmt = stream.get('pix_fmt', '').lower()
                    
                    if codec_name == 'h264' and ('444' in pix_fmt or 'yuv444' in pix_fmt):
                        return True, stream
            
            return False, None
            
        except Exception as e:
            logger.debug(f"Format check failed for {video_path}: {e}")
            return False, None
    
    def _convert_h264_444_to_420(self, video_path, gpu_id):
        """Convert H.264 4:4:4 to 4:2:0 using GPU"""
        output_path = os.path.join(
            self.temp_dirs[gpu_id], 
            f"converted_{os.path.basename(video_path)}"
        )
        
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-hide_banner',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', video_path,
            '-c:v', 'h264_nvenc',
            '-pix_fmt', 'yuv420p',
            '-preset', 'p1',
            '-cq', '23',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
                raise RuntimeError("Conversion output invalid")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"H.264 4:4:4 conversion failed: {e}")
    
    def _get_video_info(self, video_path):
        """Get comprehensive video information"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_format', '-show_streams', video_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            info = json.loads(result.stdout)
            
            video_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            fps = eval(video_stream.get('r_frame_rate', '30/1'))
            duration = float(info.get('format', {}).get('duration', 0))
            width = int(video_stream.get('width', 1920))
            height = int(video_stream.get('height', 1080))
            
            return {
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height,
                'codec': video_stream.get('codec_name'),
                'pixel_format': video_stream.get('pix_fmt')
            }
            
        except Exception as e:
            logger.error(f"Could not get video info for {video_path}: {e}")
            return None
    
    def _adjust_decode_parameters(self, video_info, target_size, sample_rate):
        """Adjust decoding parameters based on video properties"""
        width, height = video_info['width'], video_info['height']
        
        # Adjust target size for high resolution videos
        if width >= 3840 or height >= 2160:  # 4K+
            target_size = (480, 270)
            sample_rate = min(sample_rate, 1.0)
        elif width >= 2560 or height >= 1440:  # 1440p
            target_size = (560, 315)
            sample_rate = min(sample_rate, 1.5)
        
        # Limit max frames for memory management
        max_frames = min(int(video_info['duration'] * sample_rate) + 10, 200)
        
        return {
            'target_size': target_size,
            'sample_rate': sample_rate,
            'max_frames': max_frames
        }
    
    def _decode_frames_gpu(self, video_path, params, gpu_id):
        """Decode frames using GPU acceleration"""
        temp_dir = self.temp_dirs[gpu_id]
        output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
        
        target_size = params['target_size']
        max_frames = params['max_frames']
        
        # Try GPU decoders
        gpu_decoders = ['h264_cuvid', 'hevc_cuvid']
        
        for decoder in gpu_decoders:
            try:
                cmd = [
                    'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                    '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                    '-c:v', decoder,
                    '-i', video_path,
                    '-vf', f'scale={target_size[0]}:{target_size[1]}',
                    '-frames:v', str(max_frames),
                    '-q:v', '2',  # High quality JPEG
                    output_pattern
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0:
                    # Success - load frames
                    return self._load_frames_to_tensor(temp_dir, target_size, gpu_id)
                    
            except Exception as e:
                logger.debug(f"GPU decoder {decoder} failed: {e}")
                continue
        
        # Fallback to software decode
        logger.warning(f"GPU decode failed, using software fallback for {video_path}")
        return self._decode_frames_software(video_path, params, gpu_id)
    
    def _decode_frames_software(self, video_path, params, gpu_id):
        """Software fallback for frame decoding"""
        temp_dir = self.temp_dirs[gpu_id]
        output_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
        
        target_size = params['target_size']
        max_frames = params['max_frames']
        
        try:
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-hide_banner',
                '-i', video_path,
                '-vf', f'scale={target_size[0]}:{target_size[1]}',
                '-frames:v', str(max_frames),
                '-q:v', '2',
                output_pattern
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
            return self._load_frames_to_tensor(temp_dir, target_size, gpu_id)
            
        except Exception as e:
            logger.error(f"Software decode also failed: {e}")
            return None
    
    def _load_frames_to_tensor(self, temp_dir, target_size, gpu_id):
        """Load decoded frames to GPU tensor"""
        frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.jpg')))
        
        if not frame_files:
            logger.error("No frames found after decoding")
            return None
        
        device = torch.device(f'cuda:{gpu_id}')
        frames = []
        
        for frame_file in frame_files:
            try:
                # Load image
                img = cv2.imread(frame_file)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor and move to GPU
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # CHW format
                img_tensor = img_tensor.to(device)
                
                frames.append(img_tensor)
                
                # Clean up frame file
                os.remove(frame_file)
                
            except Exception as e:
                logger.debug(f"Failed to load frame {frame_file}: {e}")
                continue
        
        if not frames:
            logger.error("No valid frames loaded")
            return None
        
        # Stack into batch tensor
        frames_tensor = torch.stack(frames).unsqueeze(0)  # (1, N, C, H, W)
        
        logger.debug(f"Loaded {len(frames)} frames to GPU tensor: {frames_tensor.shape}")
        
        return frames_tensor
    
    def _validate_decoded_frames(self, frames_tensor):
        """Validate decoded frames tensor"""
        if frames_tensor is None:
            return False
        
        if not isinstance(frames_tensor, torch.Tensor):
            return False
        
        if frames_tensor.device.type != 'cuda':
            return False
        
        if frames_tensor.numel() == 0:
            return False
        
        if torch.isnan(frames_tensor).any() or torch.isinf(frames_tensor).any():
            return False
        
        return True
    
    def cleanup(self):
        """Cleanup temporary directories"""
        for temp_dir in self.temp_dirs.values():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

class ProductionFeatureExtractor:
    """Production-grade feature extraction with comprehensive validation"""
    
    def __init__(self, gpu_ids=[0, 1]):
        ProductionGPUValidator.enforce_gpu_requirements(gpu_ids)
        
        self.gpu_ids = gpu_ids
        self.devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in gpu_ids]
        
        # Initialize feature extraction models
        self.feature_models = {}
        for device in self.devices:
            self.feature_models[device] = self._create_feature_extraction_model().to(device)
        
        logger.info(f"Production feature extractor initialized on {len(self.devices)} GPUs")
    
    def _create_feature_extraction_model(self):
        """Create optimized feature extraction model"""
        class ProductionFeatureNet(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Efficient MobileNet-style architecture
                self.features = nn.Sequential(
                    # Initial conv
                    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True),
                    
                    # Depthwise separable blocks
                    self._make_separable_block(32, 64, 2),
                    self._make_separable_block(64, 128, 2),
                    self._make_separable_block(128, 256, 2),
                    self._make_separable_block(256, 512, 1),
                    
                    # Global pooling
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                
                # Multiple heads for different feature types
                self.scene_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128)
                )
                
                self.motion_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64)
                )
                
                self.texture_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64)
                )
            
            def _make_separable_block(self, in_channels, out_channels, stride):
                return nn.Sequential(
                    # Depthwise
                    nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                             padding=1, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU6(inplace=True),
                    
                    # Pointwise
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(inplace=True)
                )
            
            def forward(self, x):
                features = self.features(x)
                return {
                    'scene_features': self.scene_head(features),
                    'motion_features': self.motion_head(features),
                    'texture_features': self.texture_head(features)
                }
        
        model = ProductionFeatureNet()
        model.eval()
        return model
    
    def extract_comprehensive_features(self, frames_tensor, device_idx=0):
        """Extract comprehensive features with production-grade validation"""
        
        if device_idx >= len(self.devices):
            raise ValueError(f"Invalid device index: {device_idx}")
        
        device = self.devices[device_idx]
        model = self.feature_models[device]
        
        try:
            # Validate input
            if not self._validate_input_tensor(frames_tensor):
                raise ValueError("Invalid input tensor")
            
            # Move to correct device if needed
            if frames_tensor.device != device:
                frames_tensor = frames_tensor.to(device)
            
            # Extract features
            features = self._extract_all_features(frames_tensor, model, device)
            
            # Validate output features
            validated_features = self._validate_extracted_features(features)
            
            # Convert to CPU for storage
            cpu_features = self._convert_features_to_cpu(validated_features)
            
            logger.debug(f"Feature extraction successful: {len(cpu_features)} feature types")
            
            return cpu_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _validate_input_tensor(self, tensor):
        """Validate input tensor"""
        if tensor is None:
            return False
        
        if not isinstance(tensor, torch.Tensor):
            return False
        
        if tensor.device.type != 'cuda':
            return False
        
        if tensor.dim() != 5:  # (B, N, C, H, W)
            return False
        
        if tensor.numel() == 0:
            return False
        
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        
        return True
    
    def _extract_all_features(self, frames_tensor, model, device):
        """Extract all types of features"""
        batch_size, num_frames = frames_tensor.shape[:2]
        
        # Reshape for batch processing
        frames_flat = frames_tensor.view(-1, *frames_tensor.shape[2:])  # (B*N, C, H, W)
        
        features = {}
        
        with torch.no_grad():
            # CNN features
            cnn_features = model(frames_flat)
            
            # Reshape back to sequence format
            for key, value in cnn_features.items():
                value = value.view(batch_size, num_frames, -1)[0]  # Remove batch dim
                features[key] = value
            
            # Motion analysis
            motion_features = self._compute_motion_features(frames_tensor[0], device)
            features.update(motion_features)
            
            # Color analysis
            color_features = self._compute_color_features(frames_tensor[0], device)
            features.update(color_features)
            
            # Temporal analysis
            temporal_features = self._compute_temporal_features(frames_tensor[0], device)
            features.update(temporal_features)
            
            # Edge analysis
            edge_features = self._compute_edge_features(frames_tensor[0], device)
            features.update(edge_features)
        
        return features
    
    def _compute_motion_features(self, frames, device):
        """Compute motion features"""
        num_frames = frames.shape[0]
        
        features = {
            'motion_magnitude': torch.zeros(num_frames, device=device),
            'motion_direction': torch.zeros(num_frames, device=device),
            'acceleration': torch.zeros(num_frames, device=device),
            'jerk': torch.zeros(num_frames, device=device)
        }
        
        if num_frames < 2:
            return features
        
        # Optical flow approximation using frame differences
        for i in range(num_frames - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Convert to grayscale
            gray1 = 0.299 * frame1[0] + 0.587 * frame1[1] + 0.114 * frame1[2]
            gray2 = 0.299 * frame2[0] + 0.587 * frame2[1] + 0.114 * frame2[2]
            
            # Frame difference
            diff = torch.abs(gray2 - gray1)
            
            # Motion magnitude
            features['motion_magnitude'][i + 1] = torch.mean(diff)
            
            # Gradient-based motion direction
            grad_x = torch.mean(torch.abs(diff[:, 1:] - diff[:, :-1]))
            grad_y = torch.mean(torch.abs(diff[1:, :] - diff[:-1, :]))
            features['motion_direction'][i + 1] = torch.atan2(grad_y, grad_x + 1e-8)
        
        # Compute acceleration and jerk
        motion_mag = features['motion_magnitude']
        if num_frames > 2:
            features['acceleration'][1:] = motion_mag[1:] - motion_mag[:-1]
        
        if num_frames > 3:
            accel = features['acceleration']
            features['jerk'][2:] = accel[2:] - accel[1:-1]
        
        return features
    
    def _compute_color_features(self, frames, device):
        """Compute color features"""
        features = {}
        
        # Color variance over time
        color_variance = torch.var(frames, dim=[2, 3])  # Variance per channel per frame
        features['color_variance'] = torch.mean(color_variance, dim=1)  # Average across channels
        
        # Color histograms (simplified)
        histograms = []
        for i in range(frames.shape[0]):
            frame = frames[i]
            
            # Quantize colors to reduce histogram size
            frame_quantized = (frame * 15).long()  # 16 bins per channel
            
            # Simple histogram approximation
            hist_r = torch.bincount(frame_quantized[0].flatten(), minlength=16)[:16]
            hist_g = torch.bincount(frame_quantized[1].flatten(), minlength=16)[:16]
            hist_b = torch.bincount(frame_quantized[2].flatten(), minlength=16)[:16]
            
            hist = torch.cat([hist_r, hist_g, hist_b]).float()
            hist = hist / torch.sum(hist)  # Normalize
            
            histograms.append(hist)
        
        features['color_histograms'] = torch.stack(histograms)
        
        return features
    
    def _compute_temporal_features(self, frames, device):
        """Compute temporal features"""
        if frames.shape[0] < 2:
            return {
                'temporal_gradient': torch.zeros(frames.shape[0], device=device),
                'temporal_stability': torch.zeros(frames.shape[0], device=device)
            }
        
        # Convert to grayscale
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        # Temporal gradients
        temporal_diff = torch.abs(gray_frames[1:] - gray_frames[:-1])
        temporal_gradient = torch.mean(temporal_diff, dim=[1, 2])
        temporal_gradient = torch.cat([torch.zeros(1, device=device), temporal_gradient])
        
        # Temporal stability
        stability = torch.zeros(frames.shape[0], device=device)
        window_size = 5
        
        for i in range(window_size, frames.shape[0]):
            window_frames = gray_frames[i-window_size:i+1]
            stability[i] = 1.0 / (1.0 + torch.var(window_frames))
        
        return {
            'temporal_gradient': temporal_gradient,
            'temporal_stability': stability
        }
    
    def _compute_edge_features(self, frames, device):
        """Compute edge features"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        
        # Convert to grayscale
        gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray_frames = gray_frames.unsqueeze(1)  # Add channel dimension
        
        # Edge detection
        edges_x = F.conv2d(gray_frames, sobel_x, padding=1)
        edges_y = F.conv2d(gray_frames, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[1, 2, 3])
        
        # Edge orientation
        edge_angle = torch.atan2(edges_y, edges_x + 1e-8)
        edge_orientation = torch.mean(torch.abs(torch.cos(edge_angle)), dim=[1, 2, 3])
        
        return {
            'edge_density': edge_density,
            'edge_orientation': edge_orientation
        }
    
    def _validate_extracted_features(self, features):
        """Validate extracted features"""
        validated = {}
        
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                # Check for invalid values
                if torch.isnan(value).any() or torch.isinf(value).any():
                    logger.warning(f"Feature {key} contains invalid values, cleaning...")
                    value = torch.nan_to_num(value, 0.0)
                
                # Check for empty tensors
                if value.numel() == 0:
                    logger.warning(f"Feature {key} is empty, skipping...")
                    continue
                
                validated[key] = value
            else:
                validated[key] = value
        
        return validated
    
    def _convert_features_to_cpu(self, features):
        """Convert features to CPU for storage"""
        cpu_features = {}
        
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                cpu_features[key] = value.cpu().numpy()
            else:
                cpu_features[key] = value
        
        return cpu_features

class ProductionGPXProcessor:
    """Production-grade GPX processing with CuPy acceleration"""
    
    def __init__(self):
        # Validate CuPy availability
        if not cp.cuda.is_available():
            raise RuntimeError("PRODUCTION ERROR: CuPy CUDA required for GPX processing")
        
        # Configure memory pools
        try:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=None)
        except Exception as e:
            logger.warning(f"Could not configure CuPy memory pool: {e}")
        
        logger.info("Production GPX processor initialized with CuPy acceleration")
    
    def process_gpx_files(self, gpx_paths, max_workers=None):
        """Process GPX files with production-grade error handling"""
        
        if max_workers is None:
            max_workers = min(16, mp.cpu_count())
        
        logger.info(f"Processing {len(gpx_paths)} GPX files with {max_workers} workers")
        
        # Parse GPX files in parallel
        raw_data = {}
        failed_files = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._parse_gpx_file, path): path for path in gpx_paths}
            
            for future in tqdm(as_completed(futures), total=len(gpx_paths), desc="Parsing GPX"):
                path = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        raw_data[path] = data
                    else:
                        failed_files.append(path)
                        
                except Exception as e:
                    logger.error(f"Error parsing {path}: {e}")
                    failed_files.append(path)
        
        logger.info(f"Successfully parsed {len(raw_data)}/{len(gpx_paths)} GPX files")
        
        if failed_files:
            logger.warning(f"Failed to parse {len(failed_files)} GPX files")
        
        # Compute features on GPU
        results = {}
        for path, data in tqdm(raw_data.items(), desc="Computing GPX features"):
            try:
                enhanced_data = self._compute_comprehensive_features(data)
                results[path] = enhanced_data
                
            except Exception as e:
                logger.error(f"Feature computation failed for {path}: {e}")
                results[path] = None
        
        valid_results = sum(1 for r in results.values() if r is not None)
        logger.info(f"Successfully computed features for {valid_results}/{len(results)} GPX files")
        
        return results
    
    def _parse_gpx_file(self, gpx_path):
        """Parse single GPX file with comprehensive validation"""
        try:
            with open(gpx_path, 'r', encoding='utf-8', errors='ignore') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time is not None:
                            points.append({
                                'timestamp': point.time.replace(tzinfo=None),
                                'lat': float(point.latitude),
                                'lon': float(point.longitude),
                                'elevation': float(point.elevation or 0)
                            })
            
            if len(points) < 10:  # Minimum points required
                logger.debug(f"GPX file {gpx_path} has insufficient points: {len(points)}")
                return None
            
            # Create DataFrame and validate
            df = pd.DataFrame(points)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic validation
            if df['lat'].isna().any() or df['lon'].isna().any():
                logger.debug(f"GPX file {gpx_path} has invalid coordinates")
                return None
            
            # Check for reasonable coordinate ranges
            if not (-90 <= df['lat'].min() <= df['lat'].max() <= 90):
                logger.debug(f"GPX file {gpx_path} has invalid latitude range")
                return None
            
            if not (-180 <= df['lon'].min() <= df['lon'].max() <= 180):
                logger.debug(f"GPX file {gpx_path} has invalid longitude range")
                return None
            
            return {
                'df': df,
                'start_time': df['timestamp'].iloc[0],
                'end_time': df['timestamp'].iloc[-1],
                'point_count': len(df),
                'bounds': {
                    'lat_min': df['lat'].min(),
                    'lat_max': df['lat'].max(),
                    'lon_min': df['lon'].min(),
                    'lon_max': df['lon'].max()
                }
            }
            
        except Exception as e:
            logger.debug(f"Error parsing GPX file {gpx_path}: {e}")
            return None
    
    def _compute_comprehensive_features(self, gpx_data):
        """Compute comprehensive features using CuPy GPU acceleration"""
        df = gpx_data['df']
        n_points = len(df)
        
        # Transfer core data to GPU
        lats_gpu = cp.array(df['lat'].values, dtype=cp.float64)
        lons_gpu = cp.array(df['lon'].values, dtype=cp.float64)
        elevs_gpu = cp.array(df['elevation'].values, dtype=cp.float64)
        
        # Compute time differences
        time_diffs = self._compute_time_differences(df['timestamp'].values)
        time_diffs_gpu = cp.array(time_diffs, dtype=cp.float64)
        
        # Compute distances using Haversine formula
        distances_gpu = self._compute_distances_gpu(lats_gpu, lons_gpu)
        
        # Compute comprehensive motion features
        motion_features = self._compute_motion_features_gpu(
            lats_gpu, lons_gpu, elevs_gpu, time_diffs_gpu, distances_gpu
        )
        
        # Compute statistical features
        statistical_features = self._compute_statistical_features_gpu(
            motion_features, distances_gpu, elevs_gpu
        )
        
        # Apply smoothing
        smoothed_features = self._apply_smoothing_gpu(motion_features)
        
        # Combine all features
        all_features = {**smoothed_features, **statistical_features}
        
        # Convert to CPU
        cpu_features = {
            key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
            for key, value in all_features.items()
        }
        
        # Add metadata
        duration = self._compute_duration(df['timestamp'])
        total_distance = float(cp.sum(distances_gpu))
        
        gpx_data.update({
            'features': cpu_features,
            'duration': duration,
            'distance': total_distance,
            'max_speed': float(cp.max(motion_features['speed'])),
            'avg_speed': float(cp.mean(motion_features['speed']))
        })
        
        return gpx_data
    
    def _compute_time_differences(self, timestamps):
        """Compute time differences in seconds"""
        time_diffs = [1.0]  # First point gets 1 second
        
        for i in range(1, len(timestamps)):
            try:
                time_diff = timestamps[i] - timestamps[i-1]
                
                if hasattr(time_diff, 'total_seconds'):
                    seconds = time_diff.total_seconds()
                elif isinstance(time_diff, np.timedelta64):
                    seconds = float(time_diff / np.timedelta64(1, 's'))
                else:
                    seconds = float(time_diff)
                
                # Ensure positive and reasonable
                if seconds <= 0 or seconds > 3600:  # Max 1 hour between points
                    seconds = 1.0
                
                time_diffs.append(seconds)
                
            except Exception:
                time_diffs.append(1.0)  # Fallback
        
        return time_diffs
    
    def _compute_distances_gpu(self, lats, lons):
        """Compute distances using Haversine formula on GPU"""
        if len(lats) < 2:
            return cp.array([0.0])
        
        R = 3958.8  # Earth radius in miles
        
        # Convert to radians
        lat1_rad = cp.radians(lats[:-1])
        lon1_rad = cp.radians(lons[:-1])
        lat2_rad = cp.radians(lats[1:])
        lon2_rad = cp.radians(lons[1:])
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))  # Clip to avoid numerical issues
        
        distances = R * c
        
        # Prepend 0 for first point
        return cp.concatenate([cp.array([0.0]), distances])
    
    def _compute_motion_features_gpu(self, lats, lons, elevs, time_diffs, distances):
        """Compute comprehensive motion features on GPU"""
        n = len(lats)
        
        # Initialize features
        features = {
            'speed': cp.zeros(n),
            'acceleration': cp.zeros(n),
            'jerk': cp.zeros(n),
            'bearing': cp.zeros(n),
            'bearing_change': cp.zeros(n),
            'curvature': cp.zeros(n),
            'elevation_change_rate': cp.zeros(n),
            'turning_rate': cp.zeros(n)
        }
        
        if n < 2:
            return features
        
        # Speed computation
        speed_segments = distances[1:] * 3600 / cp.maximum(time_diffs[1:], 1e-6)  # mph
        features['speed'] = cp.concatenate([cp.array([0.0]), speed_segments])
        
        # Acceleration
        if n > 2:
            speed_diffs = features['speed'][1:] - features['speed'][:-1]
            accel_segments = speed_diffs / cp.maximum(time_diffs[1:], 1e-6)
            features['acceleration'] = cp.concatenate([cp.array([0.0]), accel_segments])
        
        # Jerk
        if n > 3:
            accel_diffs = features['acceleration'][1:] - features['acceleration'][:-1]
            jerk_segments = accel_diffs / cp.maximum(time_diffs[2:], 1e-6)
            features['jerk'] = cp.concatenate([cp.array([0.0, 0.0]), jerk_segments])
        
        # Bearings
        if n > 1:
            bearings = self._compute_bearings_gpu(lats[:-1], lons[:-1], lats[1:], lons[1:])
            features['bearing'] = cp.concatenate([cp.array([0.0]), bearings])
        
        # Bearing changes and curvature
        if n > 2:
            bearing_diffs = cp.diff(features['bearing'])
            # Handle angle wraparound
            bearing_diffs = cp.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
            bearing_diffs = cp.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)
            bearing_changes = cp.abs(bearing_diffs)
            
            features['bearing_change'] = cp.concatenate([cp.array([0.0, 0.0]), bearing_changes])
            
            # Curvature (bearing change per distance)
            curvature_segments = bearing_changes / cp.maximum(distances[2:], 1e-8)
            features['curvature'] = cp.concatenate([cp.array([0.0, 0.0]), curvature_segments])
            
            # Turning rate (bearing change per time)
            turning_segments = bearing_changes / cp.maximum(time_diffs[2:], 1e-6)
            features['turning_rate'] = cp.concatenate([cp.array([0.0, 0.0]), turning_segments])
        
        # Elevation change rate
        if n > 1:
            elev_diffs = elevs[1:] - elevs[:-1]
            elev_rate_segments = elev_diffs / cp.maximum(time_diffs[1:], 1e-6)
            features['elevation_change_rate'] = cp.concatenate([cp.array([0.0]), elev_rate_segments])
        
        return features
    
    def _compute_bearings_gpu(self, lat1, lon1, lat2, lon2):
        """Compute bearings between points on GPU"""
        lat1_rad = cp.radians(lat1)
        lon1_rad = cp.radians(lon1)
        lat2_rad = cp.radians(lat2)
        lon2_rad = cp.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = cp.sin(dlon) * cp.cos(lat2_rad)
        x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(dlon)
        
        bearings = cp.degrees(cp.arctan2(y, x))
        # Normalize to 0-360
        bearings = cp.where(bearings < 0, bearings + 360, bearings)
        
        return bearings
    
    def _compute_statistical_features_gpu(self, motion_features, distances, elevations):
        """Compute statistical summary features on GPU"""
        features = {}
        
        # Speed statistics
        speed = motion_features['speed']
        features['speed_stats'] = cp.array([
            cp.mean(speed), cp.std(speed), cp.min(speed), cp.max(speed),
            cp.percentile(speed, 25), cp.percentile(speed, 50), cp.percentile(speed, 75)
        ])
        
        # Bearing statistics
        bearing = motion_features['bearing']
        features['bearing_stats'] = cp.array([
            cp.mean(bearing), cp.std(bearing), cp.min(bearing), cp.max(bearing)
        ])
        
        # Elevation statistics
        total_climb = cp.sum(cp.maximum(cp.diff(elevations), 0))
        total_descent = cp.sum(cp.maximum(-cp.diff(elevations), 0))
        
        features['elevation_stats'] = cp.array([
            cp.mean(elevations), cp.std(elevations), cp.min(elevations), cp.max(elevations),
            total_climb, total_descent
        ])
        
        # Distance statistics
        features['distance_stats'] = cp.array([
            cp.sum(distances), cp.mean(distances), cp.std(distances), cp.max(distances)
        ])
        
        return features
    
    def _apply_smoothing_gpu(self, features, window_size=5):
        """Apply smoothing to features on GPU"""
        smoothed = {}
        
        kernel = cp.ones(window_size) / window_size
        
        for key, values in features.items():
            if isinstance(values, cp.ndarray) and len(values) > window_size:
                # Pad signal
                padded = cp.pad(values, (window_size//2, window_size//2), mode='edge')
                # Convolve
                smoothed_values = cp.convolve(padded, kernel, mode='valid')
                smoothed[key] = smoothed_values
            else:
                smoothed[key] = values
        
        return smoothed
    
    def _compute_duration(self, timestamps):
        """Compute track duration in seconds"""
        try:
            start_time = timestamps.iloc[0]
            end_time = timestamps.iloc[-1]
            
            duration_delta = end_time - start_time
            
            if hasattr(duration_delta, 'total_seconds'):
                return duration_delta.total_seconds()
            else:
                return float(duration_delta / np.timedelta64(1, 's'))
                
        except Exception as e:
            logger.debug(f"Duration computation failed: {e}")
            return 3600.0  # Default 1 hour

class ProductionCorrelator:
    """Production-grade correlation engine"""
    
    def __init__(self, gpu_ids=[0, 1]):
        ProductionGPUValidator.enforce_gpu_requirements(gpu_ids)
        
        self.gpu_ids = gpu_ids
        self.similarity_engine = ProductionSimilarityEngine()
        self.validator = ProductionFeatureValidator()
        
        logger.info(f"Production correlator initialized for GPUs: {gpu_ids}")
    
    def correlate_comprehensive(self, video_features_dict, gpx_database, output_dir, top_k=5):
        """Perform comprehensive correlation with production-grade validation"""
        
        logger.info("Starting comprehensive video-GPX correlation...")
        
        # Validate and filter features
        valid_videos, valid_gpx = self._validate_all_features(video_features_dict, gpx_database)
        
        if not valid_videos or not valid_gpx:
            raise RuntimeError("No valid features available for correlation")
        
        logger.info(f"Correlating {len(valid_videos)} videos with {len(valid_gpx)} GPX tracks")
        
        # Perform correlation
        results = {}
        total_comparisons = len(valid_videos) * len(valid_gpx)
        
        with tqdm(total=total_comparisons, desc="Computing correlations") as pbar:
            for video_path, video_features in valid_videos.items():
                try:
                    matches = self._find_best_matches(
                        video_path, video_features, valid_gpx, top_k, pbar
                    )
                    
                    results[video_path] = {'matches': matches}
                    
                    # Log best match
                    if matches:
                        best = matches[0]
                        logger.info(f"Best match for {Path(video_path).name}: "
                                  f"{Path(best['path']).name} "
                                  f"(score: {best['combined_score']:.3f}, "
                                  f"quality: {best.get('quality', 'unknown')})")
                
                except Exception as e:
                    logger.error(f"Correlation failed for {video_path}: {e}")
                    results[video_path] = None
        
        # Generate comprehensive report
        asyncio.run(self._generate_production_report(results, output_dir, valid_videos, valid_gpx))
        
        return results
    
    def _validate_all_features(self, video_features_dict, gpx_database):
        """Validate all features for production use"""
        
        logger.info("Validating video and GPX features...")
        
        valid_videos = {}
        invalid_video_count = 0
        
        for video_path, features in video_features_dict.items():
            if features is None:
                invalid_video_count += 1
                continue
            
            is_valid, issues = self.validator.validate_video_features(features, video_path)
            
            if is_valid:
                valid_videos[video_path] = features
            else:
                invalid_video_count += 1
                logger.debug(f"Invalid video features for {Path(video_path).name}: {issues}")
        
        valid_gpx = {}
        invalid_gpx_count = 0
        
        for gpx_path, gpx_data in gpx_database.items():
            if gpx_data is None or 'features' not in gpx_data:
                invalid_gpx_count += 1
                continue
            
            is_valid, issues = self.validator.validate_gpx_features(gpx_data['features'], gpx_path)
            
            if is_valid:
                valid_gpx[gpx_path] = gpx_data
            else:
                invalid_gpx_count += 1
                logger.debug(f"Invalid GPX features for {Path(gpx_path).name}: {issues}")
        
        # Log validation summary
        logger.info(f"Feature validation complete:")
        logger.info(f"  Valid videos: {len(valid_videos)}/{len(video_features_dict)} "
                   f"({invalid_video_count} invalid)")
        logger.info(f"  Valid GPX: {len(valid_gpx)}/{len(gpx_database)} "
                   f"({invalid_gpx_count} invalid)")
        
        return valid_videos, valid_gpx
    
    def _find_best_matches(self, video_path, video_features, valid_gpx, top_k, pbar):
        """Find best matches for a video"""
        matches = []
        
        for gpx_path, gpx_data in valid_gpx.items():
            try:
                gpx_features = gpx_data['features']
                
                # Compute comprehensive similarity
                similarities = self.similarity_engine.compute_comprehensive_similarity(
                    video_features, gpx_features, video_path, gpx_path
                )
                
                match_info = {
                    'path': gpx_path,
                    'combined_score': similarities['combined'],
                    'motion_score': similarities['motion_dynamics'],
                    'statistical_score': similarities['statistical_profile'],
                    'temporal_score': similarities['temporal_patterns'],
                    'feature_score': similarities['feature_correlation'],
                    'quality': similarities['quality'],
                    'distance': gpx_data.get('distance', 0),
                    'duration': gpx_data.get('duration', 0),
                    'avg_speed': gpx_data.get('avg_speed', 0),
                    'point_count': gpx_data.get('point_count', 0)
                }
                
                matches.append(match_info)
                
            except Exception as e:
                logger.debug(f"Similarity computation failed for {video_path} vs {gpx_path}: {e}")
                # Add zero-score match to maintain completeness
                matches.append({
                    'path': gpx_path,
                    'combined_score': 0.0,
                    'motion_score': 0.0,
                    'statistical_score': 0.0,
                    'temporal_score': 0.0,
                    'feature_score': 0.0,
                    'quality': 'failed',
                    'distance': gpx_data.get('distance', 0),
                    'duration': gpx_data.get('duration', 0),
                    'avg_speed': gpx_data.get('avg_speed', 0),
                    'point_count': gpx_data.get('point_count', 0)
                })
            
            pbar.update(1)
        
        # Sort by combined score
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return matches[:top_k]
    
    async def _generate_production_report(self, results, output_dir, valid_videos, valid_gpx):
        """Generate comprehensive production report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Analyze results
        analysis = self._analyze_correlation_results(results)
        
        # Create comprehensive report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'system_info': ProductionGPUValidator.validate_system_requirements()
            },
            'input_summary': {
                'total_videos': len(results),
                'valid_videos': len(valid_videos),
                'total_gpx': len(valid_gpx),
                'total_comparisons': len(valid_videos) * len(valid_gpx)
            },
            'performance_metrics': {
                'successful_correlations': analysis['successful_correlations'],
                'failed_correlations': analysis['failed_correlations'],
                'success_rate': analysis['success_rate']
            },
            'quality_analysis': analysis['quality_analysis'],
            'score_distribution': analysis['score_distribution'],
            'detailed_results': analysis['detailed_results']
        }
        
        # Save main report
        async with aiofiles.open(output_path / 'production_correlation_report.json', 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        # Save CSV summary for easy analysis
        await self._save_csv_summary(results, output_path)
        
        # Save quality analysis
        await self._save_quality_analysis(analysis, output_path)
        
        logger.info(f"Production correlation report saved to {output_path}")
        
        # Print executive summary
        self._print_executive_summary(analysis)
    
    def _analyze_correlation_results(self, results):
        """Comprehensive analysis of correlation results"""
        
        analysis = {
            'successful_correlations': 0,
            'failed_correlations': 0,
            'quality_analysis': {
                'excellent': 0,
                'good': 0,
                'fair': 0,
                'poor': 0,
                'very_poor': 0,
                'failed': 0
            },
            'score_distribution': {
                'combined_scores': [],
                'motion_scores': [],
                'statistical_scores': [],
                'temporal_scores': [],
                'feature_scores': []
            },
            'detailed_results': []
        }
        
        for video_path, result in results.items():
            if result is None or not result.get('matches'):
                analysis['failed_correlations'] += 1
                continue
            
            analysis['successful_correlations'] += 1
            best_match = result['matches'][0]
            
            # Quality analysis
            quality = best_match.get('quality', 'failed')
            analysis['quality_analysis'][quality] += 1
            
            # Score distribution
            analysis['score_distribution']['combined_scores'].append(best_match['combined_score'])
            analysis['score_distribution']['motion_scores'].append(best_match.get('motion_score', 0))
            analysis['score_distribution']['statistical_scores'].append(best_match.get('statistical_score', 0))
            analysis['score_distribution']['temporal_scores'].append(best_match.get('temporal_score', 0))
            analysis['score_distribution']['feature_scores'].append(best_match.get('feature_score', 0))
            
            # Detailed result
            analysis['detailed_results'].append({
                'video': str(video_path),
                'video_name': Path(video_path).name,
                'best_match': {
                    'gpx': str(best_match['path']),
                    'gpx_name': Path(best_match['path']).name,
                    'combined_score': best_match['combined_score'],
                    'quality': quality,
                    'breakdown': {
                        'motion': best_match.get('motion_score', 0),
                        'statistical': best_match.get('statistical_score', 0),
                        'temporal': best_match.get('temporal_score', 0),
                        'feature': best_match.get('feature_score', 0)
                    },
                    'gpx_info': {
                        'distance': best_match.get('distance', 0),
                        'duration': best_match.get('duration', 0),
                        'avg_speed': best_match.get('avg_speed', 0),
                        'point_count': best_match.get('point_count', 0)
                    }
                },
                'all_matches': [
                    {
                        'gpx_name': Path(m['path']).name,
                        'score': m['combined_score'],
                        'quality': m.get('quality', 'unknown')
                    } for m in result['matches']
                ]
            })
        
        # Compute statistics
        total_results = len(results)
        analysis['success_rate'] = analysis['successful_correlations'] / total_results if total_results > 0 else 0
        
        # Score statistics
        for score_type, scores in analysis['score_distribution'].items():
            if scores:
                analysis['score_distribution'][f'{score_type}_stats'] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'q25': float(np.percentile(scores, 25)),
                    'q75': float(np.percentile(scores, 75))
                }
        
        return analysis
    
    async def _save_csv_summary(self, results, output_path):
        """Save CSV summary for easy analysis"""
        
        csv_data = []
        
        for video_path, result in results.items():
            if result is None or not result.get('matches'):
                csv_data.append({
                    'video_name': Path(video_path).name,
                    'video_path': str(video_path),
                    'status': 'failed',
                    'best_match_gpx': '',
                    'combined_score': 0.0,
                    'quality': 'failed',
                    'motion_score': 0.0,
                    'statistical_score': 0.0,
                    'temporal_score': 0.0,
                    'feature_score': 0.0
                })
                continue
            
            best_match = result['matches'][0]
            
            csv_data.append({
                'video_name': Path(video_path).name,
                'video_path': str(video_path),
                'status': 'success',
                'best_match_gpx': Path(best_match['path']).name,
                'combined_score': best_match['combined_score'],
                'quality': best_match.get('quality', 'unknown'),
                'motion_score': best_match.get('motion_score', 0),
                'statistical_score': best_match.get('statistical_score', 0),
                'temporal_score': best_match.get('temporal_score', 0),
                'feature_score': best_match.get('feature_score', 0),
                'gpx_distance': best_match.get('distance', 0),
                'gpx_duration': best_match.get('duration', 0),
                'gpx_avg_speed': best_match.get('avg_speed', 0)
            })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        csv_path = output_path / 'correlation_summary.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV summary saved to {csv_path}")
    
    async def _save_quality_analysis(self, analysis, output_path):
        """Save detailed quality analysis"""
        
        quality_report = {
            'quality_distribution': analysis['quality_analysis'],
            'score_statistics': {
                key: value for key, value in analysis['score_distribution'].items()
                if key.endswith('_stats')
            },
            'recommendations': self._generate_recommendations(analysis)
        }
        
        async with aiofiles.open(output_path / 'quality_analysis.json', 'w') as f:
            await f.write(json.dumps(quality_report, indent=2))
    
    def _generate_recommendations(self, analysis):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Success rate analysis
        success_rate = analysis['success_rate']
        if success_rate < 0.8:
            recommendations.append({
                'priority': 'high',
                'category': 'success_rate',
                'issue': f'Low success rate: {success_rate:.1%}',
                'recommendation': 'Check video and GPX feature extraction quality. Consider adjusting processing parameters.',
                'action_items': [
                    'Review failed video processing logs',
                    'Validate GPX file quality and completeness',
                    'Consider increasing video sampling rate',
                    'Check for format compatibility issues'
                ]
            })
        
        # Quality distribution analysis
        quality_dist = analysis['quality_analysis']
        excellent_rate = quality_dist['excellent'] / max(sum(quality_dist.values()), 1)
        
        if excellent_rate < 0.2:
            recommendations.append({
                'priority': 'medium',
                'category': 'quality',
                'issue': f'Low excellent match rate: {excellent_rate:.1%}',
                'recommendation': 'Improve feature extraction or adjust similarity thresholds.',
                'action_items': [
                    'Review similarity computation weights',
                    'Consider temporal alignment methods',
                    'Validate GPS track accuracy',
                    'Check video motion detection sensitivity'
                ]
            })
        
        # Score analysis
        if 'combined_scores_stats' in analysis['score_distribution']:
            stats = analysis['score_distribution']['combined_scores_stats']
            mean_score = stats['mean']
            
            if mean_score < 0.3:
                recommendations.append({
                    'priority': 'high',
                    'category': 'scores',
                    'issue': f'Low average similarity score: {mean_score:.3f}',
                    'recommendation': 'Fundamental issues with correlation method or data quality.',
                    'action_items': [
                        'Verify temporal synchronization between videos and GPS',
                        'Check coordinate system compatibility',
                        'Review feature extraction algorithms',
                        'Consider manual validation of known matches'
                    ]
                })
        
        # Feature-specific recommendations
        if 'motion_scores_stats' in analysis['score_distribution']:
            motion_stats = analysis['score_distribution']['motion_scores_stats']
            if motion_stats['mean'] < 0.2:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'motion_features',
                    'issue': 'Poor motion correlation',
                    'recommendation': 'Improve motion feature extraction or video quality.',
                    'action_items': [
                        'Check video stabilization quality',
                        'Review motion detection parameters',
                        'Validate GPS accuracy and frequency'
                    ]
                })
        
        if not recommendations:
            recommendations.append({
                'priority': 'info',
                'category': 'performance',
                'issue': 'System performing well',
                'recommendation': 'Current correlation quality is acceptable.',
                'action_items': ['Monitor performance over time', 'Consider optimization opportunities']
            })
        
        return recommendations
    
    def _print_executive_summary(self, analysis):
        """Print executive summary to console"""
        
        print("\n" + "="*80)
        print("🎯 PRODUCTION CORRELATION EXECUTIVE SUMMARY")
        print("="*80)
        
        # Overall performance
        total_correlations = analysis['successful_correlations'] + analysis['failed_correlations']
        success_rate = analysis['success_rate']
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Videos Processed: {total_correlations}")
        print(f"   Successful Correlations: {analysis['successful_correlations']}")
        print(f"   Failed Correlations: {analysis['failed_correlations']}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        # Quality distribution
        print(f"\n🏆 QUALITY DISTRIBUTION:")
        quality_dist = analysis['quality_analysis']
        for quality, count in quality_dist.items():
            if count > 0:
                percentage = count / total_correlations * 100 if total_correlations > 0 else 0
                print(f"   {quality.title()}: {count} ({percentage:.1f}%)")
        
        # Score statistics
        if 'combined_scores_stats' in analysis['score_distribution']:
            stats = analysis['score_distribution']['combined_scores_stats']
            print(f"\n📈 SIMILARITY SCORES:")
            print(f"   Average Score: {stats['mean']:.3f}")
            print(f"   Best Score: {stats['max']:.3f}")
            print(f"   Median Score: {stats['median']:.3f}")
            print(f"   Score Range: {stats['min']:.3f} - {stats['max']:.3f}")
        
        # Top performing matches
        top_matches = sorted(
            [r for r in analysis['detailed_results'] if r['best_match']['combined_score'] > 0],
            key=lambda x: x['best_match']['combined_score'],
            reverse=True
        )[:5]
        
        if top_matches:
            print(f"\n🥇 TOP MATCHES:")
            for i, match in enumerate(top_matches, 1):
                video_name = match['video_name']
                gpx_name = match['best_match']['gpx_name']
                score = match['best_match']['combined_score']
                quality = match['best_match']['quality']
                print(f"   {i}. {video_name} → {gpx_name} ({score:.3f}, {quality})")
        
        print("\n" + "="*80)

def process_video_production(video_path, decoder, feature_extractor, gpu_idx, target_size, sample_rate):
    """Process single video with production-grade error handling"""
    
    try:
        logger.debug(f"Processing video on GPU {gpu_idx}: {Path(video_path).name}")
        
        # Decode video
        frames_tensor, fps, duration, _ = decoder.decode_video_optimized(
            video_path, sample_rate=sample_rate, target_size=target_size, gpu_id=gpu_idx
        )
        
        if frames_tensor is None:
            raise RuntimeError("Video decoding failed")
        
        # Extract features
        features = feature_extractor.extract_comprehensive_features(frames_tensor, gpu_idx)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_idx
        features['target_size'] = target_size
        features['sample_rate'] = sample_rate
        
        # Validate features
        validator = ProductionFeatureValidator()
        is_valid, issues = validator.validate_video_features(features, video_path)
        
        if not is_valid:
            logger.warning(f"Video feature validation failed for {Path(video_path).name}: {issues}")
            return None
        
        logger.debug(f"Successfully processed {Path(video_path).name}")
        return features
        
    except Exception as e:
        logger.error(f"Video processing failed for {Path(video_path).name}: {e}")
        return None

def main():
    """Production-ready main function with comprehensive error handling"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Production-Ready Ultra-Optimized Multi-GPU Video-GPX Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d /path/to/data --gpu_ids 0 1
  %(prog)s -d /path/to/data --force --debug
  %(prog)s --test_system
  %(prog)s -d /path/to/data --video_size 480 270 --sample_rate 1.0
        """
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", 
                       help="Directory containing videos and GPX files")
    
    # Optional arguments
    parser.add_argument("-o", "--output", default="./correlation_results",
                       help="Output directory (default: ./correlation_results)")
    parser.add_argument("-c", "--cache", default="./correlation_cache",
                       help="Cache directory (default: ./correlation_cache)")
    parser.add_argument("-k", "--top_k", type=int, default=5,
                       help="Number of top matches per video (default: 5)")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0],
                       help="GPU IDs to use (default: [0])")
    parser.add_argument("--video_size", nargs=2, type=int, default=[640, 360],
                       help="Target video resolution (default: 640 360)")
    parser.add_argument("--sample_rate", type=float, default=2.0,
                       help="Video sampling rate in FPS (default: 2.0)")
    
    # Processing options
    parser.add_argument("--force", action='store_true',
                       help="Force reprocessing (ignore cache)")
    parser.add_argument("--sequential", action='store_true',
                       help="Process videos sequentially")
    parser.add_argument("--max_workers", type=int,
                       help="Maximum worker processes for GPX processing")
    
    # H.264 4:4:4 options
    parser.add_argument("--disable_444_conversion", action='store_true',
                       help="Disable H.264 4:4:4 to 4:2:0 conversion")
    parser.add_argument("--force_disk_conversion", action='store_true',
                       help="Force disk storage for H.264 4:4:4 conversions")
    parser.add_argument("--keep_444_backup", action='store_true',
                       help="Keep backup of original H.264 4:4:4 files")
    
    # System options
    parser.add_argument("--test_system", action='store_true',
                       help="Test system requirements and exit")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug logging")
    parser.add_argument("--log_file", 
                       help="Log file path (default: correlation.log)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log_file or "correlation.log"
    logger = setup_logging(log_level, log_file)
    
    logger.info("Starting Production Video-GPX Correlation System")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    
    try:
        # System validation
        logger.info("Validating system requirements...")
        validation = ProductionGPUValidator.validate_system_requirements()
        
        if args.test_system:
            print("\n🔧 SYSTEM VALIDATION RESULTS:")
            print("="*50)
            print(f"CUDA Available: {'✓' if validation['cuda_available'] else '✗'}")
            print(f"CuPy Available: {'✓' if validation['cupy_available'] else '✗'}")
            print(f"FFmpeg GPU: {'✓' if validation['ffmpeg_gpu'] else '✗'}")
            print(f"GPU Count: {validation['gpu_count']}")
            print(f"System Memory: {validation['system_memory']:.1f}GB")
            
            if validation['gpu_memory']:
                print("\nGPU Details:")
                for gpu in validation['gpu_memory']:
                    print(f"  GPU {gpu['device']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
            
            if validation['issues']:
                print("\nIssues Detected:")
                for issue in validation['issues']:
                    print(f"  ⚠️  {issue}")
            
            if not validation['issues']:
                print("\n✅ System ready for production use!")
            else:
                print("\n❌ System issues detected - may affect performance")
            
            return
        
        # Validate required arguments
        if not args.directory:
            parser.error("Directory argument (-d/--directory) is required")
        
        if not os.path.exists(args.directory):
            raise FileNotFoundError(f"Directory not found: {args.directory}")
        
        # Enforce GPU requirements
        ProductionGPUValidator.enforce_gpu_requirements(args.gpu_ids)
        
        # Setup directories
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        cache_dir = Path(args.cache)
        cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Find input files
        logger.info("Scanning for video and GPX files...")
        
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
        video_files = sorted(list(set(video_files)))
        
        gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
        gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
        gpx_files = sorted(list(set(gpx_files)))
        
        logger.info(f"Found {len(video_files)} video files and {len(gpx_files)} GPX files")
        
        if not video_files:
            raise RuntimeError(f"No video files found in {args.directory}")
        
        if not gpx_files:
            raise RuntimeError(f"No GPX files found in {args.directory}")
        
        # Initialize production components
        logger.info("Initializing production components...")
        
        conversion_settings = {
            'disable_444_conversion': args.disable_444_conversion,
            'force_disk_conversion': args.force_disk_conversion,
            'keep_444_backup': args.keep_444_backup
        }
        
        decoder = OptimizedFFmpegDecoder(args.gpu_ids, conversion_settings)
        feature_extractor = ProductionFeatureExtractor(args.gpu_ids)
        gpx_processor = ProductionGPXProcessor()
        correlator = ProductionCorrelator(args.gpu_ids)
        
        logger.info("All production components initialized successfully")
        
        # Process videos
        logger.info("Processing video files...")
        video_cache_path = cache_dir / "production_video_features.pkl"
        
        video_features = {}
        if video_cache_path.exists() and not args.force:
            logger.info("Loading cached video features...")
            try:
                with open(video_cache_path, 'rb') as f:
                    video_features = pickle.load(f)
                
                valid_cached = sum(1 for v in video_features.values() if v is not None)
                logger.info(f"Loaded {valid_cached}/{len(video_features)} valid cached video features")
                
                # Check if we need to process more videos
                missing_videos = [v for v in video_files if v not in video_features]
                if missing_videos:
                    logger.info(f"Processing {len(missing_videos)} new videos...")
                    
            except Exception as e:
                logger.warning(f"Failed to load video cache: {e}")
                video_features = {}
        
        # Process missing videos
        videos_to_process = [v for v in video_files if v not in video_features or video_features[v] is None]
        
        if videos_to_process or args.force:
            logger.info(f"Processing {len(videos_to_process)} video files...")
            
            for i, video_path in enumerate(tqdm(videos_to_process, desc="Processing videos")):
                gpu_idx = i % len(args.gpu_ids)
                
                try:
                    features = process_video_production(
                        video_path, decoder, feature_extractor, gpu_idx, 
                        tuple(args.video_size), args.sample_rate
                    )
                    
                    video_features[video_path] = features
                    
                    # Periodic cache save
                    if (i + 1) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
                        logger.debug(f"Saved progress: {i+1}/{len(videos_to_process)} videos")
                
                except Exception as e:
                    logger.error(f"Failed to process video {video_path}: {e}")
                    video_features[video_path] = None
            
            # Final save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            
            logger.info("Video processing complete")
        
        # Process GPX files
        logger.info("Processing GPX files...")
        gpx_cache_path = cache_dir / "production_gpx_features.pkl"
        
        gpx_database = {}
        if gpx_cache_path.exists() and not args.force:
            logger.info("Loading cached GPX features...")
            try:
                with open(gpx_cache_path, 'rb') as f:
                    gpx_database = pickle.load(f)
                
                valid_cached = sum(1 for g in gpx_database.values() if g is not None)
                logger.info(f"Loaded {valid_cached}/{len(gpx_database)} valid cached GPX features")
                
            except Exception as e:
                logger.warning(f"Failed to load GPX cache: {e}")
                gpx_database = {}
        
        # Process missing GPX files
        missing_gpx = [g for g in gpx_files if g not in gpx_database]
        
        if missing_gpx or args.force:
            max_workers = args.max_workers or min(16, mp.cpu_count())
            gpx_results = gpx_processor.process_gpx_files(gpx_files, max_workers)
            gpx_database.update(gpx_results)
            
            # Save GPX cache
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_database, f)
            
            logger.info("GPX processing complete")
        
        # Perform correlation analysis
        logger.info("Starting correlation analysis...")
        start_time = time.time()
        
        results = correlator.correlate_comprehensive(
            video_features, gpx_database, output_dir, top_k=args.top_k
        )
        
        correlation_time = time.time() - start_time
        
        # Save results
        results_path = output_dir / "production_correlations.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Correlation analysis complete in {correlation_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")
        
        # Cleanup
        logger.info("Performing cleanup...")
        try:
            decoder.cleanup()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        logger.info("Production correlation system completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Production system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() d({
                'priority': 'high',
                'category': 'success_rate',
                'issue': f'Low success rate: {success_rate:.1%}',
                'recommendation': 'Check video and GPX feature extraction quality. Consider adjusting processing parameters.',
                'action_items': [
                    'Review failed video processing logs',
                    'Validate GPX file quality and completeness',
                    'Consider increasing video sampling rate',
                    'Check for format compatibility issues'
                ]
            })
        
        # Quality distribution analysis
        quality_dist = analysis['quality_analysis']
        excellent_rate = quality_dist['excellent'] / max(sum(quality_dist.values()), 1)
        
        if excellent_rate < 0.2:
            recommendations.append({
                'priority': 'medium',
                'category': 'quality',
                'issue': f'Low excellent match rate: {excellent_rate:.1%}',
                'recommendation': 'Improve feature extraction or adjust similarity thresholds.',
                'action_items': [
                    'Review similarity computation weights',
                    'Consider temporal alignment methods',
                    'Validate GPS track accuracy',
                    'Check video motion detection sensitivity'
                ]
            })
        
        # Score analysis
        if 'combined_scores_stats' in analysis['score_distribution']:
            stats = analysis['score_distribution']['combined_scores_stats']
            mean_score = stats['mean']
            
            if mean_score < 0.3:
                recommendations.append({
                    'priority': 'high',
                    'category': 'scores',
                    'issue': f'Low average similarity score: {mean_score:.3f}',
                    'recommendation': 'Fundamental issues with correlation method or data quality.',
                    'action_items': [
                        'Verify temporal synchronization between videos and GPS',
                        'Check coordinate system compatibility',
                        'Review feature extraction algorithms',
                        'Consider manual validation of known matches'
                    ]
                })
        
        # Feature-specific recommendations
        if 'motion_scores_stats' in analysis['score_distribution']:
            motion_stats = analysis['score_distribution']['motion_scores_stats']
            if motion_stats['mean'] < 0.2:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'motion_features',
                    'issue': 'Poor motion correlation',
                    'recommendation': 'Improve motion feature extraction or video quality.',
                    'action_items': [
                        'Check video stabilization quality',
                        'Review motion detection parameters',
                        'Validate GPS accuracy and frequency'
                    ]
                })
        
        if not recommendations:
            recommendations.append({
                'priority': 'info',
                'category': 'performance',
                'issue': 'System performing well',
                'recommendation': 'Current correlation quality is acceptable.',
                'action_items': ['Monitor performance over time', 'Consider optimization opportunities']
            })
        
        return recommendations
    
    def _print_executive_summary(self, analysis):
        """Print executive summary to console"""
        
        print("\n" + "="*80)
        print("🎯 PRODUCTION CORRELATION EXECUTIVE SUMMARY")
        print("="*80)
        
        # Overall performance
        total_correlations = analysis['successful_correlations'] + analysis['failed_correlations']
        success_rate = analysis['success_rate']
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Videos Processed: {total_correlations}")
        print(f"   Successful Correlations: {analysis['successful_correlations']}")
        print(f"   Failed Correlations: {analysis['failed_correlations']}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        # Quality distribution
        print(f"\n🏆 QUALITY DISTRIBUTION:")
        quality_dist = analysis['quality_analysis']
        for quality, count in quality_dist.items():
            if count > 0:
                percentage = count / total_correlations * 100 if total_correlations > 0 else 0
                print(f"   {quality.title()}: {count} ({percentage:.1f}%)")
        
        # Score statistics
        if 'combined_scores_stats' in analysis['score_distribution']:
            stats = analysis['score_distribution']['combined_scores_stats']
            print(f"\n📈 SIMILARITY SCORES:")
            print(f"   Average Score: {stats['mean']:.3f}")
            print(f"   Best Score: {stats['max']:.3f}")
            print(f"   Median Score: {stats['median']:.3f}")
            print(f"   Score Range: {stats['min']:.3f} - {stats['max']:.3f}")
        
        # Top performing matches
        top_matches = sorted(
            [r for r in analysis['detailed_results'] if r['best_match']['combined_score'] > 0],
            key=lambda x: x['best_match']['combined_score'],
            reverse=True
        )[:5]
        
        if top_matches:
            print(f"\n🥇 TOP MATCHES:")
            for i, match in enumerate(top_matches, 1):
                video_name = match['video_name']
                gpx_name = match['best_match']['gpx_name']
                score = match['best_match']['combined_score']
                quality = match['best_match']['quality']
                print(f"   {i}. {video_name} → {gpx_name} ({score:.3f}, {quality})")
        
        print("\n" + "="*80)

def process_video_production(video_path, decoder, feature_extractor, gpu_idx, target_size, sample_rate):
    """Process single video with production-grade error handling"""
    
    try:
        logger.debug(f"Processing video on GPU {gpu_idx}: {Path(video_path).name}")
        
        # Decode video
        frames_tensor, fps, duration, _ = decoder.decode_video_optimized(
            video_path, sample_rate=sample_rate, target_size=target_size, gpu_id=gpu_idx
        )
        
        if frames_tensor is None:
            raise RuntimeError("Video decoding failed")
        
        # Extract features
        features = feature_extractor.extract_comprehensive_features(frames_tensor, gpu_idx)
        
        # Add metadata
        features['duration'] = duration
        features['fps'] = fps
        features['processing_gpu'] = gpu_idx
        features['target_size'] = target_size
        features['sample_rate'] = sample_rate
        
        # Validate features
        validator = ProductionFeatureValidator()
        is_valid, issues = validator.validate_video_features(features, video_path)
        
        if not is_valid:
            logger.warning(f"Video feature validation failed for {Path(video_path).name}: {issues}")
            return None
        
        logger.debug(f"Successfully processed {Path(video_path).name}")
        return features
        
    except Exception as e:
        logger.error(f"Video processing failed for {Path(video_path).name}: {e}")
        return None

def main():
    """Production-ready main function with comprehensive error handling"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Production-Ready Ultra-Optimized Multi-GPU Video-GPX Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d /path/to/data --gpu_ids 0 1
  %(prog)s -d /path/to/data --force --debug
  %(prog)s --test_system
  %(prog)s -d /path/to/data --video_size 480 270 --sample_rate 1.0
        """
    )
    
    # Required arguments
    parser.add_argument("-d", "--directory", 
                       help="Directory containing videos and GPX files")
    
    # Optional arguments
    parser.add_argument("-o", "--output", default="./correlation_results",
                       help="Output directory (default: ./correlation_results)")
    parser.add_argument("-c", "--cache", default="./correlation_cache",
                       help="Cache directory (default: ./correlation_cache)")
    parser.add_argument("-k", "--top_k", type=int, default=5,
                       help="Number of top matches per video (default: 5)")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0],
                       help="GPU IDs to use (default: [0])")
    parser.add_argument("--video_size", nargs=2, type=int, default=[640, 360],
                       help="Target video resolution (default: 640 360)")
    parser.add_argument("--sample_rate", type=float, default=2.0,
                       help="Video sampling rate in FPS (default: 2.0)")
    
    # Processing options
    parser.add_argument("--force", action='store_true',
                       help="Force reprocessing (ignore cache)")
    parser.add_argument("--sequential", action='store_true',
                       help="Process videos sequentially")
    parser.add_argument("--max_workers", type=int,
                       help="Maximum worker processes for GPX processing")
    
    # H.264 4:4:4 options
    parser.add_argument("--disable_444_conversion", action='store_true',
                       help="Disable H.264 4:4:4 to 4:2:0 conversion")
    parser.add_argument("--force_disk_conversion", action='store_true',
                       help="Force disk storage for H.264 4:4:4 conversions")
    parser.add_argument("--keep_444_backup", action='store_true',
                       help="Keep backup of original H.264 4:4:4 files")
    
    # System options
    parser.add_argument("--test_system", action='store_true',
                       help="Test system requirements and exit")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug logging")
    parser.add_argument("--log_file", 
                       help="Log file path (default: correlation.log)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log_file or "correlation.log"
    logger = setup_logging(log_level, log_file)
    
    logger.info("Starting Production Video-GPX Correlation System")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    
    try:
        # System validation
        logger.info("Validating system requirements...")
        validation = ProductionGPUValidator.validate_system_requirements()
        
        if args.test_system:
            print("\n🔧 SYSTEM VALIDATION RESULTS:")
            print("="*50)
            print(f"CUDA Available: {'✓' if validation['cuda_available'] else '✗'}")
            print(f"CuPy Available: {'✓' if validation['cupy_available'] else '✗'}")
            print(f"FFmpeg GPU: {'✓' if validation['ffmpeg_gpu'] else '✗'}")
            print(f"GPU Count: {validation['gpu_count']}")
            print(f"System Memory: {validation['system_memory']:.1f}GB")
            
            if validation['gpu_memory']:
                print("\nGPU Details:")
                for gpu in validation['gpu_memory']:
                    print(f"  GPU {gpu['device']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
            
            if validation['issues']:
                print("\nIssues Detected:")
                for issue in validation['issues']:
                    print(f"  ⚠️  {issue}")
            
            if not validation['issues']:
                print("\n✅ System ready for production use!")
            else:
                print("\n❌ System issues detected - may affect performance")
            
            return
        
        # Validate required arguments
        if not args.directory:
            parser.error("Directory argument (-d/--directory) is required")
        
        if not os.path.exists(args.directory):
            raise FileNotFoundError(f"Directory not found: {args.directory}")
        
        # Enforce GPU requirements
        ProductionGPUValidator.enforce_gpu_requirements(args.gpu_ids)
        
        # Setup directories
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        cache_dir = Path(args.cache)
        cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Find input files
        logger.info("Scanning for video and GPX files...")
        
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV', 'MKV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.directory, f'*.{ext}')))
        video_files = sorted(list(set(video_files)))
        
        gpx_files = glob.glob(os.path.join(args.directory, '*.gpx'))
        gpx_files.extend(glob.glob(os.path.join(args.directory, '*.GPX')))
        gpx_files = sorted(list(set(gpx_files)))
        
        logger.info(f"Found {len(video_files)} video files and {len(gpx_files)} GPX files")
        
        if not video_files:
            raise RuntimeError(f"No video files found in {args.directory}")
        
        if not gpx_files:
            raise RuntimeError(f"No GPX files found in {args.directory}")
        
        # Initialize production components
        logger.info("Initializing production components...")
        
        conversion_settings = {
            'disable_444_conversion': args.disable_444_conversion,
            'force_disk_conversion': args.force_disk_conversion,
            'keep_444_backup': args.keep_444_backup
        }
        
        decoder = OptimizedFFmpegDecoder(args.gpu_ids, conversion_settings)
        feature_extractor = ProductionFeatureExtractor(args.gpu_ids)
        gpx_processor = ProductionGPXProcessor()
        correlator = ProductionCorrelator(args.gpu_ids)
        
        logger.info("All production components initialized successfully")
        
        # Process videos
        logger.info("Processing video files...")
        video_cache_path = cache_dir / "production_video_features.pkl"
        
        video_features = {}
        if video_cache_path.exists() and not args.force:
            logger.info("Loading cached video features...")
            try:
                with open(video_cache_path, 'rb') as f:
                    video_features = pickle.load(f)
                
                valid_cached = sum(1 for v in video_features.values() if v is not None)
                logger.info(f"Loaded {valid_cached}/{len(video_features)} valid cached video features")
                
                # Check if we need to process more videos
                missing_videos = [v for v in video_files if v not in video_features]
                if missing_videos:
                    logger.info(f"Processing {len(missing_videos)} new videos...")
                    
            except Exception as e:
                logger.warning(f"Failed to load video cache: {e}")
                video_features = {}
        
        # Process missing videos
        videos_to_process = [v for v in video_files if v not in video_features or video_features[v] is None]
        
        if videos_to_process or args.force:
            logger.info(f"Processing {len(videos_to_process)} video files...")
            
            for i, video_path in enumerate(tqdm(videos_to_process, desc="Processing videos")):
                gpu_idx = i % len(args.gpu_ids)
                
                try:
                    features = process_video_production(
                        video_path, decoder, feature_extractor, gpu_idx, 
                        tuple(args.video_size), args.sample_rate
                    )
                    
                    video_features[video_path] = features
                    
                    # Periodic cache save
                    if (i + 1) % 10 == 0:
                        with open(video_cache_path, 'wb') as f:
                            pickle.dump(video_features, f)
                        logger.debug(f"Saved progress: {i+1}/{len(videos_to_process)} videos")
                
                except Exception as e:
                    logger.error(f"Failed to process video {video_path}: {e}")
                    video_features[video_path] = None
            
            # Final save
            with open(video_cache_path, 'wb') as f:
                pickle.dump(video_features, f)
            
            logger.info("Video processing complete")
        
        # Process GPX files
        logger.info("Processing GPX files...")
        gpx_cache_path = cache_dir / "production_gpx_features.pkl"
        
        gpx_database = {}
        if gpx_cache_path.exists() and not args.force:
            logger.info("Loading cached GPX features...")
            try:
                with open(gpx_cache_path, 'rb') as f:
                    gpx_database = pickle.load(f)
                
                valid_cached = sum(1 for g in gpx_database.values() if g is not None)
                logger.info(f"Loaded {valid_cached}/{len(gpx_database)} valid cached GPX features")
                
            except Exception as e:
                logger.warning(f"Failed to load GPX cache: {e}")
                gpx_database = {}
        
        # Process missing GPX files
        missing_gpx = [g for g in gpx_files if g not in gpx_database]
        
        if missing_gpx or args.force:
            max_workers = args.max_workers or min(16, mp.cpu_count())
            gpx_results = gpx_processor.process_gpx_files(gpx_files, max_workers)
            gpx_database.update(gpx_results)
            
            # Save GPX cache
            with open(gpx_cache_path, 'wb') as f:
                pickle.dump(gpx_database, f)
            
            logger.info("GPX processing complete")
        
        # Perform correlation analysis
        logger.info("Starting correlation analysis...")
        start_time = time.time()
        
        results = correlator.correlate_comprehensive(
            video_features, gpx_database, output_dir, top_k=args.top_k
        )
        
        correlation_time = time.time() - start_time
        
        # Save results
        results_path = output_dir / "production_correlations.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Correlation analysis complete in {correlation_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")
        
        # Cleanup
        logger.info("Performing cleanup...")
        try:
            decoder.cleanup()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        logger.info("Production correlation system completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Production system failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()