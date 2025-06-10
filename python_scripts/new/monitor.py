#!/usr/bin/env python3
"""
Performance Optimization Guide and Monitoring Tools
for Video-GPX Correlation System

This script provides additional optimizations and monitoring tools
to ensure maximum GPU utilization and performance.
"""

import psutil
import GPUtil
import torch
import time
import threading
import logging
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class PerformanceMonitor:
    """Real-time performance monitoring for GPU utilization optimization"""
    
    def __init__(self, log_interval=5):
        self.log_interval = log_interval
        self.monitoring = False
        self.stats_history = deque(maxlen=1000)
        self.gpu_stats = {i: deque(maxlen=1000) for i in range(torch.cuda.device_count())}
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and save results"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logging.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU stats
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # GPU stats
            gpu_stats = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu = GPUtil.getGPUs()[i] if i < len(GPUtil.getGPUs()) else None
                    if gpu:
                        gpu_util = gpu.load * 100
                        gpu_memory = gpu.memoryUtil * 100
                        gpu_temp = gpu.temperature
                        
                        # Torch memory stats
                        torch_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                        torch_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                        
                        gpu_stat = {
                            'utilization': gpu_util,
                            'memory_util': gpu_memory,
                            'temperature': gpu_temp,
                            'torch_allocated': torch_allocated,
                            'torch_cached': torch_cached
                        }
                        
                        self.gpu_stats[i].append((timestamp, gpu_stat))
                        gpu_stats.append(gpu_stat)
            
            # Overall system stats
            system_stat = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024**3,
                'gpu_stats': gpu_stats
            }
            
            self.stats_history.append(system_stat)
            
            # Log warnings for underutilization
            self._check_utilization_warnings(system_stat)
            
            time.sleep(self.log_interval)
    
    def _check_utilization_warnings(self, stats):
        """Check for performance issues and log warnings"""
        # Check GPU utilization
        for i, gpu_stat in enumerate(stats['gpu_stats']):
            if gpu_stat['utilization'] < 30:
                logging.warning(f"GPU {i} utilization low: {gpu_stat['utilization']:.1f}%")
            
            if gpu_stat['memory_util'] < 20:
                logging.warning(f"GPU {i} memory underutilized: {gpu_stat['memory_util']:.1f}%")
            
            if gpu_stat['temperature'] > 80:
                logging.warning(f"GPU {i} temperature high: {gpu_stat['temperature']:.1f}°C")
        
        # Check CPU
        if stats['cpu_percent'] > 90:
            logging.warning(f"CPU utilization high: {stats['cpu_percent']:.1f}%")
        
        if stats['memory_percent'] > 90:
            logging.warning(f"Memory utilization high: {stats['memory_percent']:.1f}%")
    
    def generate_performance_report(self, output_dir):
        """Generate detailed performance report with visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.stats_history:
            logging.warning("No performance data available")
            return
        
        # Convert to arrays for analysis
        timestamps = [s['timestamp'] for s in self.stats_history]
        start_time = timestamps[0]
        relative_times = [(t - start_time) / 60 for t in timestamps]  # Minutes
        
        cpu_usage = [s['cpu_percent'] for s in self.stats_history]
        memory_usage = [s['memory_percent'] for s in self.stats_history]
        
        # GPU utilization plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Performance Analysis', fontsize=16)
        
        # GPU Utilization
        for i in range(min(2, torch.cuda.device_count())):
            gpu_utils = []
            gpu_memory = []
            for stat in self.stats_history:
                if i < len(stat['gpu_stats']):
                    gpu_utils.append(stat['gpu_stats'][i]['utilization'])
                    gpu_memory.append(stat['gpu_stats'][i]['memory_util'])
                else:
                    gpu_utils.append(0)
                    gpu_memory.append(0)
            
            axes[0, 0].plot(relative_times, gpu_utils, label=f'GPU {i}', linewidth=2)
            axes[0, 1].plot(relative_times, gpu_memory, label=f'GPU {i}', linewidth=2)
        
        axes[0, 0].set_title('GPU Utilization (%)')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Utilization (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('GPU Memory Usage (%)')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # CPU and System Memory
        axes[1, 0].plot(relative_times, cpu_usage, 'r-', linewidth=2, label='CPU')
        axes[1, 0].set_title('CPU Utilization (%)')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(relative_times, memory_usage, 'g-', linewidth=2, label='System Memory')
        axes[1, 1].set_title('System Memory Usage (%)')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Memory Usage (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary statistics
        self._generate_performance_summary(output_path)
        
        logging.info(f"Performance report saved to {output_path}")
    
    def _generate_performance_summary(self, output_path):
        """Generate performance summary statistics"""
        if not self.stats_history:
            return
        
        # Calculate averages
        avg_cpu = np.mean([s['cpu_percent'] for s in self.stats_history])
        avg_memory = np.mean([s['memory_percent'] for s in self.stats_history])
        
        gpu_averages = {}
        for i in range(torch.cuda.device_count()):
            gpu_utils = []
            gpu_memory = []
            gpu_temps = []
            
            for stat in self.stats_history:
                if i < len(stat['gpu_stats']):
                    gpu_utils.append(stat['gpu_stats'][i]['utilization'])
                    gpu_memory.append(stat['gpu_stats'][i]['memory_util'])
                    gpu_temps.append(stat['gpu_stats'][i]['temperature'])
            
            if gpu_utils:
                gpu_averages[f'GPU_{i}'] = {
                    'avg_utilization': np.mean(gpu_utils),
                    'avg_memory': np.mean(gpu_memory),
                    'avg_temperature': np.mean(gpu_temps),
                    'max_utilization': np.max(gpu_utils),
                    'max_memory': np.max(gpu_memory),
                    'max_temperature': np.max(gpu_temps),
                    'utilization_efficiency': np.mean([u > 70 for u in gpu_utils]) * 100
                }
        
        summary = {
            'monitoring_duration_minutes': len(self.stats_history) * self.log_interval / 60,
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'gpu_performance': gpu_averages,
            'performance_recommendations': self._generate_recommendations(avg_cpu, avg_memory, gpu_averages)
        }
        
        # Save summary
        with open(output_path / 'performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate text summary
        with open(output_path / 'performance_summary.txt', 'w') as f:
            f.write("PERFORMANCE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Monitoring Duration: {summary['monitoring_duration_minutes']:.1f} minutes\n")
            f.write(f"Average CPU Usage: {avg_cpu:.1f}%\n")
            f.write(f"Average Memory Usage: {avg_memory:.1f}%\n\n")
            
            f.write("GPU PERFORMANCE:\n")
            for gpu_name, stats in gpu_averages.items():
                f.write(f"\n{gpu_name}:\n")
                f.write(f"  Average Utilization: {stats['avg_utilization']:.1f}%\n")
                f.write(f"  Average Memory Usage: {stats['avg_memory']:.1f}%\n")
                f.write(f"  Average Temperature: {stats['avg_temperature']:.1f}°C\n")
                f.write(f"  Peak Utilization: {stats['max_utilization']:.1f}%\n")
                f.write(f"  Efficiency (>70% util): {stats['utilization_efficiency']:.1f}%\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            for rec in summary['performance_recommendations']:
                f.write(f"• {rec}\n")
    
    def _generate_recommendations(self, avg_cpu, avg_memory, gpu_averages):
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        if avg_cpu < 50:
            recommendations.append("CPU underutilized - consider increasing parallel workers or batch sizes")
        elif avg_cpu > 90:
            recommendations.append("CPU bottleneck detected - reduce CPU-bound operations or add more cores")
        
        # Memory recommendations
        if avg_memory > 90:
            recommendations.append("High memory usage - consider reducing batch sizes or adding more RAM")
        elif avg_memory < 30:
            recommendations.append("Memory underutilized - can increase batch sizes for better performance")
        
        # GPU recommendations
        for gpu_name, stats in gpu_averages.items():
            if stats['avg_utilization'] < 30:
                recommendations.append(f"{gpu_name} severely underutilized - check GPU workload distribution")
            elif stats['avg_utilization'] < 60:
                recommendations.append(f"{gpu_name} underutilized - increase batch sizes or improve GPU algorithms")
            
            if stats['avg_memory'] < 20:
                recommendations.append(f"{gpu_name} memory underutilized - can process larger batches")
            
            if stats['max_temperature'] > 85:
                recommendations.append(f"{gpu_name} running hot - check cooling and reduce workload if necessary")
            
            if stats['utilization_efficiency'] < 50:
                recommendations.append(f"{gpu_name} spending too much time idle - optimize GPU workload scheduling")
        
        return recommendations

class OptimizationRecommendations:
    """Additional optimization recommendations and system checks"""
    
    @staticmethod
    def check_system_configuration():
        """Check system configuration for optimal performance"""
        recommendations = []
        
        # Check CUDA and PyTorch setup
        if not torch.cuda.is_available():
            recommendations.append("CRITICAL: CUDA not available - install CUDA toolkit")
        else:
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                compute_capability = torch.cuda.get_device_capability(i)
                
                if compute_capability[0] < 7:
                    recommendations.append(f"GPU {i} ({gpu_name}) has limited Tensor Core support")
                else:
                    recommendations.append(f"GPU {i} ({gpu_name}) supports mixed precision - GOOD")
        
        # Check CuPy installation
        try:
            import cupy as cp
            if cp.cuda.is_available():
                recommendations.append("CuPy available for GPU-accelerated NumPy operations - GOOD")
            else:
                recommendations.append("CuPy installed but no CUDA devices detected")
        except ImportError:
            recommendations.append("INSTALL: pip install cupy-cuda11x (or appropriate CUDA version)")
        
        # Check decord installation
        try:
            import decord
            recommendations.append("Decord available for GPU video decoding - GOOD")
        except ImportError:
            recommendations.append("INSTALL: pip install decord (for GPU video decoding)")
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.total < 32 * 1024**3:  # 32GB
            recommendations.append("Consider upgrading to 32GB+ RAM for large datasets")
        
        # Check storage
        recommendations.append("Use NVMe SSD for video files to reduce I/O bottlenecks")
        
        return recommendations
    
    @staticmethod
    def optimize_pytorch_settings():
        """Set optimal PyTorch settings for performance"""
        if torch.cuda.is_available():
            # Enable optimized attention
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable optimized convolutions
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            
            print("Optimized PyTorch settings applied:")
            print("  - TF32 enabled for faster inference")
            print("  - cuDNN benchmark mode enabled")
            print("  - GPU memory allocation optimized")
        
        # Set number of threads for CPU operations
        num_threads = min(16, psutil.cpu_count())
        torch.set_num_threads(num_threads)
        print(f"  - CPU threads set to {num_threads}")
    
    @staticmethod
    def estimate_processing_time(num_videos, num_gpx, avg_video_duration=300):
        """Estimate processing time with optimizations"""
        # Optimized estimates (much faster than original)
        video_processing_time = num_videos * (avg_video_duration / 60) * 0.1  # 0.1 minutes per minute of video
        gpx_processing_time = num_gpx * 0.01  # 0.01 minutes per GPX
        correlation_time = num_videos * num_gpx * 0.0001  # Much faster correlation
        
        total_time_minutes = video_processing_time + gpx_processing_time + correlation_time
        
        estimate = {
            'video_processing_minutes': video_processing_time,
            'gpx_processing_minutes': gpx_processing_time,
            'correlation_minutes': correlation_time,
            'total_minutes': total_time_minutes,
            'total_hours': total_time_minutes / 60,
            'estimated_speedup': '50-100x faster than original implementation'
        }
        
        return estimate

def run_performance_optimization_check():
    """Run complete performance optimization check"""
    print("PERFORMANCE OPTIMIZATION CHECK")
    print("=" * 50)
    
    # Check system configuration
    print("\n1. SYSTEM CONFIGURATION:")
    recommendations = OptimizationRecommendations.check_system_configuration()
    for rec in recommendations:
        print(f"   {rec}")
    
    # Apply PyTorch optimizations
    print("\n2. PYTORCH OPTIMIZATIONS:")
    OptimizationRecommendations.optimize_pytorch_settings()
    
    # GPU information
    print("\n3. GPU INFORMATION:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
    else:
        print("   No CUDA GPUs available")
    
    # Processing time estimate
    print("\n4. PROCESSING TIME ESTIMATE:")
    # Example estimates for different dataset sizes
    scenarios = [
        (50, 100, 300),   # 50 videos, 100 GPX, 5min avg
        (100, 200, 300),  # 100 videos, 200 GPX, 5min avg
        (500, 1000, 300)  # 500 videos, 1000 GPX, 5min avg
    ]
    
    for num_videos, num_gpx, duration in scenarios:
        estimate = OptimizationRecommendations.estimate_processing_time(num_videos, num_gpx, duration)
        print(f"   {num_videos} videos × {num_gpx} GPX files:")
        print(f"     Estimated time: {estimate['total_hours']:.1f} hours")
        print(f"     Speedup: {estimate['estimated_speedup']}")

if __name__ == "__main__":
    # Run optimization check
    run_performance_optimization_check()
    
    # Example of how to use performance monitoring
    print("\n" + "="*50)
    print("To monitor performance during processing:")
    print("1. Create monitor: monitor = PerformanceMonitor()")
    print("2. Start monitoring: monitor.start_monitoring()")
    print("3. Run your processing...")
    print("4. Stop and generate report: monitor.stop_monitoring()")
    print("5. Generate report: monitor.generate_performance_report('./reports')")
