#!/usr/bin/env python3
"""
Test all possible CUDA scaling methods to find what works
"""
import subprocess
import json

class CUDAScalingTester:
    """Test all possible CUDA scaling approaches"""
    
    def __init__(self):
        self.working_methods = []
        self.test_results = {}
    
    def test_all_cuda_scaling_methods(self):
        """Test every possible CUDA scaling method"""
        
        print("🔍 TESTING ALL CUDA SCALING METHODS")
        print("=" * 60)
        
        # First, check what CUDA filters are available
        self._check_available_cuda_filters()
        
        # Test methods in order of preference
        methods = [
            ("scale_cuda", "scale_cuda=640:360"),
            ("scale_npp", "scale_npp=640:360"), 
            ("hwupload_scale_hwdownload", "hwupload_cuda,scale_cuda=640:360,hwdownload"),
            ("format_scale_cuda", "format=nv12|cuda,scale_cuda=640:360"),
            ("scale_cuda_alt_syntax", "scale_cuda=w=640:h=360"),
            ("scale_npp_alt_syntax", "scale_npp=w=640:h=360"),
            ("hwupload_scale_npp", "hwupload_cuda,scale_npp=640:360,hwdownload"),
            ("yadif_cuda_scale", "yadif_cuda=0:-1:1,scale_cuda=640:360"),
            ("scale_cuda_format", "scale_cuda=640:360,format=rgb24"),
            ("hwmap_scale", "hwmap=derive_device=cuda,scale_cuda=640:360,hwmap=reverse=1"),
            ("nvresize", "nvresize=640:360"),  # Alternative NVIDIA filter
        ]
        
        for method_name, filter_string in methods:
            success = self._test_cuda_scaling_method(method_name, filter_string)
            if success:
                self.working_methods.append((method_name, filter_string))
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 CUDA SCALING TEST RESULTS")
        print("=" * 60)
        
        if self.working_methods:
            print(f"✅ Found {len(self.working_methods)} working CUDA scaling method(s):")
            for method_name, filter_string in self.working_methods:
                print(f"  ✅ {method_name}: {filter_string}")
        else:
            print("❌ No CUDA scaling methods work")
            print("💡 Recommendation: Use CUDA decode + CPU scale (still very fast)")
        
        return self.working_methods
    
    def _check_available_cuda_filters(self):
        """Check what CUDA filters are available in FFmpeg"""
        print("\n🔍 Checking available CUDA filters...")
        
        try:
            cmd = ['ffmpeg', '-filters']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                cuda_filters = []
                for line in result.stdout.split('\n'):
                    if 'cuda' in line.lower():
                        cuda_filters.append(line.strip())
                
                if cuda_filters:
                    print(f"✅ Found {len(cuda_filters)} CUDA filters:")
                    for filter_line in cuda_filters:
                        print(f"  {filter_line}")
                else:
                    print("❌ No CUDA filters found")
            else:
                print("❌ Could not list filters")
                
        except Exception as e:
            print(f"❌ Error checking filters: {e}")
    
    def _test_cuda_scaling_method(self, method_name, filter_string):
        """Test a specific CUDA scaling method"""
        print(f"\n🧪 Testing: {method_name}")
        print(f"   Filter: {filter_string}")
        
        # Create test command
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-f', 'lavfi',
            '-i', 'testsrc=duration=1:size=1280x720:rate=1',
            '-vf', filter_string,
            '-f', 'null',
            '-',
            '-loglevel', 'error'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print(f"   ✅ SUCCESS!")
                self.test_results[method_name] = "SUCCESS"
                return True
            else:
                error_msg = result.stderr.strip()
                print(f"   ❌ FAILED: {error_msg}")
                self.test_results[method_name] = f"FAILED: {error_msg}"
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ TIMEOUT")
            self.test_results[method_name] = "TIMEOUT"
            return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            self.test_results[method_name] = f"ERROR: {e}"
            return False
    
    def test_with_real_video(self, video_path):
        """Test working methods with a real video"""
        if not self.working_methods:
            print("❌ No working CUDA scaling methods to test")
            return None
        
        print(f"\n🎬 TESTING WITH REAL VIDEO: {video_path}")
        print("=" * 60)
        
        best_method = None
        
        for method_name, filter_string in self.working_methods:
            print(f"\n🧪 Testing {method_name} with real video...")
            
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', '0',
                '-i', video_path,
                '-vf', filter_string,
                '-t', '5',  # Test first 5 seconds
                '-f', 'null',
                '-',
                '-loglevel', 'warning'
            ]
            
            try:
                import time
                start = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                elapsed = time.time() - start
                
                if result.returncode == 0:
                    print(f"   ✅ SUCCESS! Processed 5s in {elapsed:.2f}s")
                    if best_method is None:
                        best_method = (method_name, filter_string)
                else:
                    print(f"   ❌ FAILED: {result.stderr}")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
        
        if best_method:
            print(f"\n🎯 BEST CUDA SCALING METHOD: {best_method[0]}")
            print(f"   Filter: {best_method[1]}")
            return best_method
        else:
            print("\n❌ No CUDA scaling methods work with real video")
            return None


def main():
    """Main test function"""
    import sys
    
    tester = CUDAScalingTester()
    
    # Test all methods
    working_methods = tester.test_all_cuda_scaling_methods()
    
    # Test with real video if provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        best_method = tester.test_with_real_video(video_path)
        
        if best_method:
            print(f"\n🚀 RECOMMENDED CUDA SCALING METHOD:")
            print(f"   Method: {best_method[0]}")
            print(f"   Filter: {best_method[1]}")
            
            # Generate code for the working method
            print(f"\n💻 USE THIS IN YOUR DECODER:")
            print(f"   '-vf', '{best_method[1]}'")
        else:
            print(f"\n💡 RECOMMENDATION: Use CUDA decode + CPU scale")
            print(f"   Still much faster than pure CPU!")
    else:
        print(f"\nUsage: python cuda_scaling_tester.py /path/to/video.mp4")
        print(f"       (to test with your actual video)")


if __name__ == "__main__":
    main()