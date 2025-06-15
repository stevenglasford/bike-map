#!/bin/bash

echo "🔧 GPU Video Processing Diagnostic Script"
echo "=========================================="

# Create required directories
echo "📁 Setting up directories..."
mkdir -p ~/penis/temp/chunks
mkdir -p ~/penis/temp/gpu_temp
mkdir -p ~/penis/temp/processing
mkdir -p ~/penis/testingground/diagnostic_results

# Set environment variables for better CUDA debugging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "🔍 Checking GPU status..."
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "PyTorch CUDA detection:"
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)')
        try:
            with torch.cuda.device(i):
                test = torch.zeros(10, 10, device=f'cuda:{i}')
                print(f'    GPU {i}: Basic tensor operations ✅')
                del test
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'    GPU {i}: Failed basic test ❌ - {e}')
"

echo ""
echo "🧪 Running diagnostic tests..."

# Test 1: Single GPU, minimal processing
echo "Test 1: Single GPU (GPU 0), ultra-minimal processing"
python matcher.py \
    -d ~/penis/panoramics/playground/ \
    -o ~/penis/testingground/diagnostic_results/test1 \
    --max_frames 50 \
    --video_size 480 270 \
    --parallel_videos 1 \
    --gpu_ids 0 \
    --debug \
    --validation_only

if [ $? -eq 0 ]; then
    echo "✅ Test 1 (validation) passed"
    
    echo "Test 1b: Actual processing with single GPU"
    python matcher.py \
        -d ~/penis/panoramics/playground/ \
        -o ~/penis/testingground/diagnostic_results/test1b \
        --max_frames 50 \
        --video_size 480 270 \
        --parallel_videos 1 \
        --gpu_ids 0 \
        --debug \
        --powersafe
    
    if [ $? -eq 0 ]; then
        echo "✅ Test 1b (single GPU processing) passed"
        
        # Test 2: Try both GPUs
        echo "Test 2: Both GPUs, conservative processing"
        python matcher.py \
            -d ~/penis/panoramics/playground/ \
            -o ~/penis/testingground/diagnostic_results/test2 \
            --max_frames 100 \
            --video_size 720 480 \
            --parallel_videos 1 \
            --gpu_ids 0 1 \
            --debug \
            --powersafe
        
        if [ $? -eq 0 ]; then
            echo "✅ Test 2 (dual GPU) passed"
            
            # Test 3: Higher resolution
            echo "Test 3: Higher resolution, single GPU"
            python matcher.py \
                -d ~/penis/panoramics/playground/ \
                -o ~/penis/testingground/diagnostic_results/test3 \
                --max_frames 200 \
                --video_size 1920 1080 \
                --parallel_videos 1 \
                --gpu_ids 0 \
                --debug \
                --powersafe
            
            if [ $? -eq 0 ]; then
                echo "✅ Test 3 (1080p) passed"
                
                # Test 4: Your original settings but conservative
                echo "Test 4: Conservative version of original settings"
                python matcher.py \
                    -d ~/penis/panoramics/playground/ \
                    -o ~/penis/testingground/diagnostic_results/test4 \
                    --max_frames 999999 \
                    --video_size 1920 1080 \
                    --sample_rate 1.0 \
                    --parallel_videos 1 \
                    --gpu_ids 0 \
                    --debug \
                    --powersafe \
                    --enable_preprocessing \
                    --ram_cache 48.0
                
                if [ $? -eq 0 ]; then
                    echo "✅ Test 4 (conservative full processing) passed"
                    echo "🎉 All tests passed! Your system is working well."
                    echo ""
                    echo "✅ RECOMMENDED SETTINGS FOR YOUR SYSTEM:"
                    echo "   --parallel_videos 1"
                    echo "   --gpu_ids 0 1"
                    echo "   --video_size 1920 1080 (or higher if test 4 worked)"
                    echo "   --powersafe (always recommended)"
                    echo "   --debug (for monitoring)"
                else
                    echo "❌ Test 4 failed - use 1080p max resolution"
                fi
            else
                echo "❌ Test 3 failed - use 720p max resolution"
            fi
        else
            echo "❌ Test 2 failed - use single GPU only"
            echo "🔧 RECOMMENDED: Only use --gpu_ids 0"
        fi
    else
        echo "❌ Test 1b failed - GPU processing has issues"
        echo "🔧 Try CPU fallback or check GPU drivers"
    fi
else
    echo "❌ Test 1 failed - basic video validation failed"
    echo "🔧 Check video files and FFmpeg installation"
fi

echo ""
echo "🔧 TROUBLESHOOTING RECOMMENDATIONS:"
echo "=================================="

if [ -f ~/penis/testingground/diagnostic_results/test1/video_validation_report.json ]; then
    echo "✅ Video validation report created - check for corrupted files"
else
    echo "❌ No validation report - FFmpeg or video files may have issues"
fi

echo ""
echo "💡 If you're getting CUDA errors:"
echo "   1. Restart your system to clear GPU memory"
echo "   2. Use only one GPU: --gpu_ids 0"
echo "   3. Reduce resolution: --video_size 1280 720"
echo "   4. Reduce parallel processing: --parallel_videos 1"
echo "   5. Enable powersafe mode: --powersafe"
echo ""
echo "💡 If GPU 1 keeps failing:"
echo "   nvidia-smi -r -i 1  # Reset GPU 1"
echo "   Or disable GPU 1 and use only: --gpu_ids 0"
echo ""
echo "💡 To monitor during processing:"
echo "   watch -n 1 nvidia-smi"
echo "   tail -f production_correlation.log"
echo ""
echo "🎯 Based on your error, try this next:"
echo "python matcher.py \\"
echo "    -d ~/penis/panoramics/playground/ \\"
echo "    -o ~/penis/testingground/safe_results \\"
echo "    --max_frames 999999 \\"
echo "    --video_size 1280 720 \\"
echo "    --parallel_videos 1 \\"
echo "    --gpu_ids 0 \\"
echo "    --debug \\"
echo "    --powersafe \\"
echo "    --enable_preprocessing \\"
echo "    --ram_cache 32.0"

echo ""
echo "Diagnostic complete!"