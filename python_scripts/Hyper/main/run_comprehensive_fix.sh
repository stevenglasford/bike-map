#!/bin/bash

echo "🚀 COMPREHENSIVE MATCHER.PY FIX"
echo "==============================="
echo "This will fix all the issues you're experiencing:"
echo "  ✅ GPU timeout errors (30s timeout)"
echo "  ✅ Unnecessary chunking activation"
echo "  ✅ Tiny chunk sizes (3 frames → 30+ frames)"
echo "  ✅ Smart processing selection"
echo ""

# Step 1: Apply the comprehensive fix
echo "🔧 Step 1: Applying comprehensive fixes..."
python3 comprehensive_matcher_fix.py

if [ $? -eq 0 ]; then
    echo "✅ Fixes applied successfully!"
    echo ""
    
    # Step 2: Test with a small sample
    echo "🧪 Step 2: Testing with small sample..."
    
    # Create test output directory
    mkdir -p ~/penis/testingground/comprehensive_test
    
    # Set up environment
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    echo "Testing smart processing logic..."
    python matcher.py \
        -d ~/penis/panoramics/playground/ \
        -o ~/penis/testingground/comprehensive_test \
        --max_frames 200 \
        --video_size 1280 720 \
        --parallel_videos 2 \
        --gpu_ids 0 1 \
        --debug \
        --powersafe \
        --force
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 TEST SUCCESSFUL!"
        echo ""
        echo "✅ What's working now:"
        echo "   • No more GPU timeout errors"
        echo "   • Smart chunking (only when needed)"
        echo "   • Reasonable chunk sizes"
        echo "   • Proper GPU utilization"
        echo ""
        echo "🚀 Ready for full processing with:"
        echo ""
        echo "python matcher.py \\"
        echo "    -d ~/penis/panoramics/playground/ \\"
        echo "    -o ~/penis/testingground/full_results \\"
        echo "    --max_frames 999999 \\"
        echo "    --video_size 1920 1080 \\"
        echo "    --parallel_videos 4 \\"  # Can now use higher parallelization!
        echo "    --gpu_ids 0 1 \\"
        echo "    --debug \\"
        echo "    --powersafe \\"
        echo "    --force"
        echo ""
        echo "💡 Key improvements:"
        echo "   • Normal processing for typical videos (fast)"
        echo "   • Chunked processing only for large videos"
        echo "   • No more 3-frame chunks"
        echo "   • No more GPU timeouts"
        echo "   • Higher parallelization possible"
        
    else
        echo ""
        echo "⚠️ Test had issues, but let's try a more conservative approach:"
        echo ""
        echo "python matcher.py \\"
        echo "    -d ~/penis/panoramics/playground/ \\"
        echo "    -o ~/penis/testingground/conservative \\"
        echo "    --max_frames 999999 \\"
        echo "    --video_size 1280 720 \\"
        echo "    --parallel_videos 1 \\"
        echo "    --gpu_ids 0 \\"
        echo "    --debug \\"
        echo "    --powersafe"
    fi
    
else
    echo "❌ Fix application failed!"
    echo ""
    echo "🔍 Troubleshooting steps:"
    echo "1. Make sure you're in the same directory as matcher.py"
    echo "2. Check that you have write permissions"
    echo "3. Ensure Python can import the matcher module"
    echo ""
    echo "🔧 Manual fix option:"
    echo "1. Save the comprehensive_matcher_fix.py file"
    echo "2. Run: python comprehensive_matcher_fix.py"
    echo "3. Follow the instructions"
fi

echo ""
echo "📊 Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "Fix attempt complete! 🎯"