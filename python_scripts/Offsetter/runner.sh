#!/bin/bash

# Fixed Multi-GPU Runner Script
# Properly isolates GPU contexts and distributes work

echo "🚀 Starting Multi-GPU Processing..."
echo "=================================="

INPUT_FILE="../Visualizer/MatcherFiles/complete_turbo_360_report_ramcache.json"
WORKERS=5  # Reduced for stability
GPU_MEMORY=14.0
TOP_MATCHES=12

# Verify input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Input file not found: $INPUT_FILE"
    exit 1
fi

echo "📁 Input: $INPUT_FILE"
echo "🎮 GPUs: 0, 1"
echo "👥 Workers: $WORKERS per GPU"
echo "💾 Memory: ${GPU_MEMORY}GB per GPU"
echo "🎯 Top matches: $TOP_MATCHES"
echo ""

# Clean up any existing output files
rm -f gpu0.json gpu1.json out00.txt out01.txt

# Start GPU 0 process (processes even-indexed matches: 0, 2, 4, 6...)
echo "🔥 Starting GPU 0 process..."
CUDA_VISIBLE_DEVICES=0 python offsetter27.py \
    "$INPUT_FILE" \
    -o gpu0.json \
    --gpu-id 0 \
    --gpu-index 0 \
    --workers $WORKERS \
    --do-multiple-of 2 \
    --gpu-memory $GPU_MEMORY \
    --top-matches $TOP_MATCHES \
    --debug \
    --gpu-debug > out00.txt 2>&1 &

GPU0_PID=$!
echo "✅ GPU 0 process started (PID: $GPU0_PID)"

# Small delay to avoid initialization conflicts
sleep 2

# Start GPU 1 process (processes odd-indexed matches: 1, 3, 5, 7...)
echo "🔥 Starting GPU 1 process..."
CUDA_VISIBLE_DEVICES=1 python offsetter27.py \
    "$INPUT_FILE" \
    -o gpu1.json \
    --gpu-id 0 \
    --gpu-index 1 \
    --workers $WORKERS \
    --do-multiple-of 2 \
    --gpu-memory $GPU_MEMORY \
    --top-matches $TOP_MATCHES \
    --debug \
    --gpu-debug > out01.txt 2>&1 &

GPU1_PID=$!
echo "✅ GPU 1 process started (PID: $GPU1_PID)"

echo ""
echo "⏳ Waiting for both processes to complete..."
echo "💡 Monitor progress with: tail -f out00.txt out01.txt"
echo ""

# Wait for both processes
wait $GPU0_PID
GPU0_EXIT=$?

wait $GPU1_PID  
GPU1_EXIT=$?

echo ""
echo "🎉 Processing Complete!"
echo "======================"

# Check results
if [[ $GPU0_EXIT -eq 0 ]]; then
    echo "✅ GPU 0 process completed successfully"
    if [[ -f "gpu0.json" ]]; then
        SIZE0=$(stat -c%s gpu0.json)
        echo "   📁 gpu0.json created (${SIZE0} bytes)"
    else
        echo "   ⚠️  gpu0.json not found"
    fi
else
    echo "❌ GPU 0 process failed (exit code: $GPU0_EXIT)"
fi

if [[ $GPU1_EXIT -eq 0 ]]; then
    echo "✅ GPU 1 process completed successfully" 
    if [[ -f "gpu1.json" ]]; then
        SIZE1=$(stat -c%s gpu1.json)
        echo "   📁 gpu1.json created (${SIZE1} bytes)"
    else
        echo "   ⚠️  gpu1.json not found"
    fi
else
    echo "❌ GPU 1 process failed (exit code: $GPU1_EXIT)"
fi

# Show output file locations
echo ""
echo "📁 Output Files:"
[[ -f "gpu0.json" ]] && echo "   gpu0.json"
[[ -f "gpu1.json" ]] && echo "   gpu1.json"

echo ""
echo "📋 Log Files:"
echo "   out00.txt (GPU 0 log)"
echo "   out01.txt (GPU 1 log)"


# Exit with error if either process failed
if [[ $GPU0_EXIT -ne 0 ]] || [[ $GPU1_EXIT -ne 0 ]]; then
    echo ""
    echo "❌ At least one process failed. Check log files for details."
    exit 1
else
    echo ""
    echo "🎉 All processes completed successfully!"
    exit 0
fi