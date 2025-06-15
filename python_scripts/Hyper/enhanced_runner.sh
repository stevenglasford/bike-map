#!/bin/bash

# Enhanced Video-GPX Correlation Runner
# This script provides multiple accuracy improvement strategies

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Directory paths
DATA_DIR="~/penis/panoramics/playground/"
OUTPUT_DIR="~/penis/testingground"
TEMP_DIR="~/penis/temp"

echo "=== Enhanced Video-GPX Correlation System ==="
echo "Starting accuracy-optimized processing..."

# Strategy 1: High-Resolution Feature Extraction
echo "Strategy 1: High-resolution feature extraction..."
python matcher39.py \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/high_res" \
    --max_frames 300 \
    --video_size 720 405 \
    --sample_rate 2.0 \
    --parallel_videos 1 \
    --gpu_ids 0 1 \
    --debug \
    --strict \
    --powersafe \
    --save_interval 3 \
    --enable_preprocessing \
    --ram_cache 48.0 \
    --gpu_timeout 120

echo "Strategy 1 completed. Results in ${OUTPUT_DIR}/high_res"

# Strategy 2: Dense Temporal Sampling
echo "Strategy 2: Dense temporal sampling..."
python matcher39.py \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/dense_temporal" \
    --max_frames 500 \
    --video_size 480 270 \
    --sample_rate 1.5 \
    --parallel_videos 1 \
    --gpu_ids 0 1 \
    --debug \
    --strict \
    --powersafe \
    --save_interval 5 \
    --enable_preprocessing \
    --ram_cache 64.0 \
    --gpu_timeout 180

echo "Strategy 2 completed. Results in ${OUTPUT_DIR}/dense_temporal"

# Strategy 3: Ultra-High Quality (Best Results)
echo "Strategy 3: Ultra-high quality processing..."
python matcher39.py \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/ultra_quality" \
    --max_frames 400 \
    --video_size 960 540 \
    --sample_rate 1.0 \
    --parallel_videos 1 \
    --gpu_ids 0 1 \
    --debug \
    --strict \
    --powersafe \
    --save_interval 2 \
    --enable_preprocessing \
    --ram_cache 80.0 \
    --disk_cache 2000.0 \
    --gpu_timeout 300 \
    --max_gpu_memory 16.0

echo "Strategy 3 completed. Results in ${OUTPUT_DIR}/ultra_quality"

# Strategy 4: Memory-Optimized for Large Datasets
echo "Strategy 4: Memory-optimized processing..."
python matcher39.py \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/memory_optimized" \
    --max_frames 250 \
    --video_size 640 360 \
    --sample_rate 2.5 \
    --parallel_videos 1 \
    --gpu_ids 0 1 \
    --debug \
    --strict \
    --powersafe \
    --save_interval 1 \
    --enable_preprocessing \
    --memory_efficient \
    --ram_cache 32.0 \
    --gpu_timeout 90

echo "Strategy 4 completed. Results in ${OUTPUT_DIR}/memory_optimized"

# Strategy 5: Fast Preview Mode
echo "Strategy 5: Fast preview mode..."
python matcher39.py \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/fast_preview" \
    --max_frames 150 \
    --video_size 480 270 \
    --sample_rate 3.0 \
    --parallel_videos 2 \
    --gpu_ids 0 1 \
    --debug \
    --powersafe \
    --save_interval 10 \
    --gpu_timeout 60

echo "Strategy 5 completed. Results in ${OUTPUT_DIR}/fast_preview"

echo ""
echo "=== All Strategies Completed ==="
echo "Compare results in:"
echo "  - ${OUTPUT_DIR}/high_res/ (balanced quality)"
echo "  - ${OUTPUT_DIR}/dense_temporal/ (temporal focus)"
echo "  - ${OUTPUT_DIR}/ultra_quality/ (maximum quality)"
echo "  - ${OUTPUT_DIR}/memory_optimized/ (efficient)"
echo "  - ${OUTPUT_DIR}/fast_preview/ (quick test)"
echo ""
echo "Recommended: Check ultra_quality first for best correlation scores"

# Optional: Run validation-only mode to check video quality
echo ""
echo "Running video validation check..."
python matcher39.py \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/validation" \
    --validation_only \
    --strict \
    --debug

echo "Validation completed. Check validation report for video quality issues."