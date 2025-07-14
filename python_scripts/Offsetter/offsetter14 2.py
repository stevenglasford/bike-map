#!/bin/bash
# 🚀 PRODUCTION DUAL GPU RUNNER - FULL PROCESSING 🚀
# Processes ALL videos without limits - designed for 115+ video processing

echo "🚀💀🚀 PRODUCTION DUAL GPU BEAST MODE 🚀💀🚀"
echo "=============================================="
echo "💀 PROCESSES ALL VIDEOS WITHOUT LIMITS"
echo "⚡ OPTIMIZED FOR MAXIMUM THROUGHPUT"
echo "🔥 DUAL GPU MAXIMUM UTILIZATION"
echo "🚀 PRODUCTION-READY PROCESSING"
echo "=============================================="

# Configuration
INPUT_FILE="../Visualizer/MatcherFiles/complete_turbo_360_report_ramcache.json"
OUTPUT_FILE="production_offset_results_$(date +%Y%m%d_%H%M%S).json"
LOG_FILE="production_processing_$(date +%Y%m%d_%H%M%S).log"

# Processing parameters
WORKERS_PER_GPU=3           # 3 workers per GPU for optimal throughput
GPU_MEMORY=15.0             # 15GB per GPU (adjust if needed)
MIN_SCORE=0.3              # Minimum match score threshold
PYTHON_SCRIPT="offsetter13.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to get total matches
get_total_matches() {
    python3 -c "
import json
try:
    with open('$INPUT_FILE', 'r') as f:
        data = json.load(f)
    
    total_matches = 0
    valid_matches = 0
    
    for video_path, video_data in data.get('results', {}).items():
        for match in video_data.get('matches', []):
            total_matches += 1
            if match.get('combined_score', 0) >= $MIN_SCORE:
                valid_matches += 1
    
    print(f'{total_matches},{valid_matches}')
except Exception as e:
    print('0,0')
"
}

# Function to estimate processing time
estimate_time() {
    local matches=$1
    local rate=1.5  # Conservative estimate: 1.5 matches/second
    local total_seconds=$((matches / rate))
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# Cleanup function
production_cleanup() {
    echo ""
    print_header "🧹 PRODUCTION CLEANUP..."
    
    # Kill any remaining processes
    pkill -f "$PYTHON_SCRIPT" 2>/dev/null
    pkill -f "production.*gpu" 2>/dev/null
    
    # GPU cleanup
    python3 -c "
try:
    import cupy as cp
    print('🔧 Cleaning GPU memory...')
    for gpu_id in [0, 1]:
        try:
            cp.cuda.Device(gpu_id).use()
            mempool = cp.get_default_memory_pool()
            used_before = mempool.used_bytes() / (1024**3)
            mempool.free_all_blocks()
            cp.cuda.Device().synchronize()
            if used_before > 0.1:
                print(f'🧹 GPU {gpu_id}: Freed {used_before:.1f}GB')
        except Exception as e:
            print(f'⚠️ GPU {gpu_id}: {e}')
    print('✅ GPU cleanup complete')
except ImportError:
    print('⚠️ CuPy not available for cleanup')
except Exception as e:
    print(f'⚠️ Cleanup error: {e}')
"
    
    print_success "Production cleanup complete"
}

trap production_cleanup EXIT

# Initial checks
print_header "🔍 INITIAL SYSTEM CHECKS"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file not found: $INPUT_FILE"
    exit 1
fi
print_success "Input file found: $INPUT_FILE"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi
print_success "Python script found: $PYTHON_SCRIPT"

# Check Python dependencies
print_info "Checking Python dependencies..."
python3 -c "
import sys
missing = []
try:
    import cupy
except ImportError:
    missing.append('cupy-cuda12x')
try:
    import cv2
except ImportError:
    missing.append('opencv-contrib-python-headless')
try:
    import gpxpy
except ImportError:
    missing.append('gpxpy')
try:
    import pandas
except ImportError:
    missing.append('pandas')

if missing:
    print('❌ Missing dependencies:', ', '.join(missing))
    print('Install with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✅ All dependencies available')
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
print_header "🎮 GPU STATUS"

# Initial GPU status
echo "Initial GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx name free total util temp; do
    echo "  GPU $idx: $name | Memory: ${free}MB/${total}MB free | Util: ${util}% | Temp: ${temp}°C"
done

# Check for competing processes
echo ""
print_info "Checking for competing GPU processes..."
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    print_warning "Found $GPU_PROCS competing GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
    echo ""
    print_warning "Consider killing them for maximum performance:"
    echo "   nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs sudo kill"
    echo ""
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "No competing GPU processes found"
fi

echo ""
print_header "📊 MATCH ANALYSIS"

# Get match counts
print_info "Analyzing matches in input file..."
MATCH_INFO=$(get_total_matches)
TOTAL_MATCHES=$(echo $MATCH_INFO | cut -d',' -f1)
VALID_MATCHES=$(echo $MATCH_INFO | cut -d',' -f2)

if [ "$TOTAL_MATCHES" -eq 0 ]; then
    print_error "No matches found in input file"
    exit 1
fi

print_success "Total matches found: $TOTAL_MATCHES"
print_success "Valid matches (score ≥ $MIN_SCORE): $VALID_MATCHES"

if [ "$VALID_MATCHES" -eq 0 ]; then
    print_error "No valid matches found (all below score threshold $MIN_SCORE)"
    exit 1
fi

# Estimate processing time
ESTIMATED_TIME=$(estimate_time $VALID_MATCHES)
print_info "Estimated processing time: $ESTIMATED_TIME"

echo ""
print_header "🚀 PRODUCTION PROCESSING CONFIGURATION"

echo "Configuration:"
echo "  📁 Input file: $INPUT_FILE"
echo "  📄 Output file: $OUTPUT_FILE"
echo "  📝 Log file: $LOG_FILE"
echo "  🎯 Matches to process: $VALID_MATCHES"
echo "  🏭 Workers per GPU: $WORKERS_PER_GPU"
echo "  💾 GPU memory limit: ${GPU_MEMORY}GB per GPU"
echo "  📊 Minimum score: $MIN_SCORE"
echo "  ⏱️  Estimated time: $ESTIMATED_TIME"

echo ""
print_warning "This will process ALL $VALID_MATCHES matches. Continue? (y/n)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 1
fi

echo ""
print_header "🚀 STARTING PRODUCTION PROCESSING"

# Record start time
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date)

print_success "Production processing started at: $START_TIME_HUMAN"
print_info "Processing $VALID_MATCHES matches with $WORKERS_PER_GPU workers per GPU..."

# Start the main processing
python3 "$PYTHON_SCRIPT" "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --workers-per-gpu $WORKERS_PER_GPU \
    --gpu-memory $GPU_MEMORY \
    --min-score $MIN_SCORE \
    --beast-mode 2>&1 | tee "$LOG_FILE"

PROCESSING_EXIT_CODE=$?
END_TIME=$(date +%s)
END_TIME_HUMAN=$(date)
TOTAL_PROCESSING_TIME=$((END_TIME - START_TIME))

echo ""
print_header "📈 PRODUCTION PROCESSING RESULTS"

if [ $PROCESSING_EXIT_CODE -eq 0 ]; then
    print_success "Production processing completed successfully!"
    
    if [ -f "$OUTPUT_FILE" ]; then
        # Analyze results
        print_info "Analyzing processing results..."
        
        ANALYSIS=$(python3 -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    # Count results
    gpu0_count = 0
    gpu1_count = 0
    success_count = 0
    total_processed = 0
    error_count = 0
    
    for video_path, video_data in data.get('results', {}).items():
        for match in video_data.get('matches', []):
            if 'gpu_id' in match:
                total_processed += 1
                if match.get('gpu_id') == 0:
                    gpu0_count += 1
                elif match.get('gpu_id') == 1:
                    gpu1_count += 1
                
                if match.get('temporal_offset_seconds') is not None:
                    success_count += 1
                
                if 'error' in match.get('offset_method', ''):
                    error_count += 1
    
    # Get processing info
    proc_info = data.get('optimized_processing_info', {}) or data.get('robust_processing_info', {})
    proc_time = proc_info.get('processing_time_seconds', 0)
    proc_rate = proc_info.get('processing_rate_matches_per_second', 0)
    success_rate = proc_info.get('success_rate', 0) * 100
    
    print(f'{total_processed},{gpu0_count},{gpu1_count},{success_count},{error_count},{proc_time:.1f},{proc_rate:.2f},{success_rate:.1f}')
    
except Exception as e:
    print(f'0,0,0,0,0,0,0,0')
")
        
        IFS=',' read -r PROCESSED GPU0_COUNT GPU1_COUNT SUCCESS_COUNT ERROR_COUNT PROC_TIME PROC_RATE SUCCESS_RATE <<< "$ANALYSIS"
        
        echo ""
        print_header "📊 DETAILED RESULTS"
        echo "Processing Statistics:"
        echo "  📋 Total processed: $PROCESSED matches"
        echo "  ✅ Successful offsets: $SUCCESS_COUNT"
        echo "  ❌ Errors: $ERROR_COUNT"
        echo "  📈 Success rate: ${SUCCESS_RATE}%"
        echo ""
        echo "GPU Utilization:"
        echo "  🔥 GPU 0 processed: $GPU0_COUNT matches"
        echo "  🔥 GPU 1 processed: $GPU1_COUNT matches"
        
        if [ "$GPU0_COUNT" -gt 0 ] && [ "$GPU1_COUNT" -gt 0 ]; then
            print_success "Both GPUs utilized successfully!"
            GPU_BALANCE=$(python3 -c "print(f'{min($GPU0_COUNT, $GPU1_COUNT) / max($GPU0_COUNT, $GPU1_COUNT) * 100:.1f}')")
            echo "  ⚖️  GPU load balance: ${GPU_BALANCE}%"
        else
            print_warning "GPU utilization issue - only one GPU processed matches"
        fi
        
        echo ""
        echo "Performance Metrics:"
        echo "  ⏱️  Total processing time: ${TOTAL_PROCESSING_TIME}s ($(($TOTAL_PROCESSING_TIME / 60))m $(($TOTAL_PROCESSING_TIME % 60))s)"
        echo "  🚀 Processing rate: ${PROC_RATE} matches/second"
        
        if (( $(echo "$PROC_RATE >= 1.0" | bc -l) )); then
            print_success "Excellent processing rate achieved!"
        elif (( $(echo "$PROC_RATE >= 0.5" | bc -l) )); then
            print_success "Good processing rate achieved!"
        else
            print_warning "Processing rate below optimal"
        fi
        
        echo ""
        echo "Output Files:"
        echo "  📄 Results: $OUTPUT_FILE"
        echo "  📝 Log: $LOG_FILE"
        
        # File sizes
        RESULT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
        echo "  💾 Result file size: $RESULT_SIZE"
        echo "  📊 Log file size: $LOG_SIZE"
        
        echo ""
        print_header "🎉 PRODUCTION PROCESSING COMPLETE!"
        
        if [ "$SUCCESS_COUNT" -gt 0 ]; then
            COMPLETION_RATE=$(python3 -c "print(f'{$SUCCESS_COUNT / $VALID_MATCHES * 100:.1f}')")
            print_success "Successfully processed $SUCCESS_COUNT out of $VALID_MATCHES matches (${COMPLETION_RATE}%)"
            
            if (( $(echo "$COMPLETION_RATE >= 90" | bc -l) )); then
                print_success "OUTSTANDING SUCCESS RATE! 🎉"
            elif (( $(echo "$COMPLETION_RATE >= 75" | bc -l) )); then
                print_success "EXCELLENT SUCCESS RATE! 🎉"
            elif (( $(echo "$COMPLETION_RATE >= 50" | bc -l) )); then
                print_success "GOOD SUCCESS RATE! 👍"
            else
                print_warning "Success rate could be improved"
            fi
        else
            print_error "No successful matches processed"
        fi
        
    else
        print_error "Output file not created: $OUTPUT_FILE"
    fi
    
else
    print_error "Production processing failed with exit code: $PROCESSING_EXIT_CODE"
    
    echo ""
    print_header "🔧 TROUBLESHOOTING INFORMATION"
    echo "Check the following:"
    echo "  1. Review the log file: $LOG_FILE"
    echo "  2. Check GPU memory availability: nvidia-smi"
    echo "  3. Verify input file format and content"
    echo "  4. Check for system resource constraints"
    echo "  5. Try reducing workers per GPU (--workers-per-gpu 2)"
    echo "  6. Try reducing GPU memory limit (--gpu-memory 12.0)"
fi

echo ""
print_header "🎮 FINAL GPU STATUS"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx name used free util temp; do
    echo "  GPU $idx: $name | Memory: ${used}MB used, ${free}MB free | Util: ${util}% | Temp: ${temp}°C"
done

echo ""
print_header "📋 PRODUCTION SESSION SUMMARY"
echo "Session Information:"
echo "  🕐 Started: $START_TIME_HUMAN"
echo "  🕐 Ended: $END_TIME_HUMAN"
echo "  ⏱️  Total time: ${TOTAL_PROCESSING_TIME}s ($(($TOTAL_PROCESSING_TIME / 60))m)"
echo "  🎯 Target matches: $VALID_MATCHES"
echo "  ✅ Processed successfully: $SUCCESS_COUNT"
echo "  📁 Output: $OUTPUT_FILE"
echo "  📝 Log: $LOG_FILE"

if [ $PROCESSING_EXIT_CODE -eq 0 ] && [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo ""
    print_success "🎉 PRODUCTION PROCESSING MISSION ACCOMPLISHED! 🎉"
    print_success "Results ready for use in: $OUTPUT_FILE"
else
    echo ""
    print_error "❌ PRODUCTION PROCESSING INCOMPLETE"
    print_info "Check logs and troubleshooting information above"
fi

echo ""
echo "🚀💀🚀 PRODUCTION DUAL GPU SESSION COMPLETE 🚀💀🚀"