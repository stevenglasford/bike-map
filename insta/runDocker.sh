#!/bin/bash

# Complete Insta360 Docker Processing Script
# Processes entire camera directories with one command

set -e

echo "🎥 Insta360 Complete Directory Processor"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
INPUT_DIR="${1:-../insta360_backup}"
OUTPUT_DIR="${2:-./complete_output}"
MAX_WORKERS="${3:-2}"
CONTAINER_NAME="insta360-complete"

# Function to show usage
show_usage() {
    echo "Usage: $0 [input_directory] [output_directory] [max_workers]"
    echo
    echo "Examples:"
    echo "  $0                                    # Use defaults"
    echo "  $0 /path/to/camera ./output          # Custom paths"
    echo "  $0 /path/to/camera ./output 4        # Use 4 parallel workers"
    echo
    echo "Default values:"
    echo "  input_directory: ../insta360_backup"
    echo "  output_directory: ./complete_output"
    echo "  max_workers: 2"
    echo
    echo "Features:"
    echo "  ✅ Processes entire camera directory structure"
    echo "  ✅ Perfect timestamp extraction and organization"
    echo "  ✅ GPU acceleration with CPU fallback"
    echo "  ✅ Progress monitoring and resumable processing"
    echo "  ✅ Organized output by camera and date"
}

# Check if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Setup function
setup_docker() {
    print_status "Setting up Docker environment..."
    
    # Check if Dockerfile exists
    if [[ ! -f "Dockerfile" ]]; then
        print_error "Dockerfile not found! Please save the complete Dockerfile first."
        echo "Expected: Complete Insta360 Directory Processor Dockerfile"
        exit 1
    fi
    
    # Check for MediaSDK .deb file
    DEB_FILE=$(find . -name "libMediaSDK-dev*.deb" | head -1)
    if [[ -z "$DEB_FILE" ]]; then
        print_error "MediaSDK .deb file not found!"
        echo "Please copy the .deb file to current directory:"
        echo "cp ~/Documents/Linux_CameraSDK-*/libMediaSDK-dev-*.deb ."
        exit 1
    fi
    
    print_status "Found MediaSDK: $DEB_FILE"
    
    # Build Docker image
    print_status "Building Docker image (this may take a few minutes)..."
    docker build -t insta360-complete:latest .
    
    print_status "Docker image built successfully!"
}

# Check directories
check_directories() {
    print_status "Checking directories..."
    
    # Convert to absolute paths
    INPUT_DIR=$(realpath "$INPUT_DIR")
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
    
    if [[ ! -d "$INPUT_DIR" ]]; then
        print_error "Input directory not found: $INPUT_DIR"
        exit 1
    fi
    
    # Count videos in input
    VIDEO_COUNT=$(find "$INPUT_DIR" -name "*.insv" -o -name "*.insp" | wc -l)
    if [[ $VIDEO_COUNT -eq 0 ]]; then
        print_warning "No .insv/.insp videos found in $INPUT_DIR"
        echo "Directory contents:"
        ls -la "$INPUT_DIR"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_status "Found $VIDEO_COUNT videos to process"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"
    
    print_status "Input: $INPUT_DIR"
    print_status "Output: $OUTPUT_DIR"
    print_status "Workers: $MAX_WORKERS"
}

# Process videos
process_videos() {
    print_status "Starting video processing..."
    
    # Test GPU support - prioritize the method we know works
    GPU_FLAG=""
    print_status "Testing GPU access..."
    
    # Method 1: Legacy runtime (known to work for this user)
    if docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_status "✓ GPU working with --runtime=nvidia (legacy method)"
        GPU_FLAG="--runtime=nvidia"
    
    # Method 2: Modern Docker (fallback)
    elif docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_status "✓ GPU working with --gpus all"
        GPU_FLAG="--gpus all"
    
    else
        print_warning "✗ GPU not accessible - using CPU processing"
        print_warning "  MediaSDK will use software decoding (slower but works)"
    fi
    
    # Run the container with the working GPU method
    print_status "Launching processing container..."
    
    if [[ -n "$GPU_FLAG" ]]; then
        # With GPU access
        docker run --rm $GPU_FLAG \
            --name "$CONTAINER_NAME" \
            -v "$INPUT_DIR:/input:ro" \
            -v "$OUTPUT_DIR:/output" \
            -v "$OUTPUT_DIR/logs:/logs" \
            insta360-complete:latest \
            python3 /app/process_directory.py /input /output $MAX_WORKERS
    else
        # CPU only fallback
        docker run --rm \
            --name "$CONTAINER_NAME" \
            -v "$INPUT_DIR:/input:ro" \
            -v "$OUTPUT_DIR:/output" \
            -v "$OUTPUT_DIR/logs:/logs" \
            insta360-complete:latest \
            python3 /app/process_directory.py /input /output $MAX_WORKERS
    fi
}

# Monitor progress (background)
monitor_progress() {
    print_status "Starting progress monitor..."
    
    # Run monitoring in separate container
    docker run --rm -d \
        --name "${CONTAINER_NAME}-monitor" \
        -v "$OUTPUT_DIR/logs:/logs:ro" \
        insta360-complete:latest \
        python3 /app/monitor.py &
    
    MONITOR_PID=$!
    
    # Monitor function
    monitor_loop() {
        while true; do
            sleep 30
            
            # Check if processing container is still running
            if ! docker ps | grep -q "$CONTAINER_NAME"; then
                break
            fi
            
            # Show basic stats
            if [[ -f "$OUTPUT_DIR/logs/progress_report.json" ]]; then
                echo "--- Progress Update ---"
                jq -r '.stats | "Processed: \(.processed), Failed: \(.failed), Rate: \(.processing_rate | round) videos/hour"' "$OUTPUT_DIR/logs/progress_report.json" 2>/dev/null || echo "Processing..."
            fi
        done
        
        # Kill monitor container
        docker stop "${CONTAINER_NAME}-monitor" 2>/dev/null || true
    }
    
    monitor_loop &
}

# Show results
show_results() {
    print_status "Processing complete! Analyzing results..."
    
    # Count outputs
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" | wc -l)
    INPUT_COUNT=$(find "$INPUT_DIR" -name "*.insv" -o -name "*.insp" | wc -l)
    
    echo
    echo "📊 PROCESSING SUMMARY"
    echo "===================="
    echo "Input videos: $INPUT_COUNT"
    echo "Output videos: $OUTPUT_COUNT"
    echo "Success rate: $(( OUTPUT_COUNT * 100 / INPUT_COUNT ))%"
    echo
    
    # Show output structure
    echo "📁 OUTPUT STRUCTURE:"
    tree "$OUTPUT_DIR" -L 3 2>/dev/null || find "$OUTPUT_DIR" -type d | head -10
    
    echo
    echo "📋 RECENT OUTPUTS:"
    find "$OUTPUT_DIR" -name "*.mp4" -mmin -60 | head -5
    
    # Show storage info
    echo
    echo "💾 STORAGE INFO:"
    echo "Input size: $(du -sh "$INPUT_DIR" | cut -f1)"
    echo "Output size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
    
    # Show logs location
    echo
    echo "📄 LOGS LOCATION:"
    echo "Processing log: $OUTPUT_DIR/logs/processing.log"
    echo "Progress report: $OUTPUT_DIR/logs/progress_report.json"
    
    if [[ -f "$OUTPUT_DIR/logs/progress_report.json" ]]; then
        echo
        echo "🎯 FINAL STATISTICS:"
        jq -r '.cameras | to_entries[] | "\(.key): \(.value.processed)/\(.value.total) processed (\(.value.success_rate | round)% success)"' "$OUTPUT_DIR/logs/progress_report.json"
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker stop "${CONTAINER_NAME}-monitor" 2>/dev/null || true
}

# Trap cleanup
trap cleanup EXIT

# Main execution
main() {
    echo "Configuration:"
    echo "  Input: $INPUT_DIR"
    echo "  Output: $OUTPUT_DIR"
    echo "  Workers: $MAX_WORKERS"
    echo
    
    read -p "Continue with processing? (y/n): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled by user"
        exit 0
    fi
    
    setup_docker
    check_directories
    
    # Start monitoring in background
    monitor_progress
    
    # Process videos (main task)
    process_videos
    
    # Show results
    show_results
}

# Quick commands
case "${1:-main}" in
    "build")
        setup_docker
        ;;
    "monitor")
        if [[ -f "$OUTPUT_DIR/logs/progress_report.json" ]]; then
            echo "Current progress:"
            jq . "$OUTPUT_DIR/logs/progress_report.json"
        else
            echo "No progress data found"
        fi
        ;;
    "status")
        echo "Processing status:"
        docker ps | grep insta360 || echo "No containers running"
        echo
        if [[ -d "$OUTPUT_DIR" ]]; then
            OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" | wc -l)
            echo "Videos processed: $OUTPUT_COUNT"
            echo "Output size: $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "0B")"
        fi
        ;;
    "clean")
        cleanup
        docker rmi insta360-complete:latest 2>/dev/null || true
        echo "Cleaned up Docker resources"
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        main
        ;;
esac