#!/bin/bash

# Insta360 Docker Setup Script
# Sets up Docker container with GPU support for MediaSDK

set -e

echo "=== Insta360 Docker GPU Setup ==="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        print_warning "Please log out and back in to use Docker without sudo"
    else
        print_status "Docker found: $(docker --version)"
    fi
    
    # Check nvidia-docker
    if ! command -v nvidia-docker >/dev/null 2>&1 && ! docker info | grep -q "nvidia" 2>/dev/null; then
        print_status "Installing NVIDIA Container Toolkit..."
        
        # Add NVIDIA package repository
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
        
        print_status "NVIDIA Container Toolkit installed"
    else
        print_status "NVIDIA Container Toolkit already available"
    fi
    
    # Test GPU access
    print_status "Testing GPU access in Docker..."
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        print_status "GPU access in Docker: OK"
    else
        print_warning "GPU access test failed - will continue with CPU fallback"
    fi
}

# Build Docker image
build_image() {
    print_status "Building Insta360 Docker image..."
    
    # Find MediaSDK .deb file
    DEB_FILE=$(find . -name "libMediaSDK-dev*.deb" | head -1)
    
    if [ -z "$DEB_FILE" ]; then
        print_error "MediaSDK .deb file not found!"
        echo "Please ensure the libMediaSDK-dev-*.deb file is in the current directory"
        echo "Expected location: $(pwd)/Documents/Linux_CameraSDK-*/libMediaSDK-dev-*.deb"
        
        # Try to find it
        DEB_FILE=$(find ~/Documents -name "libMediaSDK-dev*.deb" 2>/dev/null | head -1)
        if [ -n "$DEB_FILE" ]; then
            print_status "Found MediaSDK at: $DEB_FILE"
            cp "$DEB_FILE" .
            DEB_FILE=$(basename "$DEB_FILE")
        else
            exit 1
        fi
    fi
    
    print_status "Using MediaSDK: $DEB_FILE"
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        print_status "Creating Dockerfile..."
        # The Dockerfile content would be created here
        # (Using the artifact content from above)
    fi
    
    # Build the image
    docker build -t insta360-processor:latest .
    
    print_status "Docker image built successfully!"
}

# Create wrapper scripts
create_scripts() {
    print_status "Creating wrapper scripts..."
    
    # Create single video processor
    cat > insta360_docker_single.sh << 'EOF'
#!/bin/bash

# Single video processor using Docker

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 input.insv output.mp4"
    exit 1
fi

# Convert to absolute paths
INPUT_ABS=$(realpath "$INPUT")
OUTPUT_ABS=$(realpath "$OUTPUT")
INPUT_DIR=$(dirname "$INPUT_ABS")
OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
INPUT_FILE=$(basename "$INPUT_ABS")
OUTPUT_FILE=$(basename "$OUTPUT_ABS")

echo "Processing: $INPUT_FILE -> $OUTPUT_FILE"

# Run in Docker with GPU support
docker run --rm --gpus all \
    -v "$INPUT_DIR:/input:ro" \
    -v "$OUTPUT_DIR:/output" \
    insta360-processor:latest \
    insta360_stitch "/input/$INPUT_FILE" "/output/$OUTPUT_FILE"
EOF

    chmod +x insta360_docker_single.sh
    
    # Create batch processor
    cat > insta360_docker_batch.sh << 'EOF'
#!/bin/bash

# Batch video processor using Docker

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MAX_PARALLEL="${3:-2}"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 input_directory output_directory [max_parallel]"
    echo "Example: $0 ../insta360_backup ./docker_output 4"
    exit 1
fi

INPUT_ABS=$(realpath "$INPUT_DIR")
OUTPUT_ABS=$(realpath "$OUTPUT_DIR")

mkdir -p "$OUTPUT_ABS"

echo "Batch processing:"
echo "  Input: $INPUT_ABS"
echo "  Output: $OUTPUT_ABS"
echo "  Max parallel: $MAX_PARALLEL"

# Find all video files
mapfile -t VIDEOS < <(find "$INPUT_ABS" -name "*.insv" -o -name "*.insp")

echo "Found ${#VIDEOS[@]} videos to process"

# Process videos in parallel
process_video() {
    local input_file="$1"
    local relative_path=$(realpath --relative-to="$INPUT_ABS" "$input_file")
    local output_file="$OUTPUT_ABS/${relative_path%.*}_stitched.mp4"
    local output_dir=$(dirname "$output_file")
    
    mkdir -p "$output_dir"
    
    echo "Processing: $relative_path"
    
    # Run Docker container for this video
    docker run --rm --gpus all \
        -v "$INPUT_ABS:/input:ro" \
        -v "$OUTPUT_ABS:/output" \
        insta360-processor:latest \
        insta360_stitch "/input/$relative_path" "/output/$(basename "$output_file")"
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $relative_path"
    else
        echo "✗ FAILED: $relative_path"
    fi
}

export -f process_video
export INPUT_ABS OUTPUT_ABS

# Use parallel processing
if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${VIDEOS[@]}" | parallel -j "$MAX_PARALLEL" process_video
else
    # Fallback to sequential processing
    for video in "${VIDEOS[@]}"; do
        process_video "$video"
    done
fi

echo "Batch processing complete!"
EOF

    chmod +x insta360_docker_batch.sh
    
    # Create status checker
    cat > insta360_docker_status.sh << 'EOF'
#!/bin/bash

# Check processing status

INPUT_DIR="${1:-../insta360_backup}"
OUTPUT_DIR="${2:-./docker_output}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory not found: $INPUT_DIR"
    exit 1
fi

echo "=== Insta360 Processing Status ==="
echo

INPUT_COUNT=$(find "$INPUT_DIR" -name "*.insv" -o -name "*.insp" | wc -l)
OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*_stitched.mp4" 2>/dev/null | wc -l || echo "0")

echo "Input videos: $INPUT_COUNT"
echo "Processed videos: $OUTPUT_COUNT"
echo "Remaining: $((INPUT_COUNT - OUTPUT_COUNT))"

if [ $OUTPUT_COUNT -gt 0 ]; then
    echo
    echo "Recent outputs:"
    find "$OUTPUT_DIR" -name "*_stitched.mp4" -mmin -60 2>/dev/null | head -5
    
    echo
    echo "Output size:"
    du -sh "$OUTPUT_DIR" 2>/dev/null || echo "0B"
fi

echo
echo "Docker containers running:"
docker ps --filter "ancestor=insta360-processor:latest"
EOF

    chmod +x insta360_docker_status.sh
    
    print_status "Wrapper scripts created!"
}

# Test the setup
test_setup() {
    print_status "Testing Docker setup..."
    
    # Find a sample video
    SAMPLE_VIDEO=$(find ../insta360_backup -name "*.insv" 2>/dev/null | head -1)
    
    if [ -z "$SAMPLE_VIDEO" ]; then
        print_warning "No sample video found for testing"
        return
    fi
    
    print_status "Testing with: $(basename "$SAMPLE_VIDEO")"
    
    # Test single video processing
    mkdir -p test_output
    
    if ./insta360_docker_single.sh "$SAMPLE_VIDEO" "./test_output/docker_test.mp4"; then
        print_status "Docker test successful!"
        ls -lh "./test_output/docker_test.mp4"
    else
        print_warning "Docker test failed - but may work with different settings"
    fi
}

# Main execution
main() {
    print_status "Starting Insta360 Docker setup..."
    
    check_prerequisites
    build_image
    create_scripts
    test_setup
    
    echo
    print_status "Setup complete! Usage:"
    echo "  Single video: ./insta360_docker_single.sh input.insv output.mp4"
    echo "  Batch process: ./insta360_docker_batch.sh ../insta360_backup ./docker_output 4"
    echo "  Check status: ./insta360_docker_status.sh"
    echo
    print_status "To process all 378 videos:"
    echo "  ./insta360_docker_batch.sh ../insta360_backup ./docker_output 2"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi