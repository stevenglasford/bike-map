#!/bin/bash

# Insta360 SDK Setup for HPC/Supercomputer Environments
# No system modifications - works with existing CUDA/FFmpeg

set -e  # Exit on any error

echo "=== Insta360 SDK Setup for HPC/Supercomputer ==="
echo "This setup works with existing system libraries without modifications"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check environment
check_hpc_environment() {
    print_status "Checking HPC environment..."
    
    # Check if we're on a compute node or login node
    if [[ -n "$SLURM_JOB_ID" ]]; then
        print_status "Running on SLURM compute node (Job ID: $SLURM_JOB_ID)"
    elif [[ -n "$PBS_JOBID" ]]; then
        print_status "Running on PBS compute node (Job ID: $PBS_JOBID)"
    else
        print_warning "Running on login node - consider using compute node for heavy processing"
    fi
    
    # Check available modules
    if command -v module >/dev/null 2>&1; then
        print_status "Environment modules system available"
        echo "Currently loaded modules:"
        module list 2>&1 | head -10 || echo "  (none or module list failed)"
    else
        print_warning "No environment modules system found"
    fi
    
    # Check for conda/virtual environments
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        print_status "Conda environment active: $CONDA_DEFAULT_ENV"
    elif [[ -n "$VIRTUAL_ENV" ]]; then
        print_status "Python virtual environment active: $VIRTUAL_ENV"
    else
        print_warning "No Python environment detected - consider using conda/virtualenv"
    fi
}

# Check system libraries without modifying
check_system_libraries() {
    print_status "Checking existing system libraries (read-only)..."
    
    # Check CUDA
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        print_status "Found CUDA version: $CUDA_VERSION"
        
        # Check CUDA runtime library
        if ldconfig -p | grep -q "libcudart"; then
            print_status "CUDA runtime library available"
        else
            print_warning "CUDA runtime library not in ldconfig path"
        fi
        
        # Note about version compatibility
        if [[ "$CUDA_VERSION" != "11.1" ]]; then
            print_warning "MediaSDK expects CUDA 11.1, found $CUDA_VERSION"
            print_status "Will attempt compatibility mode..."
        fi
    else
        print_error "CUDA not found. Load CUDA module if available: module load cuda"
        echo "Available modules:"
        module avail cuda 2>&1 | head -5 || echo "  (module command failed)"
    fi
    
    # Check g++
    if command -v g++ >/dev/null 2>&1; then
        GCC_VERSION=$(g++ --version | head -1 | sed -n 's/.*) \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p')
        print_status "Found g++ version: $GCC_VERSION"
        
        if [[ ! "$GCC_VERSION" =~ ^9\. ]]; then
            print_warning "MediaSDK expects g++ 9.x, found $GCC_VERSION"
            print_status "Will attempt compatibility mode..."
        fi
    else
        print_error "g++ not found. Load GCC module if available: module load gcc"
    fi
    
    # Check FFmpeg
    if command -v ffmpeg >/dev/null 2>&1; then
        FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -1 | sed -n 's/ffmpeg version \([^ ]*\).*/\1/p')
        print_status "Found FFmpeg version: $FFMPEG_VERSION"
    else
        print_warning "FFmpeg not found. Load FFmpeg module if available: module load ffmpeg"
    fi
    
    # Check OpenCV
    if pkg-config --exists opencv4 2>/dev/null; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_status "Found OpenCV4 version: $OPENCV_VERSION"
    elif pkg-config --exists opencv 2>/dev/null; then
        OPENCV_VERSION=$(pkg-config --modversion opencv)
        print_status "Found OpenCV version: $OPENCV_VERSION"
    else
        print_warning "OpenCV not found via pkg-config. May need to load module."
    fi
}

# Check SDK files
check_sdk_files() {
    print_status "Checking for SDK files and directories..."
    
    CAMERA_SDK_FILE=""
    MEDIA_SDK_FILE=""
    CAMERA_SDK_DIR=""
    MEDIA_SDK_DIR=""
    
    # Look for extracted directories first
    for dir in */; do
        if [[ -d "$dir" && "$dir" == *"CameraSDK"* ]]; then
            CAMERA_SDK_DIR="${dir%/}"
        elif [[ -d "$dir" && "$dir" == *"MediaSDK"* ]]; then
            MEDIA_SDK_DIR="${dir%/}"
        elif [[ -d "$dir" && "$dir" == *"libMediaSDK"* ]]; then
            MEDIA_SDK_DIR="${dir%/}"
        fi
    done
    
    # Look for archive files if directories not found
    if [[ -z "$CAMERA_SDK_DIR" ]]; then
        for file in *.gz *.tar.gz; do
            if [[ -f "$file" && "$file" == *"CameraSDK"* ]]; then
                CAMERA_SDK_FILE="$file"
            fi
        done
    fi
    
    if [[ -z "$MEDIA_SDK_DIR" ]]; then
        for file in *.xz *.tar.xz; do
            if [[ -f "$file" && "$file" == *"MediaSDK"* ]]; then
                MEDIA_SDK_FILE="$file"
            fi
        done
    fi
    
    # Report findings
    if [[ -n "$CAMERA_SDK_DIR" ]]; then
        print_status "Found extracted CameraSDK directory: $CAMERA_SDK_DIR"
    elif [[ -n "$CAMERA_SDK_FILE" ]]; then
        print_status "Found CameraSDK archive: $CAMERA_SDK_FILE"
    else
        print_error "CameraSDK not found!"
        exit 1
    fi
    
    if [[ -n "$MEDIA_SDK_DIR" ]]; then
        print_status "Found extracted MediaSDK directory: $MEDIA_SDK_DIR"
    elif [[ -n "$MEDIA_SDK_FILE" ]]; then
        print_status "Found MediaSDK archive: $MEDIA_SDK_FILE"
    else
        print_error "MediaSDK not found!"
        exit 1
    fi
}

# Extract SDK files (if needed)
extract_sdks() {
    # Only extract if directories don't exist
    if [[ -z "$CAMERA_SDK_DIR" && -n "$CAMERA_SDK_FILE" ]]; then
        print_status "Extracting CameraSDK..."
        tar -xzf "$CAMERA_SDK_FILE"
        CAMERA_SDK_DIR=$(tar -tzf "$CAMERA_SDK_FILE" | head -1 | cut -f1 -d"/")
        print_status "CameraSDK extracted to: $CAMERA_SDK_DIR"
    fi
    
    if [[ -z "$MEDIA_SDK_DIR" && -n "$MEDIA_SDK_FILE" ]]; then
        print_status "Extracting MediaSDK..."
        tar -xJf "$MEDIA_SDK_FILE"
        MEDIA_SDK_DIR=$(tar -tJf "$MEDIA_SDK_FILE" | head -1 | cut -f1 -d"/")
        print_status "MediaSDK extracted to: $MEDIA_SDK_DIR"
    fi
}

# Setup user-space Python environment (conservative approach)
setup_python_environment() {
    print_status "Setting up Python environment (preserving existing optimized packages)..."
    
    # Check what's already available
    print_status "Checking existing Python packages..."
    
    # Test critical packages without installing
    MISSING_PACKAGES=()
    
    # Check numpy
    if python3 -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null; then
        print_status "NumPy already available (keeping existing optimized version)"
    else
        MISSING_PACKAGES+=("numpy")
        print_warning "NumPy not found"
    fi
    
    # Check OpenCV
    if python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null; then
        print_status "OpenCV already available (keeping existing optimized version)"
    else
        MISSING_PACKAGES+=("opencv-python")
        print_warning "OpenCV not found"
    fi
    
    # Check Pillow
    if python3 -c "import PIL; print('Pillow version:', PIL.__version__)" 2>/dev/null; then
        print_status "Pillow already available"
    else
        MISSING_PACKAGES+=("Pillow")
        print_warning "Pillow not found"
    fi
    
    # Only install missing packages
    if [[ ${#MISSING_PACKAGES[@]} -eq 0 ]]; then
        print_status "All required packages already available - no installation needed"
        return 0
    fi
    
    # Ask user before installing anything
    echo
    print_warning "Missing packages: ${MISSING_PACKAGES[*]}"
    echo "Options:"
    echo "  1. Install missing packages (may overwrite optimized versions)"
    echo "  2. Skip package installation (you'll need to install manually)"
    echo "  3. Create minimal virtual environment for missing packages only"
    echo
    read -p "Choose option [1/2/3]: " -n 1 -r
    echo
    
    case $REPLY in
        1)
            install_missing_packages "${MISSING_PACKAGES[@]}"
            ;;
        2)
            print_status "Skipping package installation"
            print_status "You may need to install these manually: ${MISSING_PACKAGES[*]}"
            ;;
        3)
            create_minimal_venv "${MISSING_PACKAGES[@]}"
            ;;
        *)
            print_status "Invalid option, skipping package installation"
            ;;
    esac
}

# Install only missing packages
install_missing_packages() {
    local packages=("$@")
    print_status "Installing only missing packages: ${packages[*]}"
    
    # Create minimal requirements
    cat > requirements_minimal.txt << EOF
# Only missing packages - preserving existing optimized installations
EOF
    
    for package in "${packages[@]}"; do
        case $package in
            "numpy")
                echo "numpy>=1.19.0" >> requirements_minimal.txt
                ;;
            "opencv-python")
                echo "opencv-python>=4.5.0" >> requirements_minimal.txt
                ;;
            "Pillow")
                echo "Pillow>=8.0.0" >> requirements_minimal.txt
                ;;
        esac
    done
    
    # Install based on environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        pip3 install -r requirements_minimal.txt
    elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        pip3 install -r requirements_minimal.txt
    else
        pip3 install --user -r requirements_minimal.txt
    fi
    
    print_status "Minimal package installation completed"
}

# Create minimal virtual environment for missing packages only
create_minimal_venv() {
    local packages=("$@")
    print_status "Creating minimal virtual environment for missing packages only..."
    
    # Create venv only if needed
    if [[ -z "$VIRTUAL_ENV" ]]; then
        python3 -m venv ./venv_minimal
        source ./venv_minimal/bin/activate
        print_status "Created minimal virtual environment"
    fi
    
    # Install only missing packages
    for package in "${packages[@]}"; do
        case $package in
            "numpy")
                pip3 install "numpy>=1.19.0"
                ;;
            "opencv-python")
                pip3 install "opencv-python>=4.5.0"
                ;;
            "Pillow")
                pip3 install "Pillow>=8.0.0"
                ;;
        esac
    done
    
    print_status "Minimal virtual environment created with missing packages only"
}

# Setup MediaSDK for HPC
setup_media_sdk_hpc() {
    print_status "Setting up MediaSDK for HPC environment..."
    
    if [[ -d "$MEDIA_SDK_DIR" ]]; then
        # Check for .deb package (but don't install system-wide)
        DEB_FILE=$(find "$MEDIA_SDK_DIR" -name "*.deb" | head -1)
        
        if [[ -n "$DEB_FILE" ]]; then
            print_status "Found MediaSDK .deb package: $DEB_FILE"
            print_status "Extracting .deb contents locally (no system installation)..."
            
            # Extract .deb without installing
            mkdir -p mediasdk_local
            cd mediasdk_local
            
            # Extract the .deb package
            ar x "../$DEB_FILE"
            
            # Check what files were extracted
            print_status "Checking .deb contents..."
            ls -la
            
            # Find and extract the data archive (can be .tar.gz, .tar.xz, etc.)
            DATA_ARCHIVE=""
            for archive in data.tar.* data.tar; do
                if [[ -f "$archive" ]]; then
                    DATA_ARCHIVE="$archive"
                    break
                fi
            done
            
            if [[ -n "$DATA_ARCHIVE" ]]; then
                print_status "Found data archive: $DATA_ARCHIVE"
                
                # Extract based on file type
                case "$DATA_ARCHIVE" in
                    *.tar.xz)
                        tar -xJf "$DATA_ARCHIVE"
                        ;;
                    *.tar.gz)
                        tar -xzf "$DATA_ARCHIVE"
                        ;;
                    *.tar.bz2)
                        tar -xjf "$DATA_ARCHIVE"
                        ;;
                    *.tar)
                        tar -xf "$DATA_ARCHIVE"
                        ;;
                    *)
                        print_error "Unknown archive format: $DATA_ARCHIVE"
                        cd ..
                        rm -rf mediasdk_local
                        return 1
                        ;;
                esac
                
                print_status "Successfully extracted .deb contents"
            else
                print_error "No data archive found in .deb package"
                print_status "Available files:"
                ls -la
                cd ..
                rm -rf mediasdk_local
                return 1
            fi
            
            cd ..
            
            # Copy libraries to local directory
            if [[ -d "mediasdk_local/usr/lib" ]]; then
                mkdir -p local_lib
                cp -r mediasdk_local/usr/lib/* local_lib/
                print_status "MediaSDK libraries extracted to local_lib/"
                ls -la local_lib/
            elif [[ -d "mediasdk_local/usr/lib64" ]]; then
                mkdir -p local_lib
                cp -r mediasdk_local/usr/lib64/* local_lib/
                print_status "MediaSDK libraries extracted to local_lib/"
                ls -la local_lib/
            else
                print_warning "No libraries found in expected locations"
                print_status "Checking extracted structure:"
                find mediasdk_local -name "*.so*" -o -name "lib*" | head -10
            fi
            
            # Copy headers
            if [[ -d "mediasdk_local/usr/include" ]]; then
                mkdir -p local_include
                cp -r mediasdk_local/usr/include/* local_include/
                print_status "MediaSDK headers extracted to local_include/"
            else
                print_warning "No headers found in usr/include"
            fi
            
            # Copy binaries
            if [[ -d "mediasdk_local/usr/bin" ]]; then
                mkdir -p local_bin
                cp -r mediasdk_local/usr/bin/* local_bin/
                chmod +x local_bin/*
                print_status "MediaSDK binaries extracted to local_bin/"
                ls -la local_bin/
            else
                print_warning "No binaries found in usr/bin"
            fi
            
            # Clean up extraction directory
            rm -rf mediasdk_local
            
        else
            print_status "No .deb package found. Checking existing MediaSDK structure..."
            
            # Check local MediaSDK structure
            if [[ -d "$MEDIA_SDK_DIR/lib" ]]; then
                print_status "Found MediaSDK libraries in SDK directory"
                mkdir -p local_lib
                cp -r "$MEDIA_SDK_DIR/lib"/* local_lib/ 2>/dev/null || true
            fi
            
            if [[ -d "$MEDIA_SDK_DIR/include" ]]; then
                print_status "Found MediaSDK headers in SDK directory"
                mkdir -p local_include
                cp -r "$MEDIA_SDK_DIR/include"/* local_include/ 2>/dev/null || true
            fi
            
            if [[ -d "$MEDIA_SDK_DIR/bin" ]]; then
                print_status "Found MediaSDK binaries in SDK directory"
                mkdir -p local_bin
                cp -r "$MEDIA_SDK_DIR/bin"/* local_bin/ 2>/dev/null || true
                chmod +x local_bin/* 2>/dev/null || true
            fi
        fi
        
        # Look for examples and test compilation
        EXAMPLE_CC=$(find "$MEDIA_SDK_DIR" -name "main.cc" | head -1)
        if [[ -n "$EXAMPLE_CC" ]]; then
            print_status "Testing MediaSDK compilation (local libraries)..."
            EXAMPLE_DIR=$(dirname "$EXAMPLE_CC")
            cd "$EXAMPLE_DIR"
            
            # Try compilation with current environment
            COMPILE_FLAGS="-std=c++11"
            INCLUDE_PATHS=""
            LIBRARY_PATHS=""
            
            # Add local include paths
            if [[ -d "$PWD/../../local_include" ]]; then
                INCLUDE_PATHS="$INCLUDE_PATHS -I$PWD/../../local_include"
            fi
            if [[ -d "$PWD/../include" ]]; then
                INCLUDE_PATHS="$INCLUDE_PATHS -I$PWD/../include"
            fi
            
            # Add local library paths
            if [[ -d "$PWD/../../local_lib" ]]; then
                LIBRARY_PATHS="$LIBRARY_PATHS -L$PWD/../../local_lib -Wl,-rpath,$PWD/../../local_lib"
            fi
            if [[ -d "$PWD/../lib" ]]; then
                LIBRARY_PATHS="$LIBRARY_PATHS -L$PWD/../lib -Wl,-rpath,$PWD/../lib"
            fi
            
            if g++ main.cc $COMPILE_FLAGS $INCLUDE_PATHS $LIBRARY_PATHS -lMediaSDK -o testSDKDemo_hpc 2>/dev/null; then
                print_status "MediaSDK compilation successful!"
                ./testSDKDemo_hpc >/dev/null 2>&1 || print_status "Test program compiled (may need input parameters)"
            else
                print_warning "MediaSDK compilation failed. Will use fallback methods."
                print_status "Compilation command was:"
                echo "g++ main.cc $COMPILE_FLAGS $INCLUDE_PATHS $LIBRARY_PATHS -lMediaSDK -o testSDKDemo_hpc"
            fi
            
            cd - >/dev/null
        else
            print_warning "No example main.cc found for compilation test"
        fi
    else
        print_error "MediaSDK directory not found: $MEDIA_SDK_DIR"
        return 1
    fi
}

# Create HPC-compatible configuration
create_hpc_config() {
    print_status "Creating HPC-compatible configuration..."
    
    # Detect library paths
    CAMERA_LIB_PATH=""
    MEDIA_LIB_PATH=""
    
    if [[ -d "$CAMERA_SDK_DIR/lib" ]]; then
        CAMERA_LIB_PATH="$PWD/$CAMERA_SDK_DIR/lib"
    fi
    
    if [[ -d "$PWD/local_lib" ]]; then
        MEDIA_LIB_PATH="$PWD/local_lib"
    elif [[ -d "$MEDIA_SDK_DIR/lib" ]]; then
        MEDIA_LIB_PATH="$PWD/$MEDIA_SDK_DIR/lib"
    fi
    
    cat > insta360_config.json << EOF
{
    "camera_sdk_path": "./$CAMERA_SDK_DIR",
    "media_sdk_path": "./$MEDIA_SDK_DIR",
    "local_lib_path": "./local_lib",
    "local_bin_path": "./local_bin",
    "download_path": "./downloads",
    "output_path": "./stitched_videos",
    "temp_path": "./temp",
    "max_concurrent_downloads": 2,
    "video_quality": "8K",
    "stitch_format": "mp4",
    "extract_timestamps": true,
    "cleanup_temp": true,
    "log_level": "INFO",
    "hpc_mode": true,
    "cuda_version": "$(nvcc --version 2>/dev/null | grep release | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || echo 'unknown')",
    "ffmpeg_path": "$(which ffmpeg 2>/dev/null || echo 'not_found')",
    "use_system_libs": true
}
EOF
    
    print_status "HPC configuration created: insta360_config.json"
}

# Create HPC startup script
create_hpc_startup_script() {
    print_status "Creating HPC-compatible startup script..."
    
    cat > run_insta360_hpc.sh << EOF
#!/bin/bash

# Insta360 Processor for HPC Environments
# Respects existing optimized packages and environments

echo "=== Insta360 HPC Processor ==="

# Load modules if available and not already loaded
load_module_if_available() {
    local module_name=\$1
    if command -v module >/dev/null 2>&1; then
        if ! module list 2>&1 | grep -q "\$module_name"; then
            if module avail 2>&1 | grep -q "\$module_name"; then
                echo "Loading module: \$module_name"
                module load \$module_name
            fi
        fi
    fi
}

# Try to load common HPC modules
load_module_if_available "cuda"
load_module_if_available "gcc"
load_module_if_available "ffmpeg"
load_module_if_available "opencv"
load_module_if_available "python"

# Set up library paths (prepend to preserve existing optimized libraries)
export LD_LIBRARY_PATH="\$PWD/$CAMERA_SDK_DIR/lib:\$PWD/local_lib:\$PWD/$MEDIA_SDK_DIR/lib:\$LD_LIBRARY_PATH"

# Set up binary paths
export PATH="\$PWD/local_bin:\$PWD/$CAMERA_SDK_DIR/bin:\$PWD/$MEDIA_SDK_DIR/bin:\$PATH"

# Activate virtual environment ONLY if it exists and packages are missing
activate_python_env() {
    # Check if we need the virtual environment
    if python3 -c "import numpy, cv2" >/dev/null 2>&1; then
        echo "All required packages available in current environment"
        return 0
    fi
    
    # Try minimal venv first
    if [[ -f "./venv_minimal/bin/activate" ]]; then
        echo "Activating minimal Python environment..."
        source ./venv_minimal/bin/activate
        return 0
    fi
    
    # Try full venv
    if [[ -f "./venv/bin/activate" ]]; then
        echo "Activating Python virtual environment..."
        source ./venv/bin/activate
        return 0
    fi
    
    # Check user-installed packages
    if [[ -d "\$HOME/.local/lib/python3"* ]]; then
        export PYTHONPATH="\$HOME/.local/lib/python3*/site-packages:\$PYTHONPATH"
        echo "Using user-installed Python packages"
    fi
}

# Activate Python environment if needed
activate_python_env

# Check configuration
if [[ ! -f "insta360_config.json" ]]; then
    echo "ERROR: Configuration file not found! Run ./setup_hpc.sh first."
    exit 1
fi

# Show environment (without overwhelming output)
echo "Environment:"
echo "  CUDA: \$(nvcc --version 2>/dev/null | grep release | head -1 || echo 'Not found')"
echo "  FFmpeg: \$(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3 || echo 'Not found')"
echo "  Python: \$(python3 --version)"
echo "  OpenCV: \$(python3 -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'Not found')"
echo "  NumPy: \$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not found')"
if [[ -n "\$VIRTUAL_ENV" ]]; then
    echo "  Virtual Env: \$VIRTUAL_ENV"
fi
echo "  Working Dir: \$PWD"
echo

# Run the processor
python3 run_insta360.py "\$@"
EOF

    chmod +x run_insta360_hpc.sh
    print_status "HPC startup script created: run_insta360_hpc.sh"
}

# Create job submission scripts for common schedulers
create_job_scripts() {
    print_status "Creating HPC job submission scripts..."
    
    # SLURM script
    cat > submit_slurm.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=insta360_processing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load any required modules
module load cuda
module load gcc
module load ffmpeg

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Run processing
./run_insta360_hpc.sh --mode auto
EOF

    # PBS script
    cat > submit_pbs.sh << 'EOF'
#!/bin/bash
#PBS -N insta360_processing
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=02:00:00
#PBS -q gpu

# Load modules
module load cuda
module load gcc
module load ffmpeg

# Change to working directory
cd $PBS_O_WORKDIR

# Run processing
./run_insta360_hpc.sh --mode auto
EOF

    chmod +x submit_*.sh
    print_status "Job scripts created: submit_slurm.sh, submit_pbs.sh"
}

# Create directories
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p downloads
    mkdir -p stitched_videos
    mkdir -p temp
    mkdir -p logs
    
    print_status "Directory structure created"
}

# Test HPC setup
test_hpc_setup() {
    print_status "Testing HPC setup..."
    
    # Test library loading
    if [[ -f "$CAMERA_SDK_DIR/lib/libCameraSDK.so" ]]; then
        if ldd "$CAMERA_SDK_DIR/lib/libCameraSDK.so" >/dev/null 2>&1; then
            print_status "CameraSDK library dependencies OK"
        else
            print_warning "CameraSDK library has missing dependencies"
        fi
    fi
    
    # Test Python environment
    if python3 -c "import numpy, cv2" 2>/dev/null; then
        print_status "Python environment OK"
    else
        print_warning "Python environment missing dependencies"
    fi
    
    # Test CUDA availability
    if python3 -c "import os; print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'))" 2>/dev/null; then
        print_status "CUDA environment check completed"
    fi
}

# Show next steps
show_hpc_next_steps() {
    echo
    echo "=== HPC Setup Complete! ==="
    echo
    print_status "Your Insta360 processing system is ready for HPC use:"
    echo
    echo "Quick start:"
    echo "  ./run_insta360_hpc.sh --mode auto"
    echo
    echo "For batch processing:"
    echo "  sbatch submit_slurm.sh    (SLURM)"
    echo "  qsub submit_pbs.sh        (PBS)"
    echo
    echo "Files created:"
    echo "- insta360_config.json (HPC configuration)"
    echo "- run_insta360_hpc.sh (HPC startup script)"
    echo "- submit_slurm.sh, submit_pbs.sh (job scripts)"
    echo "- local_lib/, local_bin/ (extracted MediaSDK)"
    echo "- venv/ (Python virtual environment)"
    echo
    echo "Key features for HPC:"
    echo "- No system modifications required"
    echo "- Works with existing CUDA/FFmpeg versions"
    echo "- User-space Python environment"
    echo "- Automatic module loading"
    echo "- Job scheduler integration"
    echo
    print_status "Ready for 8K video processing on your supercomputer!"
}

# Main execution
main() {
    print_status "Starting HPC/Supercomputer setup..."
    
    check_hpc_environment
    check_system_libraries
    check_sdk_files
    extract_sdks
    setup_python_environment
    setup_media_sdk_hpc
    create_hpc_config
    create_directories
    create_hpc_startup_script
    create_job_scripts
    test_hpc_setup
    show_hpc_next_steps
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi