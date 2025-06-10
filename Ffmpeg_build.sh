#!/bin/bash
set -e

# Adjust paths
FFMPEG_SRC="$HOME/penis/unsorted/ffmpeg"
INSTALL_DIR="$FFMPEG_SRC/build"
CUDA_PATH="/usr/local/cuda-12.9"

# Clean up previous build (optional)
cd "$FFMPEG_SRC"
make distclean || true

./configure \
    --prefix="$INSTALL_DIR" \
    --bindir="$INSTALL_DIR/bin" \
    --extra-cflags="-I$CUDA_PATH/include" \
    --extra-ldflags="-L$CUDA_PATH/lib64 -L$CUDA_PATH/lib64/stubs -Wl,-rpath,$INSTALL_DIR/lib" \
    --extra-libs='-lpthread -lm -ldl' \
    --enable-shared --disable-static \
    --enable-gpl --enable-nonfree \
    --enable-cuda --enable-cuda-nvcc \
    --enable-nvenc --enable-nvdec --enable-cuvid --enable-libnpp \
    --enable-libx264 --enable-libx265 \
    --enable-libvpx --enable-libaom \
    --enable-libfdk-aac --enable-libmp3lame \
    --enable-libopus --enable-libvorbis \
    --enable-libass --enable-libfreetype \
    --enable-openssl --enable-libwebp --enable-libzimg

# Build and install
make -j"$(nproc)"
make install

# Add to shell PATH if not already added
BASHRC="$HOME/.bashrc"
PROFILE_LINE="export PATH=\"$INSTALL_DIR/bin:\$PATH\""
LIB_LINE="export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\""

if ! grep -Fxq "$PROFILE_LINE" "$BASHRC"; then
    echo "$PROFILE_LINE" >> "$BASHRC"
    echo "$LIB_LINE" >> "$BASHRC"
    echo "🔧 Added FFmpeg to PATH and LD_LIBRARY_PATH in ~/.bashrc"
fi

# Apply for current session
export PATH="$INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"

echo
echo "✅ FFmpeg installed to: $INSTALL_DIR"
echo "👉 You can now run 'ffmpeg' from any terminal."
echo "🧠 Restart your shell or run: source ~/.bashrc"