cd ~
rm -rf VideoProcessingFramework
git clone --recursive https://github.com/NVIDIA/VideoProcessingFramework.git
cd VideoProcessingFramework

mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

cd ~/VideoProcessingFramework/PyNvCodec
pip install .