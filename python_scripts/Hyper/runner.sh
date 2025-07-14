#last working
#python matcher39.py -d ~/penis/panoramics/playground/ -o ~/penis/testingground --debug
#nvidia-smi --gpu-reset
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENCV_FFMPEG_CAPTURE_OPTIONS="buffer_size;8388608;tcp_nodelay;1"
export OPENCV_FFMPEG_READ_ATTEMPTS=4096
python matcher51.1.py \
    -d ~/penis/panoramics/playground/ \
    -o ~/bike-map/python_scripts/Visualizer/Output2 \
	--turbo-mode \
    --debug \
    --strict \
    --powersafe \
	--cuda-streams \
	--vectorized-ops \
	--parallel_videos 2 \
	--gpu-batch-size 128 \
	--correlation-batch-size 5000 \
	--max_gpu_memory 8.0 \
	--max-cpu-workers 0 \
	--force 
	#--gpu_batch_size 100 \
	#--max_frames 50 \
    #--video_size 720 480 \
    #--sample_rate 2.0 \
	
#    --enable_preprocessing \
 #   --ram_cache 48.0 

#python matcher50.py \
    #-d ~/penis/panoramics/playground/ \
    #-o ~/penis/testingground/ \
	#--turbo-mode \
    #--debug \
    #--strict \
    #--powersafe \
    #--force \
	#--cuda-streams \
	#--vectorized-ops \
	#--parallel_videos 8 \
	#--gpu-batch-size 32 \
	#--correlation-batch-size 2000 \
	#--max-cpu-workers 0 \
	#--max_gpu_memory 8.0
