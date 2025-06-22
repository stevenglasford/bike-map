#last working
#python matcher39.py -d ~/penis/panoramics/playground/ -o ~/penis/testingground --debug
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python matcher50.py \
    -d ~/penis/panoramics/playground/ \
    -o ~/penis/testingground/ \
	--turbo-mode \
    --debug \
    --strict \
    --powersafe \
    --force \
	--cuda-streams \
	--vectorized-ops \
	--parallel_videos 2 
	#--gpu_batch_size 100
	#--max_frames 50 \
    #--video_size 720 480 \
    #--sample_rate 2.0 \
	
#    --enable_preprocessing \
 #   --ram_cache 48.0 
