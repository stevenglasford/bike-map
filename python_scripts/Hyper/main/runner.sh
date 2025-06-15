#last working
#python matcher39.py -d ~/penis/panoramics/playground/ -o ~/penis/testingground --debug
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python matcher_2.py \
    -d ~/penis/panoramics/playground/ \
    -o ~/penis/testingground/high_res \
    --max_frames 600 \
    --video_size 720 480 \
    --sample_rate 1.0 \
    --debug \
    --strict \
    --powersafe \
    --enable_preprocessing \
    --ram_cache 48.0 \
    --force \
	--parallel_videos 1
