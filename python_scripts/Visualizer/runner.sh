#last working visualizer33.py (pretty good, 42-60 frames per second)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python visualizer33.py \
	-i MatcherFiles/complete_turbo_360_report_ramcache.json \
	-o Output/ \
	--process-per-gpu 3 \
	--confidence-threshold 0.05
	#--batch-size 256
