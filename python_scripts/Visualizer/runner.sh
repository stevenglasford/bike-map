#last working visualizer33.py (pretty good, 42-60 frames per second)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python visualizer34.py \
	-i MatcherFiles/complete_turbo_360_report_ramcache.json \
	-o Output1/ \
	--processes-per-gpu 3 \
	--confidence-threshold 0.05 \
	--force \
	--powersafe
	#--batch-size 256
