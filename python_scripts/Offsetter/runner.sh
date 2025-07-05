# Production mode with optimal settings for your hardware
python offsetter1.py ../Visualizer/MatcherFiles/complete_turbo_360_report_ramcache.json \
  -o gpu_enhanced_results.json \
  --strict \
  --max-gpu-memory 10.0 \
  --gpu-batch-size 512 \
  --cuda-streams 16
