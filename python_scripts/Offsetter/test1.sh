python offsetter27.py \
    ../Visualizer/MatcherFiles/complete_turbo_360_report_ramcache.json \
    -o gpu0.json \
    --gpu-id 0 \
    --workers 8 \
    --do-multiple-of 1 \
    --gpu-memory 14.0 \
    --top-matches 4 \
    --debug \
    --gpu-debug > out00.txt