#!/bin/bash

# First, let's backup the file
cp optimized_matcher_fixed.py optimized_matcher_fixed.py.backup

# Fix the syntax error on line 379
# The correct line should be just:
# frames_gpu = cp.asarray(batch_frames)

# Use sed to fix the specific line
sed -i '379s/.*/                            frames_gpu = cp.asarray(batch_frames)/' optimized_matcher_fixed.py

# Also fix the stream synchronization issue (around line 387)
# Replace the for loop with simple synchronization
sed -i '/for stream in self.streams\[gpu_id\]:/,/stream.synchronize()/c\            cp.cuda.Stream.null.synchronize()' optimized_matcher_fixed.py

# Show the fixed lines to verify
echo "Fixed line 379:"
sed -n '379p' optimized_matcher_fixed.py

echo -e "\nFixed synchronization section (around line 387):"
sed -n '385,390p' optimized_matcher_fixed.py