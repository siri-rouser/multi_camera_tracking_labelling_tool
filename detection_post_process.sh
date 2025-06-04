#!/bin/bash

# Define the sequences
seqs=("imagesc002")

# Loop through each sequence
for seq in "${seqs[@]}"; do
    # Run the detection process
    python detection_result_process.py --seqs "$seq"
    
    # Run the crop tool
    python detection_crop_tool.py --seqs "$seq"
done
