# Define the sequences
seqs=("imagesc003")

# Loop through each sequence
for seq in "${seqs[@]}"; do
    # Run the detection process
    python /home/yuqiang/yl4300/project/MCVT_YQ/mot/tool/pre_process_label.py --seqs "$seq"
    # Run the crop tool
    python /home/yuqiang/yl4300/project/MCVT_YQ/mot/SMILEtracking_label.py --seqs "$seq"
    # Run interpolation and filtering
    python sct_tracklet_post_process.py --seqs "$seq"
done
