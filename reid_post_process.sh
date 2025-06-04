# Define the sequences
seqs=("imagesc003")


# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AICvenv

# Loop through each sequence
for seq in "${seqs[@]}"; do
    # Run the detection process
    python /home/yuqiang/yl4300/project/MCVT_YQ/reid/extract_image_feat_label.py --cam_list "$seq"
    # Run the crop tool
    python /home/yuqiang/yl4300/project/MCVT_YQ/reid/merge_reid_feat_label.py --cam_list "$seq"
done

conda deactivate