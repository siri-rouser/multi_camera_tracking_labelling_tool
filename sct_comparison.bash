seqs=("c001" "c002" "c003" "c004")
algorithms=("boosttrack" "botsort" "bytetrack" "deepocsort" "ocsort")

for seq in "${seqs[@]}"; do
    for algorithm in "${algorithms[@]}"; do
        echo "Processing sequence: $seq with algorithm: $algorithm"
        python eval_sct.py ./sct_res/${seq}_${algorithm}_sct_boxmot_results_filtered.txt ./sct_res/images${seq}_mot_interpolated_final.txt
    done
done