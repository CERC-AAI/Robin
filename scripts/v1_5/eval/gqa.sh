#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-lora3"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/gqa/"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $1 \
        --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/gqa/data/$SPLIT.jsonl \
        --image-folder /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/gqa/data/images \
        --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/gqa/answers/$3/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --model-base $2 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/gqa/answers/$3/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/daniel/simon_llaba/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/data/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier data/testdev_balanced