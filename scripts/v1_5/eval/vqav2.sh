#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


path="/localdisks/rogeralexis/robin_eval/playground"


#$1 is model_path
#$2 is model_base
#$3 is chkpt for now

CKPT="llava-v1.5-7b-lora3"

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m robin.eval.model_vqa_loader \
        --model-path $1 \
        --question-file $path/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder $path/data/eval/vqav2/test2015 \
        --answers-file $path/data/eval/vqav2/answers/$3/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --model-base $2 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$PATH/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $path/data/eval/vqav2/answers/$3/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python $path/../Robin/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir $path/playground/data/eval/vqav2

