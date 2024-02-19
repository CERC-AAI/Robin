#!/bin/bash

path="/localdisks/rogeralexis/robin_eval/playground"

python -m robin.eval.model_vqa \
    --model-path $1 \
    --question-file $path/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $path/data/eval/mm-vet/images \
    --answers-file $path/data/eval/mm-vet/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

mkdir -p $path/data/eval/mm-vet/results

python $path/../Robin/scripts/convert_mmvet_for_eval.py \
    --src $path/data/eval/mm-vet/answers/$3/llava-v1.5-13b.jsonl \
    --dst $path/data/eval/mm-vet/results/$3/llava-v1.5-13b.json

