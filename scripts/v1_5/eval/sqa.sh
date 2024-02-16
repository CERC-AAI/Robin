#!/bin/bash

path="/app/playground"

python /app/robin/robin/eval/model_vqa_science.py \
    --model-path $1 \
    --question-file  $path/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder  $path/data/eval/scienceqa/images/test \
    --answers-file  $path/data/eval/scienceqa/answers/$3/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v1 \
    --model-base $2


python /app/robin/robin/eval/eval_science_qa.py \
    --base-dir $path/data/eval/scienceqa/ \
    --result-file $path/data/eval/scienceqa/answers/$3/llava-v1.5-13b.jsonl \
    --output-file $path/data/eval/scienceqa/$3/llava-v1.5-13b_output.jsonl \
    --output-result $path/data/eval/scienceqa/answers/$3/llava-v1.5-13b_result.json
