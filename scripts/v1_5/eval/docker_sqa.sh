#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
BASE=$2
#$3=""

PATH_SQA="/app/playground/data/eval/scienceqa"

python /app/robin/robin/eval/model_vqa_science.py \
    --model-path $MODEL_PATH \
    --question-file  $PATH_SQA/llava_test_CQM-A.json \
    --image-folder  $PATH_SQA/images/test \
    --answers-file  $PATH_SQA/answers/$3/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-base $BASE


python /app/robin/robin/eval/eval_science_qa.py \
    --base-dir $PATH_SQA/ \
    --result-file $PATH_SQA/answers/$3/llava-v1.5-13b.jsonl \
    --output-file $PATH_SQA/$3/llava-v1.5-13b_output.jsonl \
    --output-result $PATH_SQA/answers/$3/llava-v1.5-13b_result.json

mkdir -p /export/$MODEL_NAME/scienceqa
mv $PATH_SQA/answers/$3/llava-v1.5-13b.jsonl /export/$MODEL_NAME/scienceqa/llava-v1.5-13b.jsonl
mv $PATH_SQA/$3/llava-v1.5-13b_output.jsonl /export/$MODEL_NAME/scienceqa/llava-v1.5-13b_output.jsonl
mv $PATH_SQA/answers/$3/llava-v1.5-13b_result.json /export/$MODEL_NAME/scienceqa/llava-v1.5-13b_result.json