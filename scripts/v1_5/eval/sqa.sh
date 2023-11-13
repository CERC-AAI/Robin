#!/bin/bash

set -e  # abort script on error
set -x

mkdir -p /tmp/eval/scienceqa/answers

cp ./playground/data/eval/scienceqa/llava_test_CQM-A.json /tmp/eval/scienceqa/llava_test_CQM-A.json
cp ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl /tmp/eval/scienceqa/answers/llava-v1.5-13b.jsonl

python -m llava.eval.model_vqa_science \
    --model-path ../trained-models/vicuna-7b-siglip-so400m-finetune-lora \
    --model-base ./hf/vicuna-7b
    --question-file /tmp/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file /tmp/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file /tmp/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file /tmp/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result /tmp/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
