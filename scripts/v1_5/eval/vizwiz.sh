#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path $1 \
    --question-file ../../..//playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ../../..//playground/data/eval/vizwiz/test \
    --answers-file ../../..//playground/data/eval/vizwiz/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

python ../../convert_vizwiz_for_submission.py \
    --annotation-file ../../..//playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ../../..//playground/data/eval/vizwiz/answers/$3/llava-v1.5-13b.jsonl \
    --result-upload-file ../../..//playground/data/eval/vizwiz/answers_upload/$3/llava-v1.5-13b.json
