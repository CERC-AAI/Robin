#!/bin/bash

path="/localdisks/rogeralexis/robin_eval/playground"

python -m robin.eval.model_vqa_loader \
    --model-path $1 \
    --question-file $path/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $path/data/eval/textvqa/train_images \
    --answers-file $path/data/eval/textvqa/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

python -m robin.eval.eval_textvqa \
    --annotation-file $path/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $path/data/eval/textvqa/answers/$3/llava-v1.5-13b.jsonl
