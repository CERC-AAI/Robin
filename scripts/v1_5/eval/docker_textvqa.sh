#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
BASE=$2
#$3=""

PATH_TEXTVQA="/app/playground/data/eval/textvqa"

python /app/robin/robin/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --question-file $PATH_TEXTVQA/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $PATH_TEXTVQA/train_images \
    --answers-file $PATH_TEXTVQA/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $BASE \
    --conv-mode vicuna_v1

mkdir -p /export/$MODEL_NAME/textvqa
cp $PATH_TEXTVQA/answers/$3/llava-v1.5-13b.jsonl /export/$MODEL_NAME/textvqa/answers.jsonl

python /app/robin/robin/eval/eval_textvqa.py \
    --annotation-file $PATH_TEXTVQA/TextVQA_0.5.1_val.json \
    --result-file $PATH_TEXTVQA/answers/$3/llava-v1.5-13b.jsonl
