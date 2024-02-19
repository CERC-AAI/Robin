#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
BASE=$2

PATH_VQAv2="/app/playground/data/eval/vqav2"

CKPT=$MODEL_NAME

SPLIT="llava_vqav2_mscoco_test-dev2015"

python /app/robin/robin/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --question-file $PATH_VQAv2/$SPLIT.jsonl \
    --image-folder $PATH_VQAv2/test2015 \
    --answers-file $PATH_VQAv2/answers/$3/$SPLIT/$CKPT/merge.jsonl \
    --temperature 0 \
    --model-base $BASE \
    --conv-mode vicuna_v1

python /app/robin/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir $PATH_VQAv2

