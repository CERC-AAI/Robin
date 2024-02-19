#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
BASE=$2
#$3=""

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/app/playground/data/eval/gqa"

python /app/robin/robin/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --question-file $GQADIR/data/$SPLIT.jsonl \
    --image-folder $GQADIR/data/images \
    --answers-file $GQADIR/answers/$3/$SPLIT/$MODEL_NAME/answers.jsonl \
    --temperature 0 \
    --model-base $BASE \
    --conv-mode vicuna_v1

mkdir -p /export/$MODEL_NAME/gqa
cp $GQADIR/answers/$3/$SPLIT/$MODEL_NAME/answers.jsonl /export/$MODEL_NAME/gqa/answers.jsonl

output_file=$GQADIR/answers/$3/$SPLIT/$MODEL_NAME/answers.jsonl

python /app/robin/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/data/testdev_balanced_predictions.json

cp $GQADIR/data/testdev_balanced_predictions.json /export/$MODEL_NAME/gqa/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier data/testdev_balanced