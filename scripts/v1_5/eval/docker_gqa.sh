#!/bin/bash

MODEL=$1
BASE=$2
#$3=""

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/app/playground/data/eval/gqa"

python /app/robin/robin/eval/model_vqa_loader.py \
    --model-path $MODEL \
    --question-file $GQADIR/data/$SPLIT.jsonl \
    --image-folder $GQADIR/data/images \
    --answers-file $GQADIR/answers/$3/$SPLIT/$MODEL/answers.jsonl \
    --temperature 0 \
    --model-base $BASE \
    --conv-mode vicuna_v1

model_name=$(basename $1)
mkdir -p /export/$model_name/gqa
mv $GQADIR/answers/$3/$SPLIT/$MODEL/answers.jsonl /export/$model_name/gqa/answers.jsonl

output_file=$GQADIR/answers/$3/$SPLIT/$MODEL/answers.jsonl

python /app/robin/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/data/testdev_balanced_predictions.json

mv $GQADIR/data/testdev_balanced_predictions.json /export/$model_name/gqa/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier data/testdev_balanced