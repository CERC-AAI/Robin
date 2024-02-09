#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $1 \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/answers/$3/$SPLIT/llava-v1.5-7b-lora3.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

mkdir -p /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/answers_upload/$3/$SPLIT

python /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/daniel/simon_llaba/scripts/convert_mmbench_for_submission.py \
    --annotation-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/answers/$3/$SPLIT \
    --upload-dir /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/answers_upload/$3/$SPLIT \
    --experiment llava-v1.5-7b-lora3
