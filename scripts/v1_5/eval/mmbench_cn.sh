#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path $1 \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench_cn/answers/$3/$SPLIT/llava-v1.5-13b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

mkdir -p /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench/answers_upload/$SPLIT

python /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/daniel/simon_llaba/scripts/convert_mmbench_for_submission.py \
    --annotation-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench_cn/answers/$3/$SPLIT \
    --upload-dir /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mmbench_cn/answers_upload/$3/$SPLIT \
    --experiment llava-v1.5-13b
