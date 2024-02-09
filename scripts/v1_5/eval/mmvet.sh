#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path $1 \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mm-vet/images \
    --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mm-vet/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

mkdir -p /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mm-vet/results

python /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/daniel/simon_llaba/scripts/convert_mmvet_for_eval.py \
    --src /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mm-vet/answers/$3/llava-v1.5-13b.jsonl \
    --dst /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/mm-vet/results/$3/llava-v1.5-13b.json

