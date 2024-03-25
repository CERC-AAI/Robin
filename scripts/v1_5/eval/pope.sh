#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/pope/val2014 \
    --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b \
    --conv-mode vicuna_v1
    

python /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/daniel/robin_llava/llava/eval/eval_pope.py \
    --annotation-dir /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/pope/coco \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
