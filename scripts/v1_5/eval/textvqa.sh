#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path $1 \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/textvqa/train_images \
    --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/textvqa/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/textvqa/answers/$3/llava-v1.5-13b.jsonl
