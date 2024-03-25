#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path $1 \
    --question-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/MME/answers/$3/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --model-base $2 \
    --conv-mode vicuna_v1

cd /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/datasets/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $3/llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/$3/llava-v1.5-13b
