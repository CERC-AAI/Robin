#!/bin/bash

# python -m llava.eval.model_vqa \
#     --model-path $1 \
#     --question-file /home/dkaplan/Documents/LiClipse Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder /home/dkaplan/Documents/LiClipse Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file /home/dkaplan/Documents/LiClipse Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/answers/$3/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --model-base $2 \
#     --conv-mode vicuna_v1


mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/llava/eval/eval_gpt_review_bench.py \
    --question /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/llava/eval/table/rule.json \
    --answer-list \
        /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/answers/$3/llava-v1.5-13b.jsonl \
    --output \
        /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/reviews/$3/llava-v1.5-13b.jsonl
#We messed this one up, and have no idea what order the evals are in because we are so smart.
python /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/llava/eval/summarize_gpt_review.py -f /home/dkaplan/Documents/LiClipse\ Workspace/robin_llava/playground/data/eval/llava-bench-in-the-wild/reviews/$3/llava-v1.5-13b.jsonl
