#!/bin/bash
#YBATCH -r rtx6000-ada_1
#SBATCH -N 1
#SBATCH -o ./myouts/eval%j.out
#SBATCH --time=72:00:00
#SBATCH -J eval_llava
#SBATCH --error ./myouts/eval%j.err

# set -e  # abort script on error
# set -x

BASE='playground-original/eval/scienceqa'
MODEL_PATH='agi-collective/pythia-410m-deduped-ViT-B-16-finetune-lora'
MODEL_BASE='EleutherAI/pythia-410m-deduped'

python -m robin.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --question-file $BASE/llava_test_CQM-A.json \
    --image-folder playground-original/llava_eval/science_qa/test \
    --answers-file $BASE/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode neox 

python robin/eval/eval_science_qa.py \
    --base-dir playground-original/llava_eval/science_qa \
    --result-file $BASE/answers/${MODEL_NAME}.jsonl \
    --output-file $BASE/answers/${MODEL_NAME}_output.jsonl \
    --output-result $BASE/answers/${MODEL_NAME}_result.json

# Prediction file playground-original/eval/scienceqa/answers/EleutherAI_pythia_2.8b_deduped_ViT_B_16_laion2b_s34b_b88K.jsonl
# Total: 4241, Correct: 245, Accuracy: 5.78%, IMG-Accuracy: 0.74%