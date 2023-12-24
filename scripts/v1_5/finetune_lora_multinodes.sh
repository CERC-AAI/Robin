#!/bin/bash

#SBATCH -A CSC538
#SBATCH -J robin 
#SBATCH -o /lustre/orion/csc538/scratch/%u/job_logs/%x-%j.out
#SBATCH -e /lustre/orion/csc538/scratch/%u/job_logs/%x-%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 4

# load rocm for AMD GPU and write hostfile for distribute environment discovery for Deepspeed

module load rocm/5.4.3

source activate /lustre/orion/csc538/scratch/$(whoami)/miniconda3/envs/robin

TRAIN_PATH=/lustre/orion/csc538/scratch/$(whoami)/robin
CHECKPOINT_PATH=/lustre/orion/csc538/scratch/$(whoami)/checkpoints/robin_v2
DATA_PATH=/lustre/orion/csc538/proj-shared/llava_finetune_2

MODEL=/lustre/orion/csc538/scratch/alexisroger/hf_cache/OpenHermes-2.5-Mistral-7B
VISION=openai/clip-vit-large-patch14-336

# clean the miopen cache before run.
rm -rf /lustre/orion/csc538/scratch/$(whoami)/miopen/*

bash /lustre/orion/csc538/scratch/$(whoami)/frontier_write_hostfile.sh

cd $TRAIN_PATH

#deepspeed --hostfile /lustre/orion/csc538/scratch/$(whoami)/hostfiles/$SLURM_JOBID-hosts \
deepspeed \
    $TRAIN_PATH/robin/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL \
    --version v1 \
    --data_path $DATA_PATH/llava_v1_5_mix665k.json \
    --image_folder $DATA_PATH \
    --vision_tower $VISION \
    --finetune_ve True \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --pretrain_mm_mlp_adapter $CHECKPOINT_PATH/pretrain/checkpoint-500/mm_projector.bin \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir $CHECKPOINT_PATH/finetune_VEtrue \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
