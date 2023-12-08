#!/bin/bash

#SBATCH -A CSC538
#SBATCH -J robin 
#SBATCH -o /lustre/orion/csc538/scratch/$(whoami)/job_logs/%x-%j.out
#SBATCH -e /lustre/orion/csc538/scratch/$(whoami)/job_logs/%x-%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 4
# load rocm for AMD GPU and write hostfile for distribute environment discovery for Deepspeed

module load rocm/5.4.3

source activate /lustre/orion/csc538/scratch/$(whoami)/miniconda3/envs/robin

TRAIN_PATH=/lustre/orion/csc538/scratch/$(whoami)/robin
CHECKPOINT_PATH=/lustre/orion/csc538/scratch/$(whoami)/checkpoints/llava-v1.5-7b
DATA_PATH=/lustre/orion/csc538/proj-shared/llava_pretrain

MODEL=lmsys/vicuna-7b-v1.5
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
    --version plain \
    --data_path $DATA_PATH/blip_laion_cc_sbu_558k.json \
    --image_folder $DATA_PATH/images \
    --vision_tower $VISION \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir $CHECKPOINT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
