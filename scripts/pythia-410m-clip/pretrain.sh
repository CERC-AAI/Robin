#!/bin/bash

# This scripts is used for testing the feasibility of 
# Training pythia-410m clip on local 4 gpus machine.

# Gradient_checkpointing is set to False because I meet NotImplementedError 
# When calling model.enable_input_require_grads() of pythia model instance

# FlashAttenion2 is disable as well at robin/train/train.py #824

MODEL='EleutherAI/pythia-410m'
VISION=openai/clip-vit-large-patch14-336
TRAIN_PATH=/home/lfsm/code/Robin/
CHECKPOINT_PATH=/home/lfsm/code/Robin/robin/checkpoints
DATA_PATH=/home/lfsm/code/Robin/playground-original/llava_pretrain

deepspeed \
    $TRAIN_PATH/robin/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL \
    --version plain \
    --data_path $DATA_PATH/blip_laion_cc_sbu_558k.json \
    --image_folder $DATA_PATH/images \
    --vision_tower $VISION \
    --finetune_ve False \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir $CHECKPOINT_PATH/pretrain \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb
