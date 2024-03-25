#!/bin/bash
MODEL=$1
VISION=$2
GAS=${3:-1}

# download models
git lfs install
cd /app/downloaded_models
git clone https://huggingface.co/$MODEL
git clone https://huggingface.co/$VISION

#get updated code
git pull

# setup variables
MODEL=${MODEL##*/}
VISION=${VISION##*/}

CHECKPOINT_PATH=/app/checkpoints/$MODEL-$VISION
EXPORT_PATH=/export/$MODEL-$VISION

MODEL=/app/downloaded_models/$MODEL
VISION=/app/downloaded_models/$VISION

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
BATCH_SIZE=$(( 128 / $GPU_COUNT / $GAS ))

DATA_PATH=/app/LLaVA-Finetune

PRETRAIN=$EXPORT_PATH/pretrain

# launch training
cd /app/robin
deepspeed \
    robin/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL \
    --version v1 \
    --data_path $DATA_PATH/llava_v1_5_mix665k.json \
    --image_folder $DATA_PATH \
    --vision_tower $VISION \
    --finetune_ve True \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --pretrain_mm_mlp_adapter $PRETRAIN/mm_projector.bin \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir $CHECKPOINT_PATH/finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GAS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --vision_lr 5e-5

mv $CHECKPOINT_PATH/finetune/*.* $EXPORT_PATH/finetune/
