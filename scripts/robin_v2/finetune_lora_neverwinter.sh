#!/bin/bash

# only change this
NAME=robin_v2_2
MODEL=OpenHermes-2.5-Mistral-7B
VISION=ViT-SO400M-14-SigLIP-384


# don't change this
DOWNLOADED_MODEL_PATH=/localdisks/$(whoami)/downloaded_models
MODEL=$DOWNLOADED_MODEL_PATH/$MODEL
VISION=$DOWNLOADED_MODEL_PATH/$VISION

TRAIN_PATH=/home/$(whoami)/robin
CHECKPOINT_PATH=/localdisks/$(whoami)/checkpoints/$NAME
DATA_PATH=/localdisks/$(whoami)/robin_data/LLaVA-Finetune

module load cuda/12.2

source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate robin

PRETRAIN="$CHECKPOINT_PATH/pretrain"
if [ ! -f "$PRETRAIN/mm_projector.bin" ]; then
    PRETRAIN=$(ls -dv $PRETRAIN/checkpoint-* | tail -1)
fi

cd $TRAIN_PATH

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
    --pretrain_mm_mlp_adapter $PRETRAIN/mm_projector.bin \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir $CHECKPOINT_PATH/finetune_VEtrue \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
