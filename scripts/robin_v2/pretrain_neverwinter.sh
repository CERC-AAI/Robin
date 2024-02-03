#!/bin/bash

# only change this
NAME=robin_v2_test
MODEL=OpenHermes-2.5-Mistral-7B
VISION=ViT-L-16-SigLIP-256
GAS=2

# don't change this
DOWNLOADED_MODEL_PATH=/localdisks/$(whoami)/downloaded_models
MODEL=$DOWNLOADED_MODEL_PATH/$MODEL
VISION=$DOWNLOADED_MODEL_PATH/$VISION

TRAIN_PATH=/home/$(whoami)/robin
CHECKPOINT_PATH=/localdisks/$(whoami)/checkpoints/$NAME
DATA_PATH=/localdisks/$(whoami)/robin_data/LLaVA-Pretrain

module load cuda/12.2

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
BATCH_SIZE=$(( 256 / GPU_COUNT / GAS ))

source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate robin

mkdir -p $CHECKPOINT_PATH

cd $TRAIN_PATH

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
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GAS \
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
