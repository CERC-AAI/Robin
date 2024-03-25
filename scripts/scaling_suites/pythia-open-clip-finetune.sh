#!/bin/bash
TRAIN_PATH=/home/lfsm/code/Robin/
DATA_PATH=/home/lfsm/code/Robin/playground-original/llava_finetune

WORLD_SIZE=8
GLOBAL_BATCHSIZE=128
MICRO_BATCHSIZE=8
ACC_STEPS=$(($GLOBAL_BATCHSIZE / MICRO_BATCHSIZE / WORLD_SIZE))

for VISION in 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K' 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K' 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K' 'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K'; do
    for MODEL in 'EleutherAI/pythia-410m-deduped' 'EleutherAI/pythia-1.4b-deduped' 'EleutherAI/pythia-2.8b-deduped' 'EleutherAI/pythia-6.9b-deduped' 'EleutherAI/pythia-12b-deduped'; do
        LM=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
        VT=$(echo "$VISION" | tr '/' '_'| tr '-' '_')
        CHECKPOINT_PATH=/home/lfsm/code/Robin/checkpoints/${LM}_${VT}
        echo $CHECKPOINT_PATH
        deepspeed robin/train/train_mem.py \
            --finetune_ve True \
            --vision_lr 5e-5 \
            --learning_rate 5e-5 \
            --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
            --deepspeed ./scripts/zero2.json \
            --model_name_or_path $MODEL \
            --version neox \
            --data_path $DATA_PATH/llava_v1_5_mix665k.json \
            --image_folder $DATA_PATH \
            --vision_tower $VISION \
            --pretrain_mm_mlp_adapter $CHECKPOINT_PATH/pretrain/mm_projector.bin \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -1 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir $CHECKPOINT_PATH/finetune \
            --num_train_epochs 1 \
            --per_device_train_batch_size $MICRO_BATCHSIZE \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps $ACC_STEPS \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 100 \
            --save_total_limit 1 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --lazy_preprocess True \
            --dataloader_num_workers 4 \
            --report_to wandb 
    done
done

