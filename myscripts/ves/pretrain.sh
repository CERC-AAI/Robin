#!/bin/bash
#YBATCH -r a100_8
#SBATCH -N 1
#SBATCH -o ./myouts/%j.out
#SBATCH --time=72:00:00
#SBATCH -J pretrain_llava
#SBATCH --error ./myouts/%j.err

TRAIN_PATH=/home/lfsm/code/Robin/
DATA_PATH=/home/lfsm/code/Robin/playground-original/llava_pretrain

WORLD_SIZE=8
GLOBAL_BATCHSIZE=256
MICRO_BATCHSIZE=4
ACC_STEPS=$(($GLOBAL_BATCHSIZE / MICRO_BATCHSIZE / WORLD_SIZE))

for VISIONS in 'timm vit_large_patch14_reg4_dinov2.lvd142m' 'clip openai/clip-vit-large-patch14'; do
    for MODEL in 'teknium/OpenHermes-2.5-Mistral-7B' ; do
        read -r VISION_TYPE VISION <<< $VISIONS
        LM=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
        VT=$(echo "$VISION" | tr '/' '_'| tr '-' '_')
        CHECKPOINT_PATH=/home/lfsm/code/Robin/checkpoints/${LM}_${VT}
        echo $CHECKPOINT_PATH 
        deepspeed \
            $TRAIN_PATH/robin/train/train_mem.py \
            --deepspeed ./scripts/zero2.json \
            --model_name_or_path $MODEL \
            --version plain \
            --data_path $DATA_PATH/blip_laion_cc_sbu_558k.json \
            --image_folder $DATA_PATH/images \
            --vision_tower $VISION \
            --vision_tower_type $VISION_TYPE \
            --finetune_ve False \
            --mm_projector_type mlp2x_gelu \
            --tune_mm_mlp_adapter True \
            --mm_vision_select_layer -1 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --bf16 True \
            --output_dir $CHECKPOINT_PATH/pretrain \
            --num_train_epochs 1 \
            --per_device_train_batch_size $MICRO_BATCHSIZE \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps $ACC_STEPS \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 100 \
            --save_total_limit 1 \
            --learning_rate 1e-3 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing False \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb
    done
done
