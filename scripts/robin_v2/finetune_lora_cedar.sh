#!/bin/bash

#SBATCH -A def-irina
#SBATCH -J robin
#SBATCH -o /scratch/%u/job_logs/%x-%j.out
#SBATCH -e /scratch/%u/job_logs/%x-%j.err
#SBATCH --gpus-per-node=v100l:4
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --exclusive

# only change this
NAME=robin_v2
MODEL=OpenHermes-2.5-Mistral-7B
VISION=DFN2B-CLIP-ViT-L-14


# don't change this
DOWNLOADED_MODEL_PATH=/scratch/$(whoami)/downloaded_models
MODEL=$DOWNLOADED_MODEL_PATH/$MODEL
VISION=$DOWNLOADED_MODEL_PATH/$VISION

TRAIN_PATH=/scratch/$(whoami)/robin
CHECKPOINT_PATH=/scratch/$(whoami)/checkpoints/$NAME
DATA_PATH=$SLURM_TMPDIR/LLaVA-Finetune

export HEAD_NODE=$(hostname)

echo "$(date)"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar -xf /scratch/$(whoami)/robin_data/LLaVA-Finetune.tar.gz -C $SLURM_TMPDIR
echo "$(date)"

module load StdEnv/2020
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.10.2
module load rust/1.70.0
source /scratch/$(whoami)/robin_venv/bin/activate

PRETRAIN="$CHECKPOINT_PATH/pretrain"
if [ ! -f "$PRETRAIN/mm_projector.bin" ]; then
    PRETRAIN=$(ls -dv $PRETRAIN/checkpoint-* | tail -1)
fi

# important to generate hostfile in condition otherwise deepspeed will crash when only 1 node
if [ $SLURM_NNODES -gt 1 ]
then
    bash /scratch/$(whoami)/cedar_write_hostfile.sh
fi

cd $TRAIN_PATH

deepspeed \
    --hostfile /scratch/$(whoami)/hostfiles/$SLURM_JOBID-hosts \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
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
    --vision_lr 5e-5
