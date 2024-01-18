#!/bin/bash

#SBATCH -A def-irina
#SBATCH -J robin
#SBATCH -o /scratch/%u/job_logs/%x-%j.out
#SBATCH -e /scratch/%u/job_logs/%x-%j.err
#SBATCH --gpus-per-node=v100l:4
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --exclusive

# time: minutes, minutes:seconds, hours:minutes:seconds, days-hours, days-hours:minutes, days-hours:minutes:seconds


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
DATA_PATH=$SLURM_TMPDIR/LLaVA-Pretrain

export HEAD_NODE=$(hostname)

echo "$(date)"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar -xf /scratch/$(whoami)/robin_data/LLaVA-Pretrain.tar.gz -C $SLURM_TMPDIR
echo "$(date)"

module load StdEnv/2020
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.10.2
module load rust/1.70.0
source /scratch/$(whoami)/robin_venv/bin/activate

mkdir -p $CHECKPOINT_PATH

# fresh miopen cache before run (need 1 cache per node)
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
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

    # --report_to wandb
