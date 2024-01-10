# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import os
import sys
sys.path.append('.')
from robin.train.train import train

if __name__ == "__main__":
    if os.path.exists("/lustre/orion/csc538"): # if running on frontier
        username = os.environ.get('USER')
        hostname = os.environ.get('HOSTNAME')
        jobid = os.environ.get('SLURM_JOB_ID')

        miopen_use_db_path = f'/lustre/orion/csc538/scratch/{username}/miopen/{jobid}/{hostname}'
        if not os.path.exists(miopen_use_db_path):
            os.makedirs(miopen_use_db_path) 
        os.environ['MIOPEN_USER_DB_PATH'] = miopen_use_db_path
        os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = os.environ['MIOPEN_USER_DB_PATH']

        os.environ['WANDB_DIR'] = f'/lustre/orion/csc538/scratch/{username}/wandb_cache'
        os.environ['TRANSFORMERS_CACHE'] = '/lustre/orion/csc538/proj-shared/downloaded_models/hf_cache'
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

    train()
