# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import sys
sys.path.append('.')  # TODO: is this really necessary?
import os

username = os.environ.get('USER')

os.environ['WANDB_DIR'] = f'/lustre/orion/csc538/scratch/{username}/wandb_cache'
os.environ['MIOPEN_USER_DB_PATH'] = f'/lustre/orion/csc538/scratch/{username}/miopen'
os.environ['TRANSFORMERS_CACHE'] = '/lustre/orion/csc538/proj-shared/hf_cache'
os.environ['WANDB_MODE'] = 'offline'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = os.environ['MIOPEN_USER_DB_PATH']

from robin.train.train import train

if __name__ == "__main__":
    train()
