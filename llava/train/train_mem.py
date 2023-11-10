# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import sys
sys.path.append('.')
import os

os.environ['WANDB_DIR'] = '.'
# os.environ['MIOPEN_USER_DB_PATH'] = '/lustre/orion/csc538/scratch/lfsm/miopoen'
# os.environ['TRANSFORMERS_CACHE'] = '/lustre/orion/csc538/proj-shared/hf_cache'
# os.environ['WANDB_MODE'] = 'offline'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = os.environ['MIOPEN_USER_DB_PATH']

# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
