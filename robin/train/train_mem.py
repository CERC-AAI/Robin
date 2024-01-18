# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import os
import sys
sys.path.append('.')
from robin.train.train import train

USE_FLASH_ATTN_2 = False

if __name__ == "__main__":
    hostname = os.environ.get('HOSTNAME') if os.environ.get('HOSTNAME') != None else os.uname()[1]

    print('Running on cluster:', end=' ')
    match hostname.lower():
        case x if 'frontier' in x:
            print('Frontier')
            USE_FLASH_ATTN_2 = False
            
            username = os.environ.get('USER')
            jobid = os.environ.get('SLURM_JOB_ID')

            miopen_use_db_path = f'/lustre/orion/csc538/scratch/{username}/miopen/{jobid}/{hostname}'
            if not os.path.exists(miopen_use_db_path):
                os.makedirs(miopen_use_db_path) 
            os.environ['MIOPEN_USER_DB_PATH'] = miopen_use_db_path
            os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = os.environ['MIOPEN_USER_DB_PATH']

            os.environ['WANDB_DIR'] = f'/lustre/orion/csc538/scratch/{username}/wandb_cache'
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['TRANSFORMERS_CACHE'] = '/lustre/orion/csc538/proj-shared/downloaded_models/hf_cache'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'

        case x if 'icewindale' in x:
            print('Icewindale')
            USE_FLASH_ATTN_2 = True

            username = os.environ.get('USER')
            os.environ['WANDB_DIR'] = f'/localdisks/{username}/wandb_cache'
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['TRANSFORMERS_CACHE'] = f'/localdisks/{username}/downloaded_models/hf_cache'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'

        case x if 'neverwinter' in x:
            print('Neverwinter')
            USE_FLASH_ATTN_2 = True

            username = os.environ.get('USER')
            os.environ['WANDB_DIR'] = f'/localdisks/{username}/wandb_cache'
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['TRANSFORMERS_CACHE'] = f'/localdisks/{username}/downloaded_models/hf_cache'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'

        case x if 'cedar' in x or 'cdr' in x:
            print('Cedar')
            USE_FLASH_ATTN_2 = False
            
            username = os.environ.get('USER')
            os.environ['WANDB_DIR'] = f'/scratch/{username}/wandb_cache'
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['WANDB__SERVICE_WAIT'] = '150'
            os.environ['TRANSFORMERS_CACHE'] = f'/scratch/{username}/downloaded_models/hf_cache'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        case _:
            print(hostname)
            print('No cluster specific config, no enviroment variables set.')

    train(USE_FLASH_ATTN_2=USE_FLASH_ATTN_2)
