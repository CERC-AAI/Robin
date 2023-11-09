# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['TRANSFORMERS_CACHE'] = '/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/'
# os.environ['HF_HOME'] = '/pfss/mila/hf'
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/pfss/mila/hf'



# tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", cache_dir = ""/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/")
# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")


tokenizer = AutoTokenizer.from_pretrained("/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b/")
model = AutoModelForCausalLM.from_pretrained("/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b/")
