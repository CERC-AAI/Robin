# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# os.environ['TRANSFORMERS_CACHE'] = '/pfss/mila/hf'
# os.environ['HF_HOME'] = '/pfss/mila/hf'
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/pfss/mila/hf'


# # Load model directly
# from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir = "/pfss/mila/hf")

