
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import open_clip


import os

# name = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
name = "hf-hub:timm/ViT-B-16-SigLIP"

tokenizer = get_tokenizer(name)
model, image_processor = create_model_from_pretrained(name)

print(fkd)