import os
os.environ['TRANSFORMERS_CACHE'] = '/localdisks/rogeralexis/downloaded_models/hfache'
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
