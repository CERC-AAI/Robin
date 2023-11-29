# Llava-Mistral

### Install
```bash
pip install git+https://github.com/agi-collective/llava_mistral.git
```

### Run interactive command line interface
```bash
python -m llava_mistral.serve.cli \
    --model-path agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg
```

### Use as library
```python
from llava_mistral.serve.pipeline import LlavaMistralPipeline

pipe = LlavaMistralPipeline("agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora")

messages = [
  {"role": "USER", "content": "What's in the image?", "image": "https://llava-vl.github.io/static/images/view.jpg"},
]
messages = pipe(messages) 
# returns original messages list plus the new response, i.e.:
# {"role": "ASSISTANT", "content": ...}
```

### Evaluations


### Training
For training details see the `pretrain.sh` and `finetune_lora.sh` scripts inside `scripts/*/`. The training code is for information purposes only and we won't be able help people debug problems with it for this preliminary release. More rigorous and less rushed and messy training scripts will be released in the future.


### Acknowledgements 
