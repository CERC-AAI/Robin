# Llava-Mistral

### Install
Ideally install into an empty venv (`python -m venv venv && source venv/bin/activate`)

```bash
pip install git+ssh://git@github.com/agi-collective/llava_mistral.git
```

### Run interactive command line interface
```bash
python -m llava.serve.cli \
    --model-path agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg
```

### Use as library
```python
from llava.serve.pipeline import LlavaMistralPipeline

pipe = LlavaMistralPipeline(
    model_path="agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora",
    model_base="teknium/OpenHermes-2.5-Mistral-7B",
)

messages = [
  {"role": "USER", "content": "What's in the image?", "image": "https://llava-vl.github.io/static/images/view.jpg"},
]
messages = pipe(messages) 
# returns original messages list plus the new response, i.e.:
# {"role": "ASSISTANT", "content": ...}
```

### Available models

| Model                                                              | Base                              | GQA   | SQA Text | SQA Image | VQAT  |
| ------------------------------------------------------------------ | --------------------------------- | ----- | -------- | --------- | ----- |
| liuhaotian/llava-v1.5-7b                                           | lmsys/vicuna-7b-v1.5              | 62    | 70.43    | 66.8      | 58.2  |
| liuhaotian/llava-v1.5-13b                                          | lmsys/vicuna-7b-v1.5              | 63.3  |          | 71.6      | 61.3  |
| agi-collective/vicuna-7b-clip-finetune-lora                        | lmsys/vicuna-7b-v1.5              | **62.04** | 70.86    | 68.72     | **57.53** |
| agi-collective/vicuna-7b-siglip-so400m-finetune-lora               | lmsys/vicuna-7b-v1.5              | 56.79 | 68.76    | 67.48     | 53.14 |
| agi-collective/mistral-7b-siglip-so400m-finetune-lora              | mistralai/Mistral-7B-v0.1         | 49.44 | 73.66    | 68.57     | 45.01 |
| agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora           | teknium/OpenHermes-2.5-Mistral-7B | 54.48 | **79.56**    | **74.22**     | 52.69 |
| agi-collective/mistral-7b-oh-siglip-so400m-frozen-ve-finetune-lora | teknium/OpenHermes-2.5-Mistral-7B | 53.59 | 78.17    | 72.73     | 53.29 |

(best 7B model results **highlighted**)


### Evaluations
Evaluations were done using the a subset of the evaluation suite in https://github.com/haotian-liu/llava


### Training
For training details see the `pretrain.sh` and `finetune_lora.sh` scripts inside `scripts/*/`. The training code is for information purposes only and we won't be able help people debug problems with it for this preliminary release. More rigorous and less rushed and messy training scripts will be released in the future.


### Acknowledgements 
We would like to thank Hessian-AI for providing us with access to 8-16 A100 GPUs for a few weeks