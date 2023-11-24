#!/bin/bash

python -m llava.serve.cli \
    --model-path agi-collective/mistral-7b-oh-siglip-so400m-frozen-ve-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" 