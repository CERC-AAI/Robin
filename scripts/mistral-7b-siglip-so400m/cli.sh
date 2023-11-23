#!/bin/bash

# NOT WORKING

python -m llava.serve.cli \
    --model-path agi-collective/mistral-7b-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" 