#!/bin/bash

python -m robin.serve.cli \
    --model-path agi-collective/mistral-7b-siglip-so400m-finetune-lora \
    --model-base mistralai/Mistral-7B-v0.1 \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" 