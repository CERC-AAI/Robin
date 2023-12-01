#!/bin/bash

python -m robin.serve.cli \
    --model-path agi-collective/vicuna-7b-siglip-so400m-finetune-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" 