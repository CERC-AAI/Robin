#!/bin/bash

python -m llava.serve.cli \
    --model-path ~/ws/trained_models/vicuna-7b-siglip-so400m-finetune-lora \
    --model-base ./hf/vicuna-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" 