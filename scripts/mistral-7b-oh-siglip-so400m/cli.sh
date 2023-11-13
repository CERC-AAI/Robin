#!/bin/bash

IMG="https://images.ctfassets.net/lzny33ho1g45/6FwyRiw9nZDf9rgwIN4zPC/b7e248b756f6e0e83d33a2a19f29558b/full-page-screenshots-in-chrome-03-developer-menu-screenshot.png"  # screenshot (it understands the big text!)
# IMG="https://pbs.twimg.com/media/F-XUy09WMAA8SWC?format=png"  # meme
# IMG="https://media.geeksforgeeks.org/wp-content/uploads/20200611183120/1406-7.png"  # equation (it can't do it)


python -m llava.serve.cli \
    --conv-mode vicuna_v1 \
    --model-path ~/ws/trained_models/mistral-7b-oh-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file $IMG


# python -m llava.serve.cli \
#     --conv-mode conv_mpt \
#     --model-path ~/ws/trained_models/mistral-7b-oh-siglip-so400m-finetune-lora \
#     --model-base teknium/OpenHermes-2.5-Mistral-7B \
#     --image-file $IMG