#!/bin/bash

IMG="https://images.ctfassets.net/lzny33ho1g45/6FwyRiw9nZDf9rgwIN4zPC/b7e248b756f6e0e83d33a2a19f29558b/full-page-screenshots-in-chrome-03-developer-menu-screenshot.png"  # screenshot (it understands the big text!)
# IMG="https://pbs.twimg.com/media/F-XUy09WMAA8SWC?format=png"  # meme
# IMG="https://media.geeksforgeeks.org/wp-content/uploads/20200611183120/1406-7.png"  # equation (it can't do it)

python -m robin.serve.cli \
    --model-path agi-collective/vicuna-7b-clip-finetune-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --max-new-tokens 2000 \
    --image-file $IMG

# Argument removed during HF tests (would not work with)
    # --conv-mode vicuna_v1 \

# python -m robin.serve.cli \
#     --conv-mode vicuna_v1 \
#     --model-path ~/ws/trained_models/vicuna-7b-clip-finetune-lora \
#     --model-base vicuna-7b-v1.5 \
#     --image-file "https://pbs.twimg.com/media/F-XUy09WMAA8SWC?format=png" 
