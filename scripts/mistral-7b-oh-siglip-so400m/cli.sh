#!/bin/bash

export TRANSFORMERS_CACHE=/localdisks/rogeralexis/downloaded_models/hf_cache
export HF_HOME=/localdisks/rogeralexis/downloaded_models/hf_cache



#IMG="https://images.ctfassets.net/lzny33ho1g45/6FwyRiw9nZDf9rgwIN4zPC/b7e248b756f6e0e83d33a2a19f29558b/full-page-screenshots-in-chrome-03-developer-menu-screenshot.png"  # screenshot (it understands the big text!)
# IMG="https://pbs.twimg.com/media/F-XUy09WMAA8SWC?format=png"  # meme
# IMG="https://media.geeksforgeeks.org/wp-content/uploads/20200611183120/1406-7.png"  # equation (it can't do it)
IMG="https://cdn.discordapp.com/attachments/1197300065119781007/1197300370234413106/7c3a537d7a775e35318148ca70b77716.png?ex=65c3fe5d&is=65b1895d&hm=56e7f751cfd10bdbb87bd72c04d9bf5b95309028da91ff842079d86358ae12da&"
CM=vicuna_v1  # this is the standard in llava and what we've trained with


python -m robin.serve.cli \
    --conv-mode $CM \
    --model-path agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file $IMG
