FROM docker.io/library/robin_env

WORKDIR /app

RUN mkdir /app/wandb_cache /app/downloaded_models /export

# Data setup
ADD LLaVA-Finetune.tar.gz /app/

# to do by hand: https://github.com/haotian-liu/LLaVA#visual-instruction-tuning
# mkdir /app/LLaVA-Finetune && cd /app/LLaVA-Finetune
# wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json

# mkdir coco gqa ocr_vqa textvqa vg

# cd coco
# wget images.cocodataset.org/zips/train2017.zip
# unzip -q train2017.zip
# rm *.zip && cd ..

# cd gqa
# wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
# unzip -q images.zip
# rm *.zip && cd ..

# download OCR

# cd textvqa
# wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
# unzip -q train_val_images.zip
# rm *.zip && cd ..

# cd vg
# wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
# unzip -q images.zip
# rm *.zip && cd ..

# cd vg
# wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
# unzip -q images2.zip
# rm *.zip && cd ..


# launch script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

