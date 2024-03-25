FROM docker.io/library/robin_env

WORKDIR /app

RUN mkdir /app/downloaded_models /export

# Data setup
ADD scienceqa.tar.gz /app/playground/data/eval
ADD gqa.tar.gz /app/playground/data/eval

# RUN git lfs install \
#     && cd /app/downloaded_models \
#     && git clone https://huggingface.co/agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
#     && git clone https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B


# launch script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

