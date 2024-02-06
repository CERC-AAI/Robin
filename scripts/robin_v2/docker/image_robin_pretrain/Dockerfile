FROM docker.io/library/robin_env

WORKDIR /app

RUN mkdir /app/wandb_cache /app/downloaded_models /export

# Data setup
#ADD LLaVA-Pretrain.tar.gz /app/
RUN git lfs install \
	&& git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain \
	&& cd LLaVA-Pretrain \
	&& rm -rf .git* \
	&& unzip -q images.zip -d images \
	&& rm images.zip

# launch script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

