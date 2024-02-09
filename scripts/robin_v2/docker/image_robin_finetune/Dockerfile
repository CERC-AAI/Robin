FROM docker.io/library/robin_env

WORKDIR /app

RUN mkdir /app/wandb_cache /app/downloaded_models /export

# Data setup
# ADD LLaVA-Finetune.tar.gz /app/
COPY data_download_parallel.py /app/
RUN apt update && apt-get install imagemagick -y
RUN python /app/data_download_parallel.py

# launch script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

