FROM docker.io/library/robin_env

WORKDIR /app

RUN mkdir /app/downloaded_models /export

# Data setup
ADD scienceqa.tar.gz /app/playground/data/eval

# launch script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

