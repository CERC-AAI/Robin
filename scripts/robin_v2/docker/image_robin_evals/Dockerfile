FROM docker.io/library/robin_env

WORKDIR /app

RUN mkdir -p /app/playground/data/eval /app/downloaded_models /export

# Data setup 
# grab sqa data
ADD scienceqa.tar.gz /app/playground/data/eval/

# launch script
# launch sqa eval
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

