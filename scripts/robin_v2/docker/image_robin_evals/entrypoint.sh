#!/bin/bash
MODEL=$1
BASE=$2
version=""

# download models
git lfs install
cd /app/downloaded_models
git clone https://huggingface.co/$MODEL
git clone https://huggingface.co/$BASE

#get updated code
cd /app/robin
git pull
git checkout eval

# setup variables
MODEL=${MODEL##*/}
BASE=${BASE##*/}

EXPORT_PATH=/export/$MODEL

MODEL=/app/downloaded_models/$MODEL
BASE=/app/downloaded_models/$BASE

mkdir -p $EXPORT_PATH

# launch training
cd /app/robin/scripts/v1_5/eval
echo "Launching SQA"
./docker_sqa.sh $MODEL $BASE $version > $EXPORT_PATH/scienceqa/results.log 2>&1

echo "Launching QGA"
./docker_gqa.sh $MODEL $BASE $version > $EXPORT_PATH/qga/results.log 2>&1

echo "Finished evals!"