#!/bin/bash

# Define the base directory for LLaVA data.
BASE_DIR="./playground/data/eval"

# Create the base directory if it doesn't exist.
mkdir -p $BASE_DIR

# NEED TO MAKE SURE THAT YOUR RUNTIME HAS GDOWN AND DATASETS
# TO DO THAT YOU NEED PYTHON >=3.8.0

# Download and extract eval.zip.
echo "Downloading and extracting eval.zip..."
gdown 'https://drive.google.com/uc?id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy' -O eval.zip
unzip eval.zip -d $BASE_DIR
rm eval.zip

# VQAv2
wget http://images.cocodataset.org/zips/test2015.zip -P $1/playground/data/eval/vqav2
unzip -q test2015.zip


# GQA
GQA_DIR="$BASE_DIR/gqa/data"
mkdir -p $GQA_DIR

# Download and extract GQA dataset
echo "Downloading and extracting GQA dataset..."
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O gqa_images.zip
unzip gqa_images.zip -d $GQA_DIR
rm gqa_images.zip


# VisWiz
VIZWIZ_DIR="$BASE_DIR/vizwiz"
mkdir -p $VIZWIZ_DIR

# Download and extract Annotations (for test.json)
echo "Downloading and extracting Annotations for VisWiz..."
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip -O vizwiz_annotations.zip
unzip vizwiz_annotations.zip -d $VIZWIZ_DIR
rm vizwiz_annotations.zip

# Download and extract test images
echo "Downloading and extracting test images for VisWiz..."
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip -O vizwiz_test_images.zip
unzip vizwiz_test_images.zip -d $VIZWIZ_DIR/test
rm vizwiz_test_images.zip


# ScienceQA
wget https://github.com/lupantech/ScienceQA/edit/main/data/scienceqa/pid_splits.json -P $1/playground/data/eval/scienceqa
wget https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json -P $1/playground/data/eval/scienceqa
/bin/bash download_sqa.sh $1/playground/data/eval/scienceqa


# TextVQA
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P $1/playground/data/eval/textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P $1/playground/data/eval/textvqa
cd $1/playground/data/eval/textvqa
unzip -q train_val_images.zip


# POPE
POPE_DIR="$BASE_DIR/pope"
mkdir -p $POPE_DIR

# Download the 'coco' folder from the POPE GitHub repository
echo "Downloading 'coco' from POPE GitHub repository..."
svn export https://github.com/AoiDragon/POPE/trunk/output/coco $POPE_DIR


# MME
# The MME dataset is not available for direct download. 
# It must be requested via email from Xiamen University.

echo "To obtain the MME dataset, please follow these instructions:"
echo "1. Send an email to yongdongluo@stu.xmu.edu.cn to request access to the dataset."
echo "2. Ensure that your email suffix matches your affiliation (e.g., xx@stu.xmu.edu.cn for Xiamen University)."
echo "   If your email suffix does not match your affiliation, provide an explanation in your email."
echo "3. Use a real-name system for better academic communication."
echo "4. Include your affiliation and contact details in the email."

# Create directory for MME dataset
MME_DIR="$BASE_DIR/MME"
mkdir -p $MME_DIR

# Reminder for manual action required
echo "Remember to manually place the MME dataset in $MME_DIR after receiving it."


# MMBench
MMBENCH_DIR="$BASE_DIR/mmbench"
mkdir -p $MMBENCH_DIR

# Download mmbench_dev_20230712.tsv
echo "Downloading mmbench_dev_20230712.tsv for MMBench..."
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv -O $MMBENCH_DIR/mmbench_dev_20230712.tsv


# MMBench-CN
# Create directory for MMBench-CN dataset (if not already created for MMBench)
MMBENCH_DIR="$BASE_DIR/mmbench"
mkdir -p $MMBENCH_DIR

# Download mmbench_dev_cn_20231003.tsv
echo "Downloading mmbench_dev_cn_20231003.tsv for MMBench-CN..."
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv -O $MMBENCH_DIR/mmbench_dev_cn_20231003.tsv


# SEED-Bench
# TODO: 

# LLaVA-Bench-in-the-Wild
# Ensure Python and necessary packages are installed
echo "Downloading LLaVA-Bench-in-the-Wild dataset..."
python download_llava_bench.py


# MM-Vet
MMVET_DIR="$BASE_DIR/mmvet"
mkdir -p $MMVET_DIR

# Download mm-vet.zip
echo "Downloading mm-vet.zip for MM-Vet..."
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -P $MMVET_DIR

# Navigate to MM-Vet directory and unzip
cd $MMVET_DIR
unzip -q mm-vet.zip

# Additional Benchmarks (e.g., Q-Bench)
# TODO: Add similar steps for any additional benchmarks as required.

echo "All datasets downloaded and prepared."
