mkdir $1/playground

#Initial setup
mkdir $1/playground/data
mkdir $1/playground/data/eval
wget https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=drive_link -P $1/playground/data/eval
unzip -q eval.zip


#VQAv2

wget http://images.cocodataset.org/zips/test2015.zip -P $1/playground/data/eval/vqav2
unzip -q test2015.zip

#SQA
#wget https://github.com/lupantech/ScienceQA/edit/main/data/scienceqa/pid_splits.json -P $1/playground/data/eval/scienceqa
#wget https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json -P $1/playground/data/eval/scienceqa
#/bin/bash download_sqa.sh $1/playground/data/eval/scienceqa



#TextVQA/VQAT


#wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P $1/playground/data/eval/textvqa
#wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P $1/playground/data/eval/textvqa
#cd $1/playground/data/eval/textvqa
#unzip -q train_val_images.zip




#MM-VET


mkdir $1/playground/data/eval/mmvet
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -P  $1/playground/data/eval/mmvet
cd  $1/playground/data/eval/mmvet
unzip -q mm-vet.zip
