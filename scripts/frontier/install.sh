# This is an install guide for the current codebase on the Frontier supercomputer

# Everything will go into your base directory, please keep the same to simplify debugging

BASE_DIR=/lustre/orion/csc538/scratch/$(whoami)

### General setup steps
mkdir $BASE_DIR
cd $BASE_DIR

mkdir checkpoints hostfiles job_logs miopen wandb_cache


cat <<frontier_write_hostfile >> frontier_write_hostfile.sh
#!/bin/bash
numbers=$(echo "$SLURM_NODELIST" | grep -o '[0-9]\+')

get_hostfiles() {
for num in $numbers
do
    echo "frontier$num slots=8"
done
}

get_hostfiles > /lustre/orion/csc538/scratch/$(whoami)/hostfiles/$SLURM_JOBID-hosts
frontier_write_hostfile


### MINICONDA
printf 'Do you want to install Miniconda? (y/N) '
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then 
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh 
else
    echo Moving on
fi


### Conda env
conda create -n robin python=3.10 -y 
conda activate robin
pip install --upgrade pip


### Install torch
module load rocm/5.4.3 
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/rocm5.4.2


### Flash attention
printf 'Do you want to install Flash attention? (y/N) '
read answer

cd $BASE_DIR
if [ "$answer" != "${answer#[Yy]}" ] ;then 
    conda activate robin
    module load rocm/5.4.3

    git clone https://github.com/ROCmSoftwarePlatform/flash-attention
    cd flash-attention
    git submodule sync
    git submodule update --init --recursive
    export GPU_ARCHS="gfx90a"
    pip install einops packaging

    patch "/lustre/orion/csc538/scratch/$(whoami)/miniconda3/envs/robin/lib/python3.10/site-packages/torch/utils/hipify/hipify_python.py" hipify_patch.patch

    # edit line 586 and 587 “std” -> “thrust”
    sed -i 's/using std::remove_cvref/using thrust::remove_cvref/g' /lustre/orion/csc538/scratch/$(whoami)/miniconda3/envs/robin/lib/python3.10/site-packages/torch/include/pybind11/detail/common.h

    pip install .

    echo 
    echo to validate: PYTHONPATH=$PWD python benchmarks/benchmark_flash_attention.py
    echo Flash2 should be about 2 times faster than Pytorch 

    echo
    echo Kindly ask Torch to not check the Flash_attn version:
    echo Edit /lustre/orion/csc538/scratch/alexisroger/miniconda3/envs/robin2/lib/python3.10/site-packages/transformers/modeling_utils.py
    echo Remove lines 1272 - 1283 (if not is_flash_attn_2_available(): etc.)

else
    echo Moving on
fi


### Codebase seetup
cd $BASE_DIR
git clone https://github.com/AGI-Collective/robin
cd robin
git checkout Frontier
pip install -e ".[train]"
pip uninstall bitsandbytes 

### Run
echo Run the following to start training:
echo sbatch scripts/v1_5/pretrain_multinodes.sh
