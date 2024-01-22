module load StdEnv/2020
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.10.2
module load rust/1.70.0

BASE_DIR=/scratch/$(whoami)

cd $BASE_DIR

mkdir hostfiles job_logs wandb_cache
mkdir -p downloaded_models/hf_cache

# create hostfile
cat << cedar_write_hostfile >> cedar_write_hostfile.sh
#!/bin/bash

get_hostfiles() {
    numbers=\$(echo "\$SLURM_NODELIST")
    numbers="\${numbers##*[}"
    numbers="\${numbers%%]*}"

    IFS=',' read -ra nodes <<< "\$numbers"

    for node in "\${nodes[@]}"; do
        if [[ \$node == *-* ]]; then
            start=\$((10#\${node%-*}))
            end=\$((10#\${node#*-}))
            for ((i=start; i<=end; i++)); do
                echo "cdr\$i slots=4"
            done
        else
            echo "cdr\$node slots=4"
        fi
    done
}

get_hostfiles > /scratch/$(whoami)/hostfiles/$SLURM_JOBID-hosts
cedar_write_hostfile


# repo setup
cd $BASE_DIR
git clone https://github.com/AGI-Collective/robin
cd robin
git checkout frontier/dev


# python setup
python -m venv $BASE_DIR/robin_venv

source $BASE_DIR/robin_venv/bin/activate
pip install --upgrade pip
pip install -e ".[train]"

# small tweeks that are needed to function properly
pip uninstall bitsandbytes 
rm -rf $BASE_DIR/robin_venv/lib/python3.10/site-packages/triton
rm -rf $BASE_DIR/robin_venv/lib/python3.10/site-packages/sklearn

# If you want to try using Flash Attention:
# pip install flash-attn==2.3.3

# downloading models
cd $BASE_DIR/downloaded_models
module load git-lfs
git lfs install
git clone https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
git clone https://huggingface.co/apple/DFN2B-CLIP-ViT-L-14

# removing git files due to space and number of files limitations
rm -rf OpenHermes-2.5-Mistral-7B/.git* 
rm -rf DFN2B-CLIP-ViT-L-14/.git*

# download data
echo "download data"
