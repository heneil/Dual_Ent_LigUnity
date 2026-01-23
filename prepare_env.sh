# source $(conda info --base)/etc/profile.d/conda.sh
# mamba env create -f ./env.yml
conda activate l_core
# pip install -e .

# export DATA_PATH=/data/proteina_data
# export DATA_PATH=/taiga/illinois/eng/cs/geliu/proteina_data

if [ "$(hostname)" = "sn4622122626" ]; then
    # ge-01
    export DATA_PATH=/mnt/data_from_server2/yanruqu2/LigUnityData
    export CKPT_PATH=/mnt/data_from_server2/yanruqu2/LigUnityCKPT
    export HF_HOME=/data/huggingface
elif [ "$(hostname)" = "sn4622122627" ]; then
    # ge-02
    export DATA_PATH=/data/yanruqu2/LigUnityData
    export CKPT_PATH=/data/yanruqu2/LigUnityCKPT
    export HF_HOME=/data/huggingface
else
    # NCSA / Slurm cluster
    export DATA_PATH=/taiga/illinois/eng/cs/geliu/ligunity_data
    export CKPT_PATH=/projects/bdfh/kevinqu/Dual_Ent_LigUnity/checkpoints
    export HF_HOME=/taiga/illinois/eng/cs/geliu/huggingface
fi

echo "DATA_PATH=${DATA_PATH}" > .env
echo "CKPT_PATH=${CKPT_PATH}" >> .env
# export HF_HOME=/projects/bdfh/kevinqu/hf_home

which conda
which python

export CUDA_VISIBLE_DEVICES="0,1"
echo $CUDA_VISIBLE_DEVICES
