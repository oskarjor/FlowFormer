#!/bin/bash
#SBATCH --job-name="full_train_VAR_imagenet"
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01-00:00:00
#SBATCH --output=4n_VAR_imagenet_%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export MASTER_PORT=29400

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    train_VAR/train.py \
    --depth=8 \
    --bs=2560 \
    --ep=100 \
    --fp16=1 \
    --alng=1e-3 \
    --wpe=0.1 \
    --data_path=imagenet \
    --ac=2 \
    --pn=128