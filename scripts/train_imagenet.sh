#!/bin/bash
#SBATCH --job-name="train_VAR_imagenet"   # Sensible name for the job
#SBATCH --account=ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:4            # Number of GPUs
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=4              # Number of tasks
#SBATCH --time=00-00:30:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --output=train_VAR_imagenet.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 train_VAR/train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=imagenet