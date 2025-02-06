#!/bin/bash
#SBATCH --job-name="train_cifar10"   # Sensible name for the job
#SBATCH --account=ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --output=train_cifar10.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

python train_cifar10.py