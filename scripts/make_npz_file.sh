#!/bin/bash
#SBATCH --job-name="make_npz_file"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

export PATH_TO_DATASET="/cluster/home/oskarjor/FlowFormer/data/imagenet_512"
export PATH_TO_OUTPUT="/cluster/home/oskarjor/FlowFormer/data/imagenet_512_npz"

python -m train_SR.make_npz_file \
    $PATH_TO_DATASET \
    $PATH_TO_OUTPUT
