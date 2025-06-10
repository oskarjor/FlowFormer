#!/bin/bash
#SBATCH --job-name="create_symlinks"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --output=symlink_%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

export PATH_TO_DATASET="/cluster/home/oskarjor/FlowFormer/output/SR/22878823/val"

python calculate_fid/create_symlinks.py \
    --source $PATH_TO_DATASET
