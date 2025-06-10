#!/bin/bash
#SBATCH --job-name="make_npz_file"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --cpus-per-task=8              # Number of tasks
#SBATCH --time=00-08:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --output=make_npz_%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

export PATH_TO_DATASET="/cluster/home/oskarjor/FlowFormer/output/SR/22878823/val_compressed"
export PATH_TO_OUTPUT="/cluster/home/oskarjor/FlowFormer/output/SR/22878823/FID_stats"

# Get the number of CPU cores available
NUM_CORES=$(nproc)
# Use 75% of available cores to leave some resources for other processes
NUM_WORKERS=$((NUM_CORES * 3 / 4))

python -m pytorch_fid \
	--num-workers $NUM_WORKERS \
	--save-stats \
	$PATH_TO_DATASET \
	$PATH_TO_OUTPUT
