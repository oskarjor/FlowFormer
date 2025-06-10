#!/bin/bash
#SBATCH --job-name="profile_var_d30"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

python calculate_efficiency/FLOPs_VAR.py \
	--model_depth=36 \
	--vae_ckpt="vae_ch160v4096z32.pth" \
	--var_ckpt="var_d36.pth" \
	--seed=0 \
	--cfg=1.5 \
	--more_smooth=False \
	--batch_size=8
