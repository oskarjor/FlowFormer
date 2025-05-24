#!/bin/bash
#SBATCH --job-name="sample_var"   # Sensible name for the job
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

python train_VAR/sample.py \
	--model_depth=16 \
	--vae_ckpt="vae_ch160v4096z32.pth" \
	--var_ckpt="var_d16.pth" \
	--output_dir="./results/VAR/$SLURM_JOB_ID/" \
	--seed=0 \
	--cfg=4.0 \
	--more_smooth=False \
	--batch_size=16