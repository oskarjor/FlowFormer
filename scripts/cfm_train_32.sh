#!/bin/bash
#SBATCH --job-name="32x32_train_cfm"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:1            # Number of GPUs
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=03-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

export WANDB_API_KEY=$(cat ~/FlowFormer/.wandb_api_key)

python train_cfm/train.py \
	--batch_size=512 \
	--total_steps=200000 \
	--model="otcfm" \
	--save_dir="./results/otcfm/$SLURM_JOB_ID/" \
	--image_size=32 \
	--wandb_name="32x32_otcfm" \
	--wandb_project="flowformer" \
	--wandb_entity="oskarjor" \
	--use_wandb=True \
	--class_conditional=True
