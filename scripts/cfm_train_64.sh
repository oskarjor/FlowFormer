#!/bin/bash
#SBATCH --job-name="64x64_train_cfm"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --constraint="a100|h100"
#SBATCH --constraint="gpu80g"
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=07-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=temp-%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

export WANDB_API_KEY=$(cat ~/FlowFormer/.wandb_api_key)

python train_cfm/train.py \
	--num_channel=192 \
	--batch_size=96 \
	--total_steps=400001 \
	--save_step=20000 \
	--model="fm" \
	--save_dir="./results/$SLURM_JOB_ID/" \
	--image_size=64 \
	--lr=1e-4 \
	--wandb_project="flowformer" \
	--wandb_entity="oskarjor" \
	--use_wandb=False \
	--class_conditional=True \
	--dataset="imagenet" \
	--debug=False
