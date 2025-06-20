#!/bin/bash
#SBATCH --job-name="32->64_train_sr"   # Sensible name for the job
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

pwd
export WANDB_API_KEY=$(cat ~/FlowFormer/.wandb_api_key)

python train_SR/train.py \
	--batch_size=4 \
	--total_steps=400001 \
	--model="otcfm" \
	--save_dir="./results/SR/$SLURM_JOB_ID/" \
	--pre_image_size=32 \
	--post_image_size=64 \
	--lr=1e-4 \
	--wandb_project="flowformer_SR" \
	--wandb_entity="oskarjor" \
	--use_wandb=True \
	--class_conditional=True \
	--use_amp=True
