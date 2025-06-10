#!/bin/bash
#SBATCH --job-name="256->512_train_sr_nearest"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --constraint=gpu80g
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

pwd
export WANDB_API_KEY=$(cat ~/FlowFormer/.wandb_api_key)

python train_SR/train.py \
	--batch_size=16 \
	--total_steps=800001 \
	--model="otcfm" \
	--save_dir="./results/SR/$SLURM_JOB_ID/" \
	--pre_image_size=256 \
	--post_image_size=512 \
	--save_step=10000 \
	--print_step=1000 \
	--lr=1e-4 \
	--wandb_name="256->512_otcfm_SR_$SLURM_JOB_ID" \
	--wandb_project="flowformer" \
	--wandb_entity="oskarjor" \
	--use_wandb=True \
	--class_conditional=True \
	--use_amp=True \
	--unet_conf="normal" \
	--naive_upscaling="nearest" \
	--damage_ratio=0.0 \
	--method="dopri5" \
	--time_steps=4 \
	--debug=False \
	--robust_augmentations=True \
	--augment_prob=0.7
