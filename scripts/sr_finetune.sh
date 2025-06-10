#!/bin/bash
#SBATCH --job-name="256->512_finetune_d30_N_sr_lw_step_0"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs 
#SBATCH --constraint=gpu80g
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=01-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

pwd
export WANDB_API_KEY=$(cat ~/FlowFormer/.wandb_api_key)

python train_SR/finetune_V2.py \
	--json_path="./results/SR/22862218_lightweight/flags.json" \
	--save_dir="./temp/$SLURM_JOB_ID/" \
	--model_path="./results/SR/22862218_lightweight/otcfm_256_to_512_weights_step_0.pt" \
	--input_data_path="./var_d30_imagenet/" \
	--target_data_path="./imagenet/" \
	--batch_size=16 \
	--num_workers=4 \
	--total_steps=100000 \
	--save_step=5000 \
	--use_wandb=True \
	--use_amp=True \
	--ot_mapping_path="./output/datasetfile/bs_200/ot_mappings.pkl" \
	--learning_rate=0.0001
