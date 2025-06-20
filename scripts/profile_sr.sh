#!/bin/bash
#SBATCH --job-name="profile_sr"   # Sensible name for the job
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

python calculate_efficiency/FLOPs_CFM.py \
	--json_path="./results/SR/22862218_lightweight/flags.json" \
	--model_path="./results/SR/22862218_lightweight/otcfm_256_to_512_weights_step_100000.pt" \
	--batch_size=8 \
	--time_steps=50
