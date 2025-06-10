#!/bin/bash
#SBATCH --job-name="256->512_sample_sr_n_d16"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs 
#SBATCH --constraint=gpu80g
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-12:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

pwd
export WANDB_API_KEY=$(cat ~/FlowFormer/.wandb_api_key)

python train_SR/sample_from_VAR.py \
	--json_path="./results/SR/22629494_nearest/flags.json" \
	--save_dir="./output/SR/$SLURM_JOB_ID" \
	--model_path="./results/SR/22629494_nearest/otcfm_256_to_512_weights_step_400000.pt" \
	--data_path="var_d16_imagenet/" \
	--time_steps=4 \
	--batch_size=32 \
	--num_workers=4 \
	--split="val" \
	--atol=0.0001 \
	--rtol=0.0001 \
	--ode_method="dopri5"
