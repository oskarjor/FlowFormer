#!/bin/bash
#SBATCH --job-name="256->512_sanity_check_l_128"   # Sensible name for the job
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

python train_SR/sanity_check.py \
	--json_path="./results/SR/22635170_lanczos/flags.json" \
	--model_path="./results/SR/22635170_lanczos/otcfm_256_to_512_weights_step_400000.pt" \
	--save_dir="./output/sanity_check/$SLURM_JOB_ID" \
	--dataset="imagenet" \
	--pre_image_size=128 \
	--post_image_size=512 \
	--batch_size=64 \
	--num_workers=4 \
	--ode_method="dopri5" \
	--atol=1e-4 \
	--rtol=1e-4
