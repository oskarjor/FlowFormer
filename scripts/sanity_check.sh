#!/bin/bash
#SBATCH --job-name="256->512_train_sr"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:1            # Number of GPUs
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

python train_SR/sanity_check.py \
	--json_path="./results/otcfm/22635170/flags.json" \
	--model_path="./results/otcfm/22635170/otcfm_256_to_512_weights_step_400000.pt" \
	--save_dir="./output/SR/$SLURM_JOB_ID" \
	--dataset="imagenet" \
	--pre_image_size=256 \
	--post_image_size=512 \
	--batch_size=32 \
	--num_workers=4 \
	--ode_method="dopri5" \
	--atol=1e-4 \
	--rtol=1e-4