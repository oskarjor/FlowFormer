#!/bin/bash
#SBATCH --job-name="256->512_train_sr"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --constraint=gpu80g
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

python train_SR/get_static_dataset.py \
	--save_dir="./output/datasetfile/$SLURM_JOB_ID" \
	--input_data_path="./var_d16_imagenet/" \
	--target_data_path="./imagenet/" \
	--batch_size=4 \
	--num_workers=4 \
	--model="otcfm" \
	--upscaling_mode="lanczos"