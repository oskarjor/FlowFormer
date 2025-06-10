#!/bin/bash
#SBATCH --job-name="naive_upscale_sample"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-02:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

pwd

python train_SR/naive_upscale.py \
	--save_dir="./output/naive/$SLURM_JOB_ID" \
	--data_path="./var_d30_imagenet/" \
	--naive_upscaling="lanczos" \
	--file_format="png" \
	--final_size=299
