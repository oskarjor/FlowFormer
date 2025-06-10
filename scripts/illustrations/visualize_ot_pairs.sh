#!/bin/bash
#SBATCH --job-name="visualize_ot_pairs"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=20G
#SBATCH --output=%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

pwd

python illustrations/visualize_ot_pairs.py \
	--output_dir="./images/ot_pairs/$SLURM_JOB_ID" \
	--input_data_path="./var_d30_imagenet/val_cfg_1_5" \
	--target_data_path="./imagenet/val" \
	--num_pairs=16 \
	--num_classes=4 \
	--upscaling_mode="nearest" \
	--mapping_file="output/datasetfile/bs_200/ot_mappings.pkl"
