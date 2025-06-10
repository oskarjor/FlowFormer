#!/bin/bash
#SBATCH --job-name="get_similar_pairs"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --output=illustrations_%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

python illustrations/get_similar_images.py \
    --save_dir output/illustrations/similar_pairs/$SLURM_JOB_ID \
    --input_data_path ./var_d30_imagenet \
    --target_data_path ./imagenet \
    --batch_size 4 \
    --num_workers 4 \
    --naive_upscaling="lanczos" \
    --class_idx=0
