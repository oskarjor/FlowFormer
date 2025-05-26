#!/bin/bash
#SBATCH --job-name="get_similar_pairs"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --partition=CPUQ
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --mem=100G
#SBATCH --output=%j_illustrations.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=NONE

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

python illustrations/get_similar_images.py \
    --json_path path/to/flags.json \
    --save_dir output/illustrations/similar_pairs \
    --input_data_path path/to/input_data \
    --target_data_path path/to/target_data \
    --batch_size 16 \
    --num_workers 4
