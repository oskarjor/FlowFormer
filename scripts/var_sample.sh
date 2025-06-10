#!/bin/bash
#SBATCH --job-name="sample_var_cfg_1_5_d36"   # Sensible name for the job
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

# flash_attn and fused_mlp should only be true if running on h100
python train_VAR/sample.py \
	--model_depth=36 \
	--vae_ckpt="vae_ch160v4096z32.pth" \
	--var_ckpt="var_d36.pth" \
	--job_id="$SLURM_JOB_ID" \
	--seed=0 \
	--num_sampling_steps=250 \
	--cfg=1.5 \
	--num_classes=1000 \
	--num_samples_per_class=50 \
	--more_smooth=False \
	--debug=True \
	--flash_attn=False \
	--fused_mlp=True \
	--batch_size=16 \
	--split="val_cfg_1_5" \
	--shared_aln=True
