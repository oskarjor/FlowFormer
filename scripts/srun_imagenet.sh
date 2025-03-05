#!/bin/bash
#SBATCH --job-name="train_VAR_imagenet"
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00-00:30:00
#SBATCH --output=train_VAR_imagenet_%j.out
#SBATCH --mail-user=oskarjor@ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load Python/3.10.8-GCCcore-12.2.0
source /cluster/home/oskarjor/.virtualenv/flowformer/bin/activate

# Debug info
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Set unique port based on job ID
export MASTER_PORT=$(expr 29500 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

srun python train_VAR/train.py \
  --depth=16 \
  --bs=768 \
  --ep=200 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --data_path=imagenet \
  --workers=2