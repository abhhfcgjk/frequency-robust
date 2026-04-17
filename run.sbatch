#!/bin/bash
#SBATCH --job-name=28i_mel_convnext
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodelist=cn[1-12,14-19,21-30]


set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/ConvNeXt
DATASET=/scratch/amoskalenko/users/28i_mel/datasets/imagenet

echo "=== Run ${cfg} ==="
echo "$SLURM_JOB_NODELIST"
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-node=8 \
        --container-image "${IMAGE}" \
        --container-mounts "${WORKDIR}:/workspace",${DATASET}:/dataset \
        bash -c "
            set -e
            cd /workspace
            apt -qq update > /dev/null

            source /opt/miniconda3/etc/profile.d/conda.sh
            conda create -n convnext python=3.11 -y || true
            conda activate convnext

            which python
            python --version

            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            pip install timm==0.4.12 tensorboardX six
            pip install ptwt

            echo 'Start training'
            python -m torch.distributed.launch --nproc_per_node=8 main.py \
                --model lipreg_aa --drop_path 0.5 \
                --batch_size 64 --lr 4e-3 --update_freq 4 \
                --model_ema true --model_ema_eval true \
                --data_path /dataset --output_dir output
        " > "output.log" 2>&1 

echo $?
