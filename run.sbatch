#!/bin/bash
#SBATCH --job-name=28i_mel_swin
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodelist=cn[1-12,14-19,21-30]


set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/Swin-Transformer
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
            conda create -n swin python=3.11 -y || true
            conda activate swin

            pip install --upgrade pip
            # install pytorch first
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            pip install timm==0.4.12
            pip install opencv-python termcolor yacs pyyaml scipy
            pip install ptwt

            cd /workspace/kernels/window_process
            python setup.py install
            cd /workspace/

            echo 'Start training'
            python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  \
                main.py --cfg configs/lipreg_aa/swin_base_patch4_window7_224.yaml \
                --data-path /dataset --batch-size 64 --optim adamw_aa \
                --accumulation-steps 2 \
                --use-checkpoint
        " > "output.log" 2>&1 

echo $?

# python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/lipreg_aa/swin_base_patch4_window7_224.yaml --data-path /dataset --batch-size 64 --optim adamw_aa --accumulation-steps 2 --use-checkpoint 