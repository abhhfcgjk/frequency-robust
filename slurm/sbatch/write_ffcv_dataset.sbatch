#!/bin/bash
#SBATCH --job-name=28i_mel
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodelist=cn[1-12,14-19,21-30]

set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline
PYTHON_VERSION=3.11
OUTPUT_LOGS=ffcv.out

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --gpus-per-node=8 \
        --container-image "${IMAGE}" \
        --container-mounts "${WORKDIR}:/workspace/ufbrp" \
        bash -c "
                set -e
                cd /workspace/ufbrp
                apt -qq update > /dev/null
                apt install -y -qq libopencv-dev pkg-config libturbojpeg-dev > /dev/null

                echo 'create conda python=${PYTHON_VERSION}'
                source /opt/miniconda3/etc/profile.d/conda.sh
                conda create -n ffcv python=${PYTHON_VERSION} -y || true 2> /dev/null
                conda activate ffcv > /dev/null
                which python
                python --version

                echo 'pip install -r requirements.txt'
                pip install -r requirements_ffcv.txt
                pip install -e .

                echo 'Write dataset to .ffcv format'
                pip install natsort
                export IMAGENET_DIR=data/imagenet
                export WRITE_DIR=data/imagenet/ffcv_smart
                bash ufbrp/write_imagenet.sh 500 0.50 90
        " > "${OUTPUT_LOGS}" 2>&1 &
wait