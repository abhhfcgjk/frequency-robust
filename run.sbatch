#!/bin/bash
#SBATCH --job-name=28i_mel_wideresnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=cn[1-12,14-19,21-30]

set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline

CONFIGS=(
ufbrp/presets/cifar10/wideresnet_lipreg_aa-adv.yaml
ufbrp/presets/cifar100/wideresnet_lipreg_aa-adv.yaml
)

LOGS=(
wideresnet-cifar10-lipreg.log
wideresnet-cifar100-lipreg.log
)

srun -n 1 \
     -G 4 \
     --container-image "${IMAGE}" \
     --container-mounts "${WORKDIR}:/workspace" \
     bash -lc '
set -euo pipefail
cd /workspace

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

CONFIGS=(
  "ufbrp/presets/cifar10/wideresnet_lipreg_aa-adv.yaml"
  "ufbrp/presets/cifar100/wideresnet_lipreg_aa-adv.yaml"
)

LOGS=(
  "wideresnet-cifar10-lipreg.log"
  "wideresnet-cifar100-lipreg.log"
)

ID=${SLURM_PROCID}
LOCAL_ID=${SLURM_LOCALID}
NTASKS=${SLURM_NTASKS}
NCONFIGS=${#CONFIGS[@]}
k=0

for i in {0..1}; do
    export CUDA_VISIBLE_DEVICES=${k}
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    CONFIG=${CONFIGS[$i]}
    LOG=${LOGS[$i]}

    echo "Task ${ID} running ${CONFIG} -> ${LOG}"
    python ufbrp/train.py --config "${CONFIG}" > "${LOG}" 2>&1 &
    k=$((k + 1))
done
wait 
'
