#!/bin/bash
#SBATCH --job-name=28i_mel-lipreg
#SBATCH --cpus-per-task=2
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -e

IMAGE=/scratch/amoskalenko/users/26a_bry/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline

configs=(
  ufbrp/presets/imagenet/lipreg.yaml
  ufbrp/presets/imagenet/lipreg-adv.yaml
)

port=29500

for cfg in "${configs[@]}"; do
    log="ufbrp-${cfg//\//_}"
    log="${log//.yaml/}.out"

    echo "=== Run ${cfg} ==="
    echo "$SLURM_JOB_NODELIST"
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 \
        --container-image "${IMAGE}" \
        --container-mounts "${WORKDIR}:/workspace/ufbrp" \
        bash -c "
            set -e
            cd /workspace/ufbrp
            apt update
            apt install -y libopencv-dev pkg-config libturbojpeg-dev
            pip install -q -r requirements.txt
            pip install ffcv
            pip install -e .
            export OMP_NUM_THREADS=8
            torchrun --nproc_per_node=8 --master_port=${port} ufbrp/train_dist.py --config ${cfg}
        " >> "${log}" 2>&1 &

    ((port++))
    echo $port

done

wait
echo $?
