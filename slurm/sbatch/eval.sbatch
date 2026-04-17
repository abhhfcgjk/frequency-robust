#!/bin/bash
#SBATCH --job-name=28i_mel_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --mem=0
#SBATCH --nodelist=cn[1-12,14-19,21-30]


set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline
TRAIN_CONFIG=validation.yaml
GPUS_PER_MODEL=${GPUS_PER_MODEL:-4}
WORKERS_PER_RANK=${WORKERS_PER_RANK:-2}

configs=(
  ufbrp/presets/imagenet/lipreg-adv_copy.yaml
  ufbrp/presets/imagenet/lipreg_aa-adv.yaml 
)

LOCAL_BASE=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p $LOCAL_BASE
echo "Allocated GPUs:"
nvidia-smi

echo "Launching container..."

export CONFIG_LIST="${configs[*]}"
export GPUS_PER_MODEL WORKERS_PER_RANK LOCAL_BASE TRAIN_CONFIG

srun --ntasks=1 -G 8 \
     --container-image "${IMAGE}" \
     --container-mounts "${WORKDIR}:/workspace/ufbrp",${LOCAL_BASE}:${LOCAL_BASE} \
     bash -lc '

set -e
cd /workspace/ufbrp

apt -qq update > /dev/null
apt install -y -qq libopencv-dev pkg-config libturbojpeg-dev ca-certificates > /dev/null
update-ca-certificates > /dev/null

python -m pip install --upgrade pip
python -m pip install -q -r requirements_ffcv.txt
python -m pip install -e .

SHARED_FFCV=${LOCAL_BASE}/ffcv_shared
mkdir -p ${SHARED_FFCV}

echo "=== Writing shared validation dataset once ==="
export IMAGENET_DIR=data/imagenet
export WRITE_DIR=${SHARED_FFCV}
bash ufbrp/write_imagenet_val.sh 500 0.50 90

TOTAL_GPUS=$(python - <<'"'"'PY'"'"'
import torch
print(torch.cuda.device_count())
PY
)
MAX_PARALLEL=$((TOTAL_GPUS / GPUS_PER_MODEL))
ALL_CONFIGS=(${CONFIG_LIST})

if [ ${#ALL_CONFIGS[@]} -gt ${MAX_PARALLEL} ]; then
    echo "Warning: configs=${#ALL_CONFIGS[@]}, but only ${MAX_PARALLEL} fit with GPUS_PER_MODEL=${GPUS_PER_MODEL}. Running first ${MAX_PARALLEL}."
fi

for idx in "${!ALL_CONFIGS[@]}"; do
    if [ ${idx} -ge ${MAX_PARALLEL} ]; then
        break
    fi
    cfg=${ALL_CONFIGS[$idx]}
    GPU_START=$((idx * GPUS_PER_MODEL))
    GPU_END=$((GPU_START + GPUS_PER_MODEL - 1))
    GPU_LIST=$(seq -s, ${GPU_START} ${GPU_END})

    LOCAL=${LOCAL_BASE}/job_${idx}
    mkdir -p ${LOCAL}

    log_name=$(echo ${cfg} | sed "s/\//_/g" | sed "s/.yaml//")
    ckpt_dir=logs/${log_name}
    mkdir -p ${ckpt_dir}

    echo "=== Start evaluation for ${cfg} on GPUs [${GPU_LIST}] ==="

    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2

    CUDA_VISIBLE_DEVICES=${GPU_LIST} \
    NUMBA_THREADING_LAYER=omp \
    python ufbrp/eval_ffcv.py \
        --config-file ufbrp/presets/ffcv/${TRAIN_CONFIG} \
        --data.config=${cfg} \
        --data.val_dataset=${SHARED_FFCV}/val.ffcv \
        --data.num_workers=${WORKERS_PER_RANK} \
        --data.in_memory=1 \
        --training.distributed=1 \
        --dist.world_size=${GPUS_PER_MODEL} \
        --dist.port=$((12355 + idx)) \
        --logging.folder=${ckpt_dir} \
        > ${ckpt_dir}/log-eval-copy2.out 2>&1 &

    sleep 1
done

wait
echo "All evaluations finished."
'
