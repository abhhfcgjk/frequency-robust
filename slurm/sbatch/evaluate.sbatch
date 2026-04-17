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

configs=(
  ufbrp/presets/imagenet/resnet50_aa-adv.yaml \
  ufbrp/presets/imagenet/lipreg.yaml \
  ufbrp/presets/imagenet/lipreg_copy.yaml \
  ufbrp/presets/imagenet/lipreg-adv.yaml \
  ufbrp/presets/imagenet/lipreg-adv_copy.yaml \
  ufbrp/presets/imagenet/lipreg_aa-adv.yaml \
  ufbrp/presets/imagenet/lipreg_aa-adv_copy.yaml
)

LOCAL_BASE=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p $LOCAL_BASE
echo "Allocated GPUs:"
nvidia-smi

echo "Launching container..."

CONFIG_LIST="${configs[*]}"

srun --ntasks=1 -G 8 \
     --container-image "${IMAGE}" \
     --container-mounts "${WORKDIR}:/workspace/ufbrp",${LOCAL_BASE}:${LOCAL_BASE} \
     bash -c "

set -e
cd /workspace/ufbrp

apt -qq update > /dev/null
apt install -y -qq libopencv-dev pkg-config libturbojpeg-dev ca-certificates > /dev/null
update-ca-certificates > /dev/null

python -m pip install --upgrade pip
python -m pip install -q -r requirements_ffcv.txt
python -m pip install -e .

GPU_ID=0

for cfg in ${CONFIG_LIST}; do

    LOCAL=${LOCAL_BASE}/gpu_\${GPU_ID}
    mkdir -p \${LOCAL}/ffcv

    log_name=\$(echo \${cfg} | sed 's/\//_/g' | sed 's/.yaml//')
    ckpt_dir=logs/\${log_name}
    mkdir -p \${ckpt_dir}

    echo \"=== Writing dataset for \${cfg} (GPU \${GPU_ID}) ===\"

    export IMAGENET_DIR=data/imagenet
    export WRITE_DIR=\${LOCAL}/ffcv
    bash ufbrp/write_imagenet_val.sh 500 0.50 90

    echo \"=== Start evaluation on GPU \${GPU_ID} ===\"

    CUDA_VISIBLE_DEVICES=\${GPU_ID} \
    NUMBA_THREADING_LAYER=omp \
    python ufbrp/eval_ffcv.py \
        --config-file ufbrp/presets/ffcv/${TRAIN_CONFIG} \
        --data.config=\${cfg} \
        --data.val_dataset=\${LOCAL}/ffcv/val.ffcv \
        --data.num_workers=16 \
        --data.in_memory=1 \
        --dist.world_size=1 \
        --logging.folder=\${ckpt_dir} \
        > \${ckpt_dir}/log-eval-copy.out 2>&1 &

    GPU_ID=\$((GPU_ID + 1))
    sleep 3
done

wait
echo 'All evaluations finished.'
"