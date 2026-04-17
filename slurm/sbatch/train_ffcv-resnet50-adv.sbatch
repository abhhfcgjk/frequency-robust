#!/bin/bash
#SBATCH --job-name=28i_mel
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodelist=cn[1-12,14-19,21-30]


set -e

IMAGE=/scratch/amoskalenko/users/28i_mel/base270.sqsh
WORKDIR=/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline
TRAIN_CONFIG=rn50_88_epochs.yaml
configs=(
  ufbrp/presets/imagenet/resnet50-adv.yaml
)

for cfg in "${configs[@]}"; do
    log="ufbrp-${cfg//\//_}"
    ckpt_dir="logs/${log//.yaml/}"
    log="logs/${log//.yaml/}"
    mkdir -p ${log}
    log="${log}/log.out"

    LOCAL=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
    mkdir -p "$LOCAL/ffcv"
    SRC_TRAIN="/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline/data/imagenet/ffcv_raw/train.ffcv"
    SRC_VAL="/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline/data/imagenet/ffcv_raw/val.ffcv"

    echo "Coping dataset to the ${LOCAL}"
    rsync -aP "$SRC_TRAIN" "$LOCAL/ffcv/train.ffcv.tmp" && mv "$LOCAL/ffcv/train.ffcv.tmp" "$LOCAL/ffcv/train.ffcv"
    rsync -aP "$SRC_VAL"   "$LOCAL/ffcv/val.ffcv.tmp"   && mv "$LOCAL/ffcv/val.ffcv.tmp"   "$LOCAL/ffcv/val.ffcv"


    echo "=== Run ${cfg} ==="
    echo "$SLURM_JOB_NODELIST"
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-node=8 \
            --container-image "${IMAGE}" \
            --container-mounts "${WORKDIR}:/workspace/ufbrp",${LOCAL}:${LOCAL} \
            bash -c "
                set -e
                cd /workspace/ufbrp
                ls -l ${LOCAL}/ffcv
                apt -qq update > /dev/null
                apt install -y -qq libopencv-dev pkg-config libturbojpeg-dev ca-certificates > /dev/null
                update-ca-certificates > /dev/null

                echo 'Install dependencies'
                python -m pip install --upgrade pip
                python -m pip install -q -r requirements_ffcv.txt
                python -m pip install -e .
                mkdir -p ${ckpt_dir}

                echo 'Free dev storage'
                free -h
                df -h /dev/shm

                echo 'Start training'
                export NUMBA_THREADING_LAYER=omp
                export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
                python ufbrp/train_ffcv.py --config-file ufbrp/presets/ffcv/${TRAIN_CONFIG} \
                --data.config=${cfg} \
                --data.train_dataset=${LOCAL}/ffcv/train.ffcv \
                --data.val_dataset=${LOCAL}/ffcv/val.ffcv \
                --data.num_workers=6 --data.in_memory=1 \
                --dist.world_size=8 \
                --logging.folder=${ckpt_dir}
            " > "${log}" 2>&1 &
    sleep 20
done

wait
echo $?