#!/bin/bash
set -euo pipefail

SBATCH_SCRIPT="slurm/sbatch/train-lipreg_aa.sbatch"
WORKDIR=/scratch/amoskalenko/users/28i_mel/unified-frequency-based-robustness-pipeline/ufbrp

while true; do
    if [ -d "$WORKDIR/logs" ] && [ "$(ls -A "$WORKDIR/logs")" ]; then
        TS=$(date +"%Y-%m-%d_%H-%M-%S")
        BACKUP="$WORKDIR/logs_lipreg_aa$TS"
        mkdir -p "$BACKUP"
        cp -a "$WORKDIR/logs/." "$BACKUP/"
    fi
    # Submit job and capture JobID
    JOBID=$(sbatch --parsable "$SBATCH_SCRIPT")
    echo "Submitted job $JOBID"

    # Wait until job finishes
    while squeue -j "$JOBID" >/dev/null 2>&1; do
        sleep 30
    done

    # Get final state
    STATE=$(sacct -j "$JOBID" --format=State --noheader | head -n 1 | awk '{print $1}')
    echo "Job $JOBID finished with state: $STATE"

    # If job died because of time limit → resubmit
    if [[ "$STATE" == "TIMEOUT" ]]; then
        echo "Time limit reached. Resubmitting..."
        sleep 5
    else
        echo "Job finished successfully or failed. Stopping."
        break
    fi
done
