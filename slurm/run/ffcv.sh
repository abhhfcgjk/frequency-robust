#!/bin/bash
set -euo pipefail

SBATCH_SCRIPT="slurm/sbatch/$1"

if [ -f "$SBATCH_SCRIPT" ]; then
    JOBID=$(sbatch --parsable "$SBATCH_SCRIPT")
    echo "Submitted job $JOBID"

    while squeue -j "$JOBID" >/dev/null 2>&1; do
        sleep 30
    done

    STATE=$(sacct -j "$JOBID" --format=State --noheader | head -n 1 | awk '{print $1}')
    echo "Job $JOBID finished with state: $STATE"
else
    echo "does not exists ${SBATCH_SCRIPT}"
fi
