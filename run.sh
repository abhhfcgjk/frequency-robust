#!/bin/bash
set -euo pipefail

SBATCH_SCRIPT="run.sbatch"

if [ -f "$SBATCH_SCRIPT" ]; then
    while true
    do
        JOBID=$(sbatch --parsable "$SBATCH_SCRIPT")
        echo "Submitted job $JOBID"

        while squeue -j "$JOBID" >/dev/null 2>&1; do
            sleep 30
        done

        STATE=$(sacct -j "$JOBID" --format=State --noheader | head -n 1 | awk '{print $1}')

        echo "Job $JOBID finished with state: $STATE"
        if [[ "$STATE" != "TIMEOUT" ]]; then
            break
        fi
    done
else
    echo "does not exists ${SBATCH_SCRIPT}"
fi
