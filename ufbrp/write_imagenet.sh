#!/bin/bash

write_dataset () {
    split=$1
    maxres=$2
    prob=$3
    quality=$4

    final="$WRITE_DIR/${split}.ffcv"
    tmp="${final}.tmp.${SLURM_JOB_ID:-$$}"

    echo "Writing ImageNet ${split} -> ${final}"

    python ufbrp/write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split="${split}" \
        --cfg.data_dir="$IMAGENET_DIR" \
        --cfg.write_path="$tmp" \
        --cfg.max_resolution="${maxres}" \
        --cfg.write_mode=smart \
        --cfg.compress_probability="${prob}" \
        --cfg.jpeg_quality="${quality}" \
        --cfg.num_workers=12

    mv -f "$tmp" "$final"
    sync
    chmod 444 "$final"

    ls -lh "$final"
}

write_dataset train $1 $2 $3
write_dataset  val  $1 $2 $3