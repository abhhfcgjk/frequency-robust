#!/usr/bin/env python3
from pathlib import Path
import argparse

import scipy.io


def build_train_class_to_idx(train_dir: Path):
    if not train_dir.is_dir():
        raise ValueError(f"Train directory does not exist: {train_dir}")

    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in: {train_dir}")

    return {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}


def load_ilsvrcid_to_wnid(meta_mat_path: Path):
    """
    Reads meta.mat from ImageNet devkit and returns:
        {ILSVRC2012_ID (1-based int): WNID string}
    Only leaf synsets (the 1000 classification classes) are kept.
    """
    if not meta_mat_path.is_file():
        raise ValueError(f"meta.mat not found: {meta_mat_path}")

    meta = scipy.io.loadmat(meta_mat_path, simplify_cells=True)
    synsets = meta["synsets"]

    ilsvrcid_to_wnid = {}

    for s in synsets:
        # Typical fields:
        # s["ILSVRC2012_ID"], s["WNID"], s["num_children"]
        num_children = int(s["num_children"])
        if num_children != 0:
            continue  # keep only leaf classes

        ilsvrc_id = int(s["ILSVRC2012_ID"])
        wnid = str(s["WNID"])
        ilsvrcid_to_wnid[ilsvrc_id] = wnid

    if len(ilsvrcid_to_wnid) != 1000:
        raise ValueError(
            f"Expected 1000 leaf classes in meta.mat, got {len(ilsvrcid_to_wnid)}"
        )

    return ilsvrcid_to_wnid


def read_ground_truth(gt_path: Path):
    """
    Reads ILSVRC2012_validation_ground_truth.txt
    Each line contains a 1-based ILSVRC2012 class ID.
    """
    if not gt_path.is_file():
        raise ValueError(f"Ground-truth file not found: {gt_path}")

    with gt_path.open("r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    if not labels:
        raise ValueError(f"No labels found in: {gt_path}")

    return labels


def make_val_map(
    val_dir: Path,
    train_dir: Path,
    ground_truth_file: Path,
    meta_mat_file: Path,
    output_file: Path,
):
    if not val_dir.is_dir():
        raise ValueError(f"Validation directory does not exist: {val_dir}")

    train_class_to_idx = build_train_class_to_idx(train_dir)
    ilsvrcid_to_wnid = load_ilsvrcid_to_wnid(meta_mat_file)
    gt_labels = read_ground_truth(ground_truth_file)

    val_images = sorted([p for p in val_dir.iterdir() if p.is_file()])
    if len(val_images) != len(gt_labels):
        raise ValueError(
            f"Number of validation images ({len(val_images)}) != "
            f"number of ground-truth labels ({len(gt_labels)})"
        )

    with output_file.open("w", encoding="utf-8") as f:
        for img_path, ilsvrc_id in zip(val_images, gt_labels):
            wnid = ilsvrcid_to_wnid[ilsvrc_id]
            class_idx = train_class_to_idx[wnid]
            print(f"{img_path.name}\t{class_idx}\n")
            f.write(f"{img_path.name}\t{class_idx}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create val_map.txt for standard flat ImageNet validation directory."
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default="/dataset/validation",
        help="Path to ImageNet validation folder, e.g. /dataset/validation",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default="/dataset/train",
        help="Path to ImageNet train folder, e.g. /dataset/train",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default="/dataset/ILSVRC2012_validation_ground_truth.txt",
        help="Path to ILSVRC2012_validation_ground_truth.txt",
    )
    parser.add_argument(
        "--meta-mat",
        type=Path,
        default="/dataset/meta.mat",
        help="Path to meta.mat from ImageNet devkit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/dataset/val_map.txt"),
        help="Output txt file path",
    )
    args = parser.parse_args()

    make_val_map(
        val_dir=args.val_dir,
        train_dir=args.train_dir,
        ground_truth_file=args.ground_truth,
        meta_mat_file=args.meta_mat,
        output_file=args.output,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()