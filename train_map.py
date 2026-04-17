#!/usr/bin/env python3
from pathlib import Path
import argparse


def make_train_map(train_dir: Path, output_file: Path) -> None:
    if not train_dir.is_dir():
        raise ValueError(f"Train directory does not exist: {train_dir}")

    # Standard ImageNet uses synset folder names like n01440764.
    # Sort folders to get deterministic class indices.
    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}

    with output_file.open("w", encoding="utf-8") as f:
        for cls_dir in class_dirs:
            class_name = cls_dir.name
            class_idx = class_to_idx[class_name]

            # Write relative paths like: n01440764/n01440764_10026.JPEG\t0
            for img_path in sorted([p for p in cls_dir.iterdir() if p.is_file()]):
                rel_path = img_path.relative_to(train_dir).as_posix()
                print(f"{rel_path}\t{class_idx}\n")
                f.write(f"{rel_path}\t{class_idx}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create train_map.txt from a standard ImageNet train folder."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/dataset/train"),
        help="Path to ImageNet train folder, e.g. /path/to/imagenet/train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/dataset/train_map.txt"),
        help="Output txt file path",
    )
    args = parser.parse_args()

    make_train_map(args.train_dir, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()