from __future__ import annotations

import math
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fastargs import Section, Param, get_current_config
from fastargs.decorators import param, section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField


Section("cfg", "arguments to give the writer").params(
    dataset=Param(And(str, OneOf(["cifar", "imagenet"])), "Which dataset to write", default="imagenet"),
    split=Param(And(str, OneOf(["train", "val"])), "Train or val set", required=True),
    data_dir=Param(str, "Where to find the PyTorch dataset", required=True),
    write_path=Param(str, "Where to write the new dataset", required=True),

    # FFCV write modes: raw, jpg, smart, proportion
    write_mode=Param(And(str, OneOf(["raw", "jpg", "smart", "proportion"])), "Image write mode", default="raw"),

    max_resolution=Param(int, "Max image side length", required=True),
    num_workers=Param(int, "Number of workers to use", default=16),
    chunk_size=Param(int, "Chunk size for writing", default=100),
    jpeg_quality=Param(int, "Quality of jpeg images", default=90),

    # Use -1 for "unset"
    compress_probability=Param(float, "For proportion mode: probability of JPEG", default=-1.0),
    smart_threshold=Param(int, "For smart mode: JPEG if raw_bytes > threshold (-1 unset)", default=-1),

    subset=Param(int, "How many images to use (-1 for all)", default=-1),

    # Sharding: write separate beton files in parallel (safe)
    num_shards=Param(int, "Total number of shards to write", default=1),
    shard_id=Param(int, "Shard index in [0, num_shards)", default=0),
)


class FlatFolderDataset(Dataset):
    """
    ImageNet val: flat folder with filenames like ILSVRC2012_val_00000001.JPEG
    targets list is in synset-id order; we map filename -> 0-based index -> target.
    """
    def __init__(self, root_dir: str, targets: List[int], transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.targets = targets
        self.transform = transform

        exts = (".JPEG", ".jpg", ".jpeg", ".png")
        self.image_files = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(exts)
        )

        if len(self.targets) != len(self.image_files):
            raise ValueError(
                f"Val mismatch: {len(self.targets)} labels vs {len(self.image_files)} images. "
                f"Check devkit ground truth + val folder."
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)

        # ILSVRC2012_val_00000001.JPEG -> 1 -> 0-based
        try:
            img_id0 = int(filename.rsplit("_", 1)[-1].split(".")[0]) - 1
        except Exception:
            # fallback: assume sorted order corresponds to gt order
            img_id0 = idx

        label = self.targets[img_id0]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ImageNetDataset:
    def __init__(self, directory: str):
        self.data_dir = directory
        self.train_set_path = os.path.join(self.data_dir, "train")
        self.val_set_path = os.path.join(self.data_dir, "val")
        self.synset_id_to_train_idx: Dict[int, int] = {}

    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None) -> Dataset:
        train_ds = ImageFolder(self.train_set_path, transform=train_transforms)

        wnid_to_train_idx: Dict[str, int] = {k.replace(".tar", ""): v for k, v in train_ds.class_to_idx.items()}
        synset_id_to_wnid = self.parse_meta_mat_synset_to_wnid(self.data_dir)

        missing = [wnid for wnid in synset_id_to_wnid.values() if wnid not in wnid_to_train_idx]
        if missing:
            raise RuntimeError(
                f"Train folder structure mismatch. Missing wnids in train/: {missing[:10]} "
                f"(total missing {len(missing)})."
            )

        self.synset_id_to_train_idx = {sid: wnid_to_train_idx[wnid] for sid, wnid in synset_id_to_wnid.items()}
        return train_ds

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        synset_ids = self.parse_val_gt_txt(self.data_dir)
        if not self.synset_id_to_train_idx:
            _ = self.get_train_dataset()
        targets = [self.synset_id_to_train_idx[sid] for sid in synset_ids]
        return FlatFolderDataset(self.val_set_path, targets=targets, transform=val_transforms)

    @staticmethod
    def parse_val_gt_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as f:
            return [int(line.strip()) for line in f]

    @staticmethod
    def parse_meta_mat_synset_to_wnid(devkit_root: str) -> Dict[int, str]:
        metafile = os.path.join(devkit_root, "meta.mat")
        synsets = sio.loadmat(metafile, squeeze_me=True)["synsets"]

        # synsets is a structured array; "num_children" is typically field index 4
        nums_children = [int(s[4]) for s in synsets]
        leaf = [synsets[i] for i, nchild in enumerate(nums_children) if nchild == 0]

        # fields: (ILSVRC2012_ID, wnid, words, gloss, num_children, children, ...)
        idcs = [int(s[0]) for s in leaf]
        wnids = [str(s[1]) for s in leaf]
        return {sid: wnid for sid, wnid in zip(idcs, wnids)}


def _lock_path(path: str) -> str:
    return path + ".lock"


def _acquire_lock(path: str) -> int:
    lock = _lock_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(lock, flags)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.fsync(fd)
    return fd


def _release_lock(path: str, fd: int) -> None:
    try:
        os.close(fd)
    finally:
        try:
            os.remove(_lock_path(path))
        except FileNotFoundError:
            pass


def _apply_subset_and_shard(ds: Dataset, subset: int, shard_id: int, num_shards: int) -> Dataset:
    if subset > 0:
        ds = Subset(ds, range(min(subset, len(ds))))

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    if num_shards == 1:
        return ds

    n = len(ds)
    per = math.ceil(n / num_shards)
    start = shard_id * per
    end = min(n, (shard_id + 1) * per)
    if start >= end:
        raise ValueError(f"Shard is empty: shard_id={shard_id}, num_shards={num_shards}, n={n}")

    return Subset(ds, range(start, end))


@section("cfg")
@param("dataset")
@param("split")
@param("data_dir")
@param("write_path")
@param("max_resolution")
@param("num_workers")
@param("chunk_size")
@param("subset")
@param("jpeg_quality")
@param("write_mode")
@param("compress_probability")
@param("smart_threshold")
@param("num_shards")
@param("shard_id")
def main(
    dataset, split, data_dir, write_path, max_resolution, num_workers,
    chunk_size, subset, jpeg_quality, write_mode, compress_probability,
    smart_threshold, num_shards, shard_id
):
    if dataset != "imagenet":
        raise ValueError(f"Unrecognized dataset: {dataset}")

    ds = ImageNetDataset(data_dir)
    if split == "train":
        base_ds = ds.get_train_dataset()
    elif split == "val":
        base_ds = ds.get_val_dataset()
    else:
        raise ValueError(f"Unrecognized split: {split}")

    base_ds = _apply_subset_and_shard(base_ds, subset=subset, shard_id=shard_id, num_shards=num_shards)

    # If sharding, enforce unique output per shard
    final_path = write_path
    if num_shards > 1:
        root, ext = os.path.splitext(write_path)
        if ext == "":
            ext = ".beton"
        final_path = f"{root}.shard{shard_id:03d}-of-{num_shards:03d}{ext}"

    # Validate mode args
    if write_mode == "proportion":
        if compress_probability < 0.0 or compress_probability > 1.0:
            raise ValueError("For write_mode='proportion', set compress_probability in [0, 1].")
    else:
        compress_probability = None  # ignored

    if write_mode == "smart":
        st = None if smart_threshold < 0 else int(smart_threshold)
    else:
        st = None  # ignored

    fields = {
        "image": RGBImageField(
            write_mode=write_mode,
            max_resolution=max_resolution,
            compress_probability=compress_probability,
            jpeg_quality=int(jpeg_quality),
            smart_threshold=st,
        ),
        "label": IntField(),
    }

    # Lock + atomic write prevents corrupted/partial beton (SIGBUS later)
    lock_fd = _acquire_lock(final_path)
    tmp_path = final_path + f".tmp.{os.getpid()}"
    try:
        writer = DatasetWriter(tmp_path, fields, num_workers=int(num_workers))
        writer.from_indexed_dataset(base_ds, chunksize=int(chunk_size))
        os.replace(tmp_path, final_path)  # atomic rename
        print(f"Wrote: {final_path}  (n={len(base_ds)})")
    finally:
        # cleanup tmp if failed
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        _release_lock(final_path, lock_fd)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    main()
