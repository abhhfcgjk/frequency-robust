import os
from typing import Dict, List, Optional, Tuple

import scipy.io as sio
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

train_transforms = transforms.Compose(
    [
        transforms.Resize((226, 226)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((226, 226)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ]
)


class FlatFolderDataset(Dataset):
    def __init__(self, root_dir: str, targets: List[int], transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.targets = targets
        self.transform = transform
        self.image_files = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".JPEG")
        )
        if len(self.targets) != 50000:
            raise ValueError(f"Expected 50000 val labels, got {len(self.targets)}")
        if len(self.image_files) != 50000:
            raise ValueError(f"Expected 50000 val images, got {len(self.image_files)}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        img_id0 = int(filename.rsplit("_", 1)[-1].replace(".JPEG", "")) - 1
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
                f"(total missing {len(missing)}). Example train classes: {list(wnid_to_train_idx)[:10]}"
            )

        self.synset_id_to_train_idx = {sid: wnid_to_train_idx[wnid] for sid, wnid in synset_id_to_wnid.items()}
        return train_ds

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        synset_ids = self.parse_val_gt_txt(self.data_dir)
        if not self.synset_id_to_train_idx:
            raise RuntimeError("Call get_train_dataset() before get_val_dataset()")
        targets = [self.synset_id_to_train_idx[sid] for sid in synset_ids]
        return FlatFolderDataset(self.val_set_path, targets=targets, transform=val_transforms)

    def get_test_dataset(self, test_transforms: Optional[transforms.Compose] = None) -> Dataset:
        synset_ids = self.parse_val_gt_txt(self.data_dir)
        if not self.synset_id_to_train_idx:
            raise RuntimeError("Call get_train_dataset() before get_test_dataset()")
        targets = [self.synset_id_to_train_idx[sid] for sid in synset_ids]
        return FlatFolderDataset(self.val_set_path, targets=targets, transform=test_transforms)

    @staticmethod
    def parse_val_gt_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as f:
            return [int(line.strip()) for line in f]

    @staticmethod
    def parse_meta_mat_synset_to_wnid(devkit_root: str) -> Dict[int, str]:
        metafile = os.path.join(devkit_root, "meta.mat")
        synsets = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*synsets))[4]
        synsets = [synsets[i] for i, nchild in enumerate(nums_children) if nchild == 0]
        idcs, wnids = list(zip(*synsets))[:2]
        return {int(sid): str(wnid) for sid, wnid in zip(idcs, wnids)}


def get_data_loader(
    directory: str,
    num_workers: int,
    batch_size: int,
    world_size: int = 0,
    rank: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    imagenet = ImageNetDataset(directory=directory)
    train_dataset = imagenet.get_train_dataset(train_transforms)
    val_dataset = imagenet.get_val_dataset(test_transforms)
    test_dataset = imagenet.get_test_dataset(test_transforms)

    use_ddp = world_size is not None and world_size > 1
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
    )

    print("-" * 50)
    print(f"Train len: {len(train_loader)}\nVal len: {len(val_loader)}")
    print("-" * 50)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    loader_train, loader_val, loader_test = get_data_loader(
        "/home/28i_mel@lab.graphicon.ru/unified-frequency-based-robustness-pipeline/data/imagenet",
        num_workers=4,
        batch_size=4,
        world_size=0,
        rank=None,
    )
    for im, label in loader_test:
        im = im.to("cuda")
        label = label.to("cuda")
        print(im.shape, label.shape)
