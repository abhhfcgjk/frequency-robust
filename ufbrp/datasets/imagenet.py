import os
from typing import Optional

import scipy.io as sio
from datasets.base import BaseDataLoader
from PIL import Image
from torch.utils.data import Dataset
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
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        class_name = filename.rsplit("_", 1)[-1].replace(".JPEG", "")
        class_name = int(class_name)
        label = self.class_to_idx[class_name]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetDataset(BaseDataLoader):
    def __init__(self, directory: str, *args, **kwargs) -> Dataset:
        super().__init__(directory)
        self.train_set_path = os.path.join(self.data_dir, "train")
        self.val_set_path = os.path.join(self.data_dir, "val")

    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None) -> Dataset:
        self.calss_to_idx = self.parse_meta_mat(self.data_dir)
        folder = ImageFolder(self.train_set_path, transform=train_transforms)
        folder.class_to_idx = self.calss_to_idx
        return folder

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        self.calss_to_idx = self.parse_val_gt_txt(self.data_dir)
        return FlatFolderDataset(self.val_set_path, transform=val_transforms, class_to_idx=self.calss_to_idx)

    def get_test_dataset(self, test_transforms: Optional[transforms.Compose] = None) -> Dataset:
        self.calss_to_idx = self.parse_val_gt_txt(self.data_dir)
        return FlatFolderDataset(self.val_set_path, transform=test_transforms, class_to_idx=self.calss_to_idx)

    @staticmethod
    def parse_val_gt_txt(devkit_root: str) -> list[int]:
        file = os.path.join(devkit_root, "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return {i: int(val_idx) for i, val_idx in enumerate(val_idcs)}

    @staticmethod
    def parse_meta_mat(devkit_root: str) -> tuple[dict[int, str], dict[str, tuple[str, ...]]]:
        metafile = os.path.join(devkit_root, "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, wnids)}
        # wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return wnid_to_idx


def get_data_loader(directory: str, num_workers: int, batch_size: int, val_split_ratio: float = 0.8):
    imagenet = ImageNetDataset(directory=directory)
    return (
        imagenet.get_train_loader(batch_size, num_workers, shuffle=True, train_transforms=train_transforms),
        imagenet.get_val_loader(batch_size, num_workers, shuffle=False, val_transforms=test_transforms),
        imagenet.get_test_loader(batch_size, num_workers, shuffle=False, test_transforms=test_transforms),
    )


if __name__ == "__main__":
    loader_train, loader_val, loader_test = get_data_loader(
        "/home/28i_mel@lab.graphicon.ru/unified-frequency-based-robustness-pipeline/data/imagenet",
        num_workers=4,
        batch_size=4,
    )
    for im, label in loader_test:
        im = im.to("cuda")
        label = label.to("cuda")
        print(im.shape, label.shape)
