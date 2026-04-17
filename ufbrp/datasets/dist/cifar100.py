from typing import Optional
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from datasets.base import BaseDataLoader

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class CIFAR100Data(Dataset):
    def __init__(self, root: str, phase: str = "train", transform=None):
        self.phase = phase
        self.transform = transform
        is_train = (phase == "train")
        self.dataset = datasets.CIFAR100(root=root, train=is_train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class CIFAR100Dataset(BaseDataLoader):
    def __init__(self, directory: str, val_split_ratio: float):
        super().__init__(directory)
        self.val_split_ratio = val_split_ratio

    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return CIFAR100Data(root=self.data_dir, transform=train_transforms, phase="train")

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return CIFAR100Data(root=self.data_dir, transform=val_transforms, phase="val")

    def get_test_dataset(self, test_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return CIFAR100Data(root=self.data_dir, transform=test_transforms, phase="test")


def get_data_loader(directory: str, num_workers: int, batch_size: int, world_size: int=0, rank: Optional[int]=None):
    """
    Returns distributed data loaders for CIFAR-10:
    train_loader, val_loader, test_loader
    """
    dataset_wrapper = CIFAR100Dataset(directory, val_split_ratio=0.1)

    train_dataset = dataset_wrapper.get_train_dataset(train_transforms)
    val_dataset = dataset_wrapper.get_val_dataset(test_transforms)
    test_dataset = dataset_wrapper.get_test_dataset(test_transforms)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=False, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, sampler=val_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, sampler=test_sampler, pin_memory=True)

    return train_loader, val_loader, test_loader