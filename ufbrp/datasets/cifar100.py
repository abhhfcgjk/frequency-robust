from typing import Optional

from datasets.base import BaseDataLoader
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose([transforms.ToTensor()])


class CIFAR100Dataset(BaseDataLoader):
    def __init__(self, directory: str, val_split_ratio: float) -> Dataset:
        super().__init__(directory)
        self.val_split_ratio = val_split_ratio

    def __split_train_val_data(self, trainsform):
        full_train_dataset = CIFAR100(root=self.data_dir, transform=trainsform, train=True, download=True)
        self.train_size = int(self.val_split_ratio * len(full_train_dataset))
        self.val_size = len(full_train_dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [self.train_size, self.val_size])

    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None) -> Dataset:
        if hasattr(self, "train_dataset"):
            return self.train_dataset
        self.__split_train_val_data(train_transforms)
        return self.train_dataset

    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None) -> Dataset:
        if hasattr(self, "val_dataset"):
            return self.val_dataset
        self.__split_train_val_data(val_transforms)
        return self.val_dataset

    def get_test_dataset(self, test_transforms: Optional[transforms.Compose] = None) -> Dataset:
        return CIFAR100(root=self.data_dir, transform=test_transforms, train=False, download=True)


def get_data_loader(directory: str, num_workers: int, batch_size: int, val_split_ratio: float = 0.8):
    cifar10 = CIFAR100Dataset(directory=directory, val_split_ratio=val_split_ratio)
    return (
        cifar10.get_train_loader(batch_size, num_workers, shuffle=True, train_transforms=train_transforms),
        cifar10.get_val_loader(batch_size, num_workers, shuffle=False, val_transforms=test_transforms),
        cifar10.get_test_loader(batch_size, num_workers, shuffle=False, test_transforms=test_transforms),
    )
