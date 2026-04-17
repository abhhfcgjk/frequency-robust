from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BaseDataLoader(ABC):
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir

    @abstractmethod
    def get_train_dataset(self, train_transforms: Optional[transforms.Compose] = None, **kwargs) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_val_dataset(self, val_transforms: Optional[transforms.Compose] = None, **kwargs) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_test_dataset(self, test_transforms: Optional[transforms.Compose] = None, **kwargs) -> Dataset:
        raise NotImplementedError

    def get_train_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        train_transforms: Optional[transforms.Compose] = None,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_train_dataset(train_transforms=train_transforms),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

    def get_val_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
        val_transforms: Optional[transforms.Compose] = None,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_val_dataset(val_transforms=val_transforms),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

    def get_test_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
        test_transforms: Optional[transforms.Compose] = None,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_test_dataset(test_transforms=test_transforms),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )
