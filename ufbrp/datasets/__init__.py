from .cifar10 import (
    CIFAR10Dataset,
)
from .cifar10 import (
    test_transforms as cifar10_test_transforms,
)
from .cifar10 import (
    train_transforms as cifar10_train_transforms,
)
from .cifar100 import (
    CIFAR100Dataset,
)
from .cifar100 import (
    test_transforms as cifar100_test_transforms,
)
from .cifar100 import (
    train_transforms as cifar100_train_transforms,
)
from .imagenet import (
    ImageNetDataset,
)
from .imagenet import (
    test_transforms as imagenet_test_transforms,
)
from .imagenet import (
    train_transforms as imagenet_train_transforms,
)

datasets = {
    "CIFAR10": (CIFAR10Dataset, cifar10_train_transforms, cifar10_test_transforms),
    "CIFAR100": (CIFAR100Dataset, cifar100_train_transforms, cifar100_test_transforms),
    "ImageNet": (ImageNetDataset, imagenet_train_transforms, imagenet_test_transforms),
}


def get_data_loader(dataset_name: str, directory: str, num_workers: int, batch_size: int, val_split_ratio: float = 0.9):
    dataset_class, train_transforms, test_transforms = datasets[dataset_name]
    dataset = dataset_class(directory=directory, val_split_ratio=val_split_ratio)
    return (
        dataset.get_train_loader(batch_size, num_workers, shuffle=True, train_transforms=train_transforms),
        dataset.get_val_loader(batch_size, num_workers, shuffle=False, val_transforms=test_transforms),
        dataset.get_test_loader(batch_size, num_workers, shuffle=False, test_transforms=test_transforms),
    )
