from .cifar10 import get_data_loader as get_data_loader_cifar10
from .cifar100 import get_data_loader as get_data_loader_cifar100
from .imagenet import get_data_loader as get_data_loader_imagenet

dataloader = {
    "CIFAR10": get_data_loader_cifar10,
    "CIFAR100": get_data_loader_cifar100,
    "ImageNet": get_data_loader_imagenet
}


def get_data_loader(dataset_name: str, directory: str, num_workers: int, batch_size: int, world_size: int, rank: int):
    loader = dataloader[dataset_name]
    return loader(directory, num_workers, batch_size, world_size, rank)
