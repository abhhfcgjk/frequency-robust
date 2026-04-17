import math

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from torchvision import transforms
from torchvision.datasets import CIFAR10


class ExternalInputIterator(object):
    def __init__(self, batch_size, root, phase):
        self.batch_size = batch_size
        self.phase = phase

        train = phase == "train"
        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=False,
            transform=transforms.ToTensor(),  # we won't use it directly, DALI will handle transforms
        )

        self.n = len(self.dataset)
        self.indices = np.arange(self.n)
        if train:
            np.random.shuffle(self.indices)
            global TRAIN_LENGTH
            TRAIN_LENGTH = len(self)

        print(f"[CIFAR10] Loaded {self.n} images ({phase})")

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batch, labels = [], []
        for _ in range(self.batch_size):
            if self.i == self.n:
                break
            img, label = self.dataset[self.indices[self.i]]
            # img = (img.numpy().transpose(1, 2, 0)).astype(np.uint8)
            batch.append(img.numpy().astype(np.uint8))
            labels.append(np.array(label, dtype=np.long))
            self.i += 1
        return (batch, labels)


def ExternalSourcePipeline(
    batch_size, num_threads, external_data, device_id=0, seed=20, phase="train", crop_size=32, resize_size=40
):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    with pipe:
        images, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.transpose(images, perm=[1, 2, 0])
        # images = fn.decoders.image(images, device="cpu", output_type=types.RGB)

        if phase == "train":
            # RandomCrop(32, padding=4)
            images = fn.pad(images, fill_value=0, axes=(0, 1), shape=(resize_size, resize_size))
            images = fn.crop(
                images,
                crop=(crop_size, crop_size),
                crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
                crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            )
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            mirror = 0

        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0, 0, 0],
            std=[255, 255, 255],
            mirror=mirror,
        )
        labels = labels.gpu()
        pipe.set_outputs(images, labels)
    return pipe


def get_data_loader(directory: str, num_workers: int, batch_size: int, phase="train", seed=20):
    if phase == "train":
        eii_train = ExternalInputIterator(batch_size, root=directory, phase="train")
        train_pipe = ExternalSourcePipeline(
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=0,
            external_data=eii_train,
            seed=seed,
            phase="train",
        )
        train_pipe.build()
        train_loader = DALIClassificationIterator(
            train_pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL
        )

        eii_val = ExternalInputIterator(batch_size, root=directory, phase="val")
        val_pipe = ExternalSourcePipeline(
            batch_size=batch_size, num_threads=num_workers, device_id=0, external_data=eii_val, seed=seed, phase="val"
        )
        val_pipe.build()
        val_loader = DALIClassificationIterator(
            val_pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL
        )

        return train_loader, val_loader, None

    else:
        eii_test = ExternalInputIterator(batch_size, root=directory, phase="test")
        test_pipe = ExternalSourcePipeline(
            batch_size=batch_size, num_threads=num_workers, device_id=0, external_data=eii_test, seed=seed, phase="test"
        )
        test_pipe.build()
        test_loader = DALIClassificationIterator(
            test_pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL
        )
        return None, None, test_loader


def get_train_loader_length():
    global TRAIN_LENGTH
    return TRAIN_LENGTH


if __name__ == "__main__":
    loader = get_data_loader("data/CIFAR10", num_workers=2, batch_size=16, phase="train")
    counter = 0
    for data in loader[0]:
        counter += 1
    print("Total batches:", counter)
