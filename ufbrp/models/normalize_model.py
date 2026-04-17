from collections import OrderedDict

import torch
import torch.nn as nn


class ImageNormalizer(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        persistent: bool = True,
    ):
        super(ImageNormalizer, self).__init__()

        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1), persistent=persistent)
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1), persistent=persistent)

    def forward(self, inputs: torch.Tensor):
        return (inputs - self.mean) / self.std


def create_model(
    model: nn.Module,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)
