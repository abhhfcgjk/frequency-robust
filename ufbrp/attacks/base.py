from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Attacker(ABC):
    """A base class for untargeted adversarial attacks on images.

    :param model: The model to be attacked.
    :param device: The device on which the attack will be performed.
    """

    def __init__(self, model: nn.Module, device: torch.device = "cuda") -> None:
        self.model = model
        self.device = device
        # device = next(model.buffers()).device
        # if self.device != device:
        #     self.model.to(self.device)

    @abstractmethod
    def run(self, inputs: Tensor, target: Tensor) -> Tensor:
        """Perform the untargeted attack.

        :param inputs: Benign image batch to attack. Must have shape
            (batch size, 3, height, width) and values in range [0..1].
        :param target: Ground truth labels in one-hot format. The attacker's goal is to push the predictions *away* from ground truth.
        :returns: The attacked image batch.

        Both inputs and target must be on device (.cuda()).
        """
        pass
