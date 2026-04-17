from collections.abc import Callable

import numpy as np
import torch.nn.functional as F
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


class ImageNetPGD(Attacker):
    def __init__(self, model, loss_computer, eps, alpha=None, iters=5, mode="zero",
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), *args, **kwargs):
        super().__init__(model)
        self.eps = eps / 255.0
        self.alpha = (alpha / 255.0) if alpha is not None else self.eps
        self.iters = iters
        self.loss_computer = loss_computer
        self.mode = mode

        self._mean255 = torch.tensor(mean).view(1, 3, 1, 1) * 255.0
        self._std255  = torch.tensor(std).view(1, 3, 1, 1) * 255.0

    def run(self, inputs_norm: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = inputs_norm.device
        mean255 = self._mean255.to(device=device, dtype=torch.float32)
        std255  = self._std255.to(device=device, dtype=torch.float32)

        x_norm = inputs_norm.detach().float()

        x255 = x_norm * std255 + mean255
        x01 = (x255 / 255.0).clamp(0.0, 1.0)
        noise = torch.empty_like(x01).uniform_(-self.eps, self.eps)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.enable_grad():
                for _ in range(self.iters):
                    noise = noise.detach().requires_grad_(True)

                    noisy01 = (x01 + noise).clamp(0.0, 1.0)
                    noisy255 = noisy01 * 255.0
                    noisy_norm = (noisy255 - mean255) / std255

                    self.model.zero_grad(set_to_none=True)

                    logits = self.model(noisy_norm)
                    loss = self.loss_computer(logits, target)

                    # checkpointing-safe
                    loss.backward()

                    grad = noise.grad.detach()
                    noise = noise + self.alpha * grad.sign()
                    noise = noise.clamp(-self.eps, self.eps)

        self.model.zero_grad(set_to_none=True)

        adv01 = (x01 + noise).clamp(0.0, 1.0)
        adv255 = adv01 * 255.0
        adv_norm = (adv255 - mean255) / std255
        return adv_norm.to(dtype=inputs_norm.dtype)
