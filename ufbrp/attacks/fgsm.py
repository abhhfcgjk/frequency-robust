from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .base import Attacker


class FGSM(Attacker):
    """
    Fast Gradient Sign Attack.

    :param model: The model to be attacked. Must be on device (.cuda()).
    :param scaler: The loss scaler.
    :param eps: Upper bound for linf norm of perturbations.
    :param alpha: Step size which is multiplied by the gradient sign. It is
        reasonable to choose values slightly greater than eps as shown by fast
        training paper.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_computer: Callable[..., Tensor],
        eps: float,
        alpha: float = None,
        mode: str = "zero",
    ) -> None:
        super().__init__(model)
        self.eps = eps / 255
        self.alpha = alpha / 255 if alpha is not None else self.eps

        self.loss_computer = loss_computer
        self.mode = mode
        self.noise: Tensor | None = None

    def reset(self, input_shape: tuple[int, ...], device: torch.device = "cuda") -> None:
        """Reset noise batch to zeros or random uniform (-eps, eps) values.

        :param input_shape: Input batch shape, needed only once at the start of training process.
        :param mode: Resetting mode (choices: 'uniform', 'zero').
        """
        if self.noise is None:
            self.noise = torch.empty(*input_shape, device=device)

        if self.mode == "uniform":
            self.noise.uniform_(-self.eps, self.eps)
        elif self.mode == "zero":
            self.noise.zero_()
        else:
            raise ValueError(f"Unknown mode `{self.mode}`")

    def run(self, inputs: Tensor, target: Tensor, return_grad: bool = False):
        self.reset(input_shape=inputs.shape, device=inputs.device)

        noisy_inputs = inputs.clone() + self.noise[: inputs.shape[0]]
        noisy_inputs.requires_grad_()

        noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)
        noisy_outputs = self.model(noisy_inputs)
        loss = self.loss_computer(noisy_outputs, target)

        grad = torch.autograd.grad(loss, [noisy_inputs])[0].detach()
        if return_grad:
            return noisy_inputs, torch.sign(grad)
        adv = noisy_inputs + self.alpha * torch.sign(grad)
        attacked_inputs = torch.clamp(
            torch.min(torch.max(adv, noisy_inputs - self.eps), noisy_inputs + self.eps),
            0.0,
            1.0,
        )

        return attacked_inputs.detach()
