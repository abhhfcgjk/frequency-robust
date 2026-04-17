"""
Loss for TRADES training.

| Paper: https://arxiv.org/abs/1901.08573
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TRADESLoss(nn.Module):
    """Loss for TRADES training.
    :param criterion_ce: Cross entropy loss.
    :param beta: Weight of the robust loss.
    """

    def __init__(self, criterion_ce: nn.Module, beta: float = 1.0) -> None:
        super().__init__()

        self.beta = beta
        self.criterion_ce = criterion_ce
        self.criterion_kl = nn.KLDivLoss(size_average=False)

    def forward(self, logits: Tensor, adv_logits: Tensor, target: Tensor) -> Tensor:
        batch_size = logits.size(0)
        loss_natural = self.criterion_ce(logits, target)
        loss_robust = (1.0 / batch_size) * self.criterion_kl(
            F.log_softmax(logits, dim=1), F.softmax(adv_logits.detach(), dim=1)
        )
        # print(loss_natural, loss_robust)
        return loss_natural + self.beta * loss_robust
