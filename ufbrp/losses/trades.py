"""
Loss for TRADES training.

| Paper: https://arxiv.org/abs/1901.08573
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TRADESLoss(nn.Module):
    def __init__(self, criterion_ce: nn.Module, beta: float = 1.0):
        super().__init__()

        self.beta = beta
        self.criterion_ce = criterion_ce
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits, adv_logits, target):
        loss_natural = self.criterion_ce(logits, target)

        loss_robust = self.criterion_kl(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(logits.detach(), dim=1),
        )

        return loss_natural + self.beta * loss_robust