import torch
import torch.nn as nn


class NMSERegularizationLoss(nn.Module):
    """
    Implements the Normalized Mean Square Error (NMSE) regularization loss.

    L_NMSE = (1 - p_clean) * || f(x)/||f(x)||_2 - f(x_adv)/||f(x_adv)||_2 ||_2^2
    """

    def __init__(self, p_clean: float = 0.5):
        super().__init__()
        if not (0.0 <= p_clean <= 1.0):
            raise ValueError("p_clean must be between 0.0 and 1.0")
        self.p_clean = p_clean

    def _unit_normalize(self, f_output: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.matrix_norm(f_output, ord="fro", keepdim=True)
        # print(norm.shape, f_output.shape)
        epsilon = 1e-6
        return f_output / (norm + epsilon)

    def forward(self, f_clean: torch.Tensor, f_adv: torch.Tensor) -> torch.Tensor:
        f_clean_norm = self._unit_normalize(f_clean)
        f_adv_norm = self._unit_normalize(f_adv)
        sq_diff = (f_clean_norm - f_adv_norm).pow(2)
        loss_per_sample = sq_diff.sum(dim=list(range(1, sq_diff.dim())))
        weight = 1.0 - self.p_clean
        loss = weight * torch.mean(loss_per_sample)
        return loss
