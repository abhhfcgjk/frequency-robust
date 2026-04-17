from typing import Tuple

import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet50


class Consistency(nn.Module):
    def __init__(self, num_classes=10, wavelet_level=4, jacobian_delta=0.005):
        super(Consistency, self).__init__()
        self.wavelet_level = wavelet_level
        self.num_classes = num_classes
        self.jacobian_delta = jacobian_delta
        model = ResNet50(num_classes=self.num_classes)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
        self.register_spectral_value_hook()

    def register_spectral_value_hook(self):
        for m in self.modules():
            if hasattr(m, "weight"):
                m.register_forward_hook(self.spectral_value_hook)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes a batch of images into high and low frequency domains using wavelet transform.

        Args:
            batch: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (high_freq, low_freq) tensors with same shape as input
        """
        coeffs = ptwt.wavedec2(batch, "haar", level=self.wavelet_level, mode="constant")
        high_freq = self.__reconstruct_from_coeffs(coeffs, keep_approx=False, keep_details=True)
        low_freq = self.__reconstruct_from_coeffs(coeffs, keep_approx=True, keep_details=False)
        return high_freq, low_freq

    @staticmethod
    def __reconstruct_from_coeffs(coeffs: list, keep_approx: bool = True, keep_details: bool = True) -> torch.Tensor:
        """
        Reconstructs image from modified wavelet coefficients.

        Args:
            coeffs: Wavelet coefficients from ptwt.wavedec2
            keep_approx: Whether to keep approximation coefficients
            keep_details: Whether to keep detail coefficients

        Returns:
            Reconstructed tensor
        """
        modified_coeffs = []
        for i, level in enumerate(coeffs):
            if i == 0:
                modified_coeffs.append(level if keep_approx else torch.zeros_like(level))
            else:
                if keep_details:
                    modified_coeffs.append(level)
                else:
                    modified_coeffs.append(tuple(torch.zeros_like(d) for d in level))
        return ptwt.waverec2(modified_coeffs, "haar")

    def __jacobian_penalty(self, x, sigma=0.25):
        noise = torch.randn_like(x) * sigma
        difference = self.model(x) - self.model(x + noise)
        norm = torch.norm(difference, p="fro")
        return self.jacobian_delta * norm

    def __singular_penalty(self, beta=30):
        penalty = 0
        for module in self.modules():
            if hasattr(module, "weight"):
                penalty += max(0, module.singular_value - beta)
        # print(penalty)
        return penalty

    # @property
    def penalty(self):
        return self.__penalty  # + self.__singular_penalty()

    def spectral_value_hook(self, module, input, output):
        if hasattr(module, "weight"):
            module.singular_value = self.compute_singular_value(module.weight)

    @staticmethod
    def compute_singular_value(weight, n_iter=5):
        w = weight.view(weight.shape[0], -1)
        u = F.normalize(torch.randn(w.size(0), device=w.device), dim=0, eps=1e-12)
        for _ in range(n_iter):
            v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
        return torch.dot(u, torch.mv(w, v)).item()

    def forward(self, x):
        x_high, _ = self.__decompose_high_low_domains(x)
        self.__penalty = self.__jacobian_penalty(x_high)
        feature = self.model(x)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output
