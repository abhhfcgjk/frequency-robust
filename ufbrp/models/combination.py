from typing import Tuple

import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .irevnet import RevNetSmall
from .resnet import ResNet34, ResNet50


class RevNetDBFTT(nn.Module):
    def __init__(self, wavelet_level=4, num_classes=10, jacobian_delta=0.005, inv_delta=0.005, in_shape=[3, 32, 32]):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta
        self.inv_delta = inv_delta

        model = ResNet50(self.num_classes)
        self.low_model = nn.Sequential(*list(model.children())[:-1])
        self.low_conv = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=True)

        self.high_model = RevNetSmall(self.num_classes, in_shape, return_bij=True, pretrained=True)
        self.__apply_spectral_norm(self.high_model)
        self.softmax_weights = nn.Parameter(torch.randn((512, 1, 1)))  # dim is Channels count
        self.softmax = nn.Softmax(dim=1)  # (B, C, H, W)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes a batch of images into high and low frequency domains using wavelet transform.

        Args:
            batch: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (high_freq, low_freq) tensors with same shape as input
        """
        coeffs = ptwt.wavedec2(batch, "coif5", level=self.wavelet_level, mode="constant")
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
            if i == 0:  # Approximation coefficients
                modified_coeffs.append(level if keep_approx else torch.zeros_like(level))
            else:  # Detail coefficients
                if keep_details:
                    modified_coeffs.append(level)
                else:
                    modified_coeffs.append(tuple(torch.zeros_like(d) for d in level))
        return ptwt.waverec2(modified_coeffs, "coif5")

    def __jacobian_penalty(self, x, sigma=0.25):
        noise = torch.randn_like(x) * sigma
        difference = self.high_model(x)[1] - self.high_model(x + noise)[1]
        norm = torch.norm(difference, p="fro")
        return self.jacobian_delta * norm * norm

    def __inverse_penalty(self, x, feature_high):
        inv_x = self.high_model.inverse(feature_high)
        return self.inv_delta * torch.norm((x - inv_x), p="fro")

    @property
    def penalty(self):
        return self.__penalty

    def __apply_spectral_norm(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, nn.utils.spectral_norm(module))
            else:
                self.__apply_spectral_norm(module)
        return model

    def forward(self, x: Tensor) -> Tensor:
        batches = x.shape[0]
        x_high, x_low = self.__decompose_high_low_domains(x)
        _, feature_high = self.high_model(x_high)
        self.__penalty = self.__jacobian_penalty(x_high)  # + self.__inverse_penalty(x_high, feature_high)
        feature_low = self.low_model(x_low)
        feature_low = self.low_conv(feature_low)
        beta = self.softmax(self.softmax_weights).repeat(batches, 1, 1, 1)
        feature = beta * feature_high + (1 - beta) * feature_low
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output


class ConsistencyDBFTT(nn.Module):
    def __init__(self, wavelet_level=4, num_classes=10, jacobian_delta=0.005):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.num_classes = num_classes
        self.__penalty = 0
        # self.jacobian_delta = jacobian_delta

        model = ResNet50(self.num_classes)
        self.low_model = nn.Sequential(*list(model.children())[:-1])
        self.low_conv = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=True)
        model = ResNet34(self.num_classes)
        self.high_model = nn.Sequential(*list(model.children())[:-1])
        self.__register_spectral_value_hook(self.high_model)
        # self.__apply_spectral_norm(self.high_model)
        self.softmax_weights = nn.Parameter(torch.randn((512, 1, 1)))  # dim is Channels count
        self.softmax = nn.Softmax(dim=1)  # (B, C, H, W)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes a batch of images into high and low frequency domains using wavelet transform.

        Args:
            batch: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (high_freq, low_freq) tensors with same shape as input
        """

        # Orthogonal wavelet decomposition
        coeffs = ptwt.wavedec2(batch, "coif5", level=self.wavelet_level, mode="constant")

        # Reconstruct domains
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
            if i == 0:  # Approximation coefficients
                modified_coeffs.append(level if keep_approx else torch.zeros_like(level))
            else:  # Detail coefficients
                if keep_details:
                    modified_coeffs.append(level)
                else:
                    modified_coeffs.append(tuple(torch.zeros_like(d) for d in level))
        return ptwt.waverec2(modified_coeffs, "coif5")

    def __jacobian_penalty(self, x, sigma=0.25):
        noise = torch.randn_like(x) * sigma
        difference = self.high_model(x) - self.high_model(x + noise)
        norm = torch.norm(difference, p="fro")
        return self.jacobian_delta * norm * norm

    @property
    def penalty(self):
        return self.__penalty + self.__singular_penalty(self.high_model)

    def __apply_spectral_norm(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, nn.utils.spectral_norm(module))
            else:
                self.__apply_spectral_norm(module)
        return model

    def spectral_value_hook(self, module, input, output):
        module.singular_value = self.compute_singular_value(module.weight)

    def __singular_penalty(self, module, beta=30):
        penalty = 0
        for module in module.modules():
            if hasattr(module, "weight"):
                penalty += max(0, module.singular_value - beta)
        # print(penalty)
        return penalty

    def __register_spectral_value_hook(self, module):
        for m in module.modules():
            if hasattr(m, "weight"):
                m.register_forward_hook(self.spectral_value_hook)

    @staticmethod
    def compute_singular_value(weight, n_iter=5):
        w = weight.view(weight.shape[0], -1)
        u = F.normalize(torch.randn(w.size(0), device=w.device), dim=0, eps=1e-12)
        for _ in range(n_iter):
            v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
        return torch.dot(u, torch.mv(w, v)).item()

    def forward(self, x: Tensor) -> Tensor:
        batches = x.shape[0]
        x_high, x_low = self.__decompose_high_low_domains(x)
        feature_high = self.high_model(x_high)
        # self.__penalty = self.__jacobian_penalty(x_high)
        feature_low = self.low_model(x_low)
        feature_low = self.low_conv(feature_low)
        beta = self.softmax(self.softmax_weights).repeat(batches, 1, 1, 1)
        feature = beta * feature_high + (1 - beta) * feature_low
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output
