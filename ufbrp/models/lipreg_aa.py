from typing import Tuple

import ptwt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .resnet_blur import ResNet50Blur
from .blurpool import BlurPool
from .wideresnet import WideResNetBlur_L

class LipReg_aa(nn.Module):
    def __init__(
        self,
        wavelet_level=4,
        wavelet_method="haar",
        num_classes=10,
        jacobian_delta=0.5,
        k=0.5,
        filter_size=5,
        learnable=False,
    ):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.wavelet_method = wavelet_method
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta
        self.k = k
        self.filter_size = filter_size
        self.learnable = learnable

        model = ResNet50Blur(num_classes=self.num_classes, filter_size=self.filter_size, learnable=self.learnable)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes a batch of images into high and low frequency domains using wavelet transform.

        Args:
            batch: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (high_freq, low_freq) tensors with same shape as input
        """
        coeffs = ptwt.wavedec2(batch, self.wavelet_method, level=self.wavelet_level, mode="constant")
        high_freq = self.__reconstruct_from_coeffs(
            coeffs, keep_approx=False, keep_details=True, wavelet_method=self.wavelet_method
        )
        low_freq = self.__reconstruct_from_coeffs(
            coeffs, keep_approx=True, keep_details=False, wavelet_method=self.wavelet_method
        )
        return high_freq, low_freq

    @staticmethod
    def __reconstruct_from_coeffs(
        coeffs: list, keep_approx: bool = True, keep_details: bool = True, wavelet_method: str = "haar"
    ) -> torch.Tensor:
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
        return ptwt.waverec2(modified_coeffs, wavelet_method)

    def __jacobian_penalty(self, x, sigma=0.25):
        bs = x.shape[0]
        noise = torch.randn_like(x) * sigma
        difference = self.model(x) - self.model(x + noise)
        norm = torch.norm(difference, p="fro")
        return self.jacobian_delta * norm * norm / bs

    def pre_penalty(self, *args, **kwargs):
        return self.__penalty

    def post_penalty(self, x, *args, **kwargs) -> None:
        self.clamp_grad(x)

    @property
    def penalty(self):
        return self.pre_penalty()

    def clamp_grad(self, x):
        # print("GRad", x.grad)
        g_high = self.x_high.grad
        mask = g_high.abs() > self.k * x.grad.abs()
        x.grad[mask] = torch.clamp(x.grad[mask], min=-self.k, max=self.k)

    def forward(self, x: Tensor) -> Tensor:
        self.x_high, x_low = self.__decompose_high_low_domains(x)
        self.x_high.requires_grad_(True)
        self.x_high.retain_grad()
        x_low = x_low.detach()
        x_decompose = self.x_high + x_low
        self.__penalty = self.__jacobian_penalty(x_decompose)
        feature = self.model(x_decompose)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output



class LipReg_aa_WideResNet(nn.Module):
    def __init__(
        self,
        wavelet_level=4,
        wavelet_method="haar",
        num_classes=10,
        jacobian_delta=0.5,
        k=0.5,
        filter_size=5,
        learnable=False,
    ):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.wavelet_method = wavelet_method
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta
        self.k = k
        self.filter_size = filter_size
        self.learnable = learnable

        model = WideResNetBlur_L(num_classes=self.num_classes, filter_size=self.filter_size, learnable=self.learnable)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=self.num_classes, bias=True)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes a batch of images into high and low frequency domains using wavelet transform.

        Args:
            batch: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (high_freq, low_freq) tensors with same shape as input
        """
        coeffs = ptwt.wavedec2(batch, self.wavelet_method, level=self.wavelet_level, mode="constant")
        high_freq = self.__reconstruct_from_coeffs(
            coeffs, keep_approx=False, keep_details=True, wavelet_method=self.wavelet_method
        )
        low_freq = self.__reconstruct_from_coeffs(
            coeffs, keep_approx=True, keep_details=False, wavelet_method=self.wavelet_method
        )
        return high_freq, low_freq

    @staticmethod
    def __reconstruct_from_coeffs(
        coeffs: list, keep_approx: bool = True, keep_details: bool = True, wavelet_method: str = "haar"
    ) -> torch.Tensor:
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
        return ptwt.waverec2(modified_coeffs, wavelet_method)

    def __jacobian_penalty(self, x, sigma=0.25):
        bs = x.shape[0]
        noise = torch.randn_like(x) * sigma
        difference = self.model(x) - self.model(x + noise)
        norm = torch.norm(difference, p="fro")
        return self.jacobian_delta * norm * norm / bs

    def pre_penalty(self, *args, **kwargs):
        return self.__penalty

    def post_penalty(self, x, *args, **kwargs) -> None:
        self.clamp_grad(x)

    @property
    def penalty(self):
        return self.pre_penalty()

    def clamp_grad(self, x):
        # print("GRad", x.grad)
        g_high = self.x_high.grad
        mask = g_high.abs() > self.k * x.grad.abs()
        x.grad[mask] = torch.clamp(x.grad[mask], min=-self.k, max=self.k)

    def forward(self, x: Tensor) -> Tensor:
        self.x_high, x_low = self.__decompose_high_low_domains(x)
        self.x_high.requires_grad_(True)
        self.x_high.retain_grad()
        x_low = x_low.detach()
        x_decompose = self.x_high + x_low
        self.__penalty = self.__jacobian_penalty(x_decompose)
        feature = self.model(x_decompose)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output