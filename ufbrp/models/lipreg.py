from typing import Tuple

import ptwt
import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from .resnet import ResNet50


class LipReg(nn.Module):
    def __init__(self, wavelet_level=4, wavelet_method="haar", num_classes=10, jacobian_delta=0.5, k=0.5):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.wavelet_method = wavelet_method
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta
        self.k = k

        model = ResNet50(num_classes=self.num_classes)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

    def __decompose_high_low_domains(self, batch: torch.Tensor):
        with autocast(enabled=False, device_type="cuda"):
            batch_fp32 = batch.float()

            coeffs = ptwt.wavedec2(
                batch_fp32,
                self.wavelet_method,
                level=self.wavelet_level,
                mode="constant",
            )

            high_freq = self.__reconstruct_from_coeffs(
                coeffs, keep_approx=False, keep_details=True, wavelet_method=self.wavelet_method,
            )
            low_freq = self.__reconstruct_from_coeffs(
                coeffs, keep_approx=True, keep_details=False, wavelet_method=self.wavelet_method,
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
                if keep_approx:
                    modified_coeffs.append(level.float())
                else:
                    modified_coeffs.append(torch.zeros_like(level, dtype=torch.float32))
            else:  # Detail coefficients
                if keep_details:
                    modified_coeffs.append(tuple(d.float() for d in level))
                else:
                    modified_coeffs.append(
                        tuple(torch.zeros_like(d, dtype=torch.float32) for d in level)
                    )
        with autocast(enabled=False, device_type="cuda"):
            out = ptwt.waverec2(modified_coeffs, wavelet_method)
        return out

    def __jacobian_penalty(self, x, sigma=0.25):
        bs = x.shape[0]
        w = x.shape[-1]
        k = (w / 32)**2
        with autocast(enabled=False, device_type="cuda"):
            x_fp32 = x.float()
            noise = torch.randn_like(x_fp32) * sigma
            out1 = self.model(x_fp32)
            out2 = self.model(x_fp32 + noise)
            difference = out1 - out2
            norm = torch.norm(difference, p="fro")
        # penalty = self.jacobian_delta * norm * norm/ (bs)
        penalty = self.jacobian_delta * norm / (bs*k)
        return penalty


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
        if self.training:
            self.__penalty = self.__jacobian_penalty(x_decompose)
        feature = self.model(x_decompose)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output
    # def forward(self, x: Tensor) -> Tensor:
    #     # x is FP16 under AMP
    #     high_fp32, low_fp32 = self.__decompose_high_low_domains(x)

    #     # IMPORTANT: cast BACK to FP16 before backbone
    #     high = high_fp32.to(dtype=x.dtype)
    #     low = low_fp32.to(dtype=x.dtype)

    #     high.requires_grad_(True)
    #     high.retain_grad()

    #     x_decompose = high + low

    #     if self.training:
    #         self.__penalty = self.__jacobian_penalty(x_decompose)

    #     feature = self.model(x_decompose)
    #     pool = self.avgpool(feature)
    #     pool = pool.view(pool.size(0), -1)
    #     output = self.fc(pool)

    #     self.x_high = high  # for grad clamp later

    #     # print(
    #     #     "x grad:",
    #     #     x.grad.abs().mean().item(),
    #     #     "high grad:",
    #     #     self.x_high.grad.abs().mean().item(),
    #     #     "fc grad:",
    #     #     self.fc.weight.grad.abs().mean().item(),
    #     # )

    #     return output
