import ptwt
import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from typing import Tuple

from .resnet_blur import ResNet50Blur
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

    # def _logits_from_input(self, x: Tensor) -> Tensor:
    #     feat = self.model(x)
    #     feat = self.avgpool(feat)
    #     feat = torch.flatten(feat, 1)
    #     logits = self.fc(feat)
    #     return logits

    # def __jacobian_penalty(self, x: Tensor, sigma=0.25) -> Tensor:
    #     device_type = x.device.type
    #     with autocast(device_type=device_type, enabled=False):
    #         x_fp32 = x.float()
    #         noise = torch.randn_like(x_fp32) * sigma
    #         x_noisy = (x_fp32 + noise).clamp(0.0, 1.0)

    #         logits_1 = self._logits_from_input(x_fp32)
    #         logits_2 = self._logits_from_input(x_noisy)

    #         diff = logits_1 - logits_2
    #         penalty = self.jacobian_delta * diff.pow(2).sum(dim=1).mean()
    #     print(penalty)
    #     return penalty

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
        # x_low = x_low.detach()
        x_decompose = self.x_high + x_low
        if self.training:
            self.__penalty = self.__jacobian_penalty(x_decompose)
        feature = self.model(x_decompose)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output



# class LipReg_aa(nn.Module):
#     def __init__(
#         self,
#         wavelet_level=4,
#         wavelet_method="haar",
#         num_classes=10,
#         jacobian_delta=0.5,
#         k=0.5,
#         filter_size=5,
#         learnable=False,
#         sigma=8 / 255,   # noise scale for Jacobian penalty
#     ):
#         super().__init__()
#         self.wavelet_level = wavelet_level
#         self.wavelet_method = wavelet_method
#         self.num_classes = num_classes
#         self.jacobian_delta = jacobian_delta
#         self.k = k
#         self.filter_size = filter_size
#         self.learnable = learnable
#         self.sigma = sigma

#         backbone = ResNet50Blur(
#             num_classes=self.num_classes,
#             filter_size=self.filter_size,
#             learnable=self.learnable,
#         )

#         # backbone without final linear layer
#         self.backbone = nn.Sequential(*list(backbone.children())[:-1])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, self.num_classes, bias=True)

#         self.__penalty = 0.0

#     def __decompose_high_low_domains(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
#         device_type = batch.device.type
#         with autocast(device_type=device_type, enabled=False):
#             batch_fp32 = batch.float()

#             coeffs = ptwt.wavedec2(
#                 batch_fp32,
#                 self.wavelet_method,
#                 level=self.wavelet_level,
#                 mode="constant",
#             )

#             high_freq = self.__reconstruct_from_coeffs(
#                 coeffs,
#                 keep_approx=False,
#                 keep_details=True,
#                 wavelet_method=self.wavelet_method,
#             )
#             low_freq = self.__reconstruct_from_coeffs(
#                 coeffs,
#                 keep_approx=True,
#                 keep_details=False,
#                 wavelet_method=self.wavelet_method,
#             )

#         return high_freq, low_freq

#     @staticmethod
#     def __reconstruct_from_coeffs(
#         coeffs: list,
#         keep_approx: bool = True,
#         keep_details: bool = True,
#         wavelet_method: str = "haar",
#     ) -> Tensor:
#         modified_coeffs = []
#         for i, level in enumerate(coeffs):
#             if i == 0:
#                 if keep_approx:
#                     modified_coeffs.append(level.float())
#                 else:
#                     modified_coeffs.append(torch.zeros_like(level, dtype=torch.float32))
#             else:
#                 if keep_details:
#                     modified_coeffs.append(tuple(d.float() for d in level))
#                 else:
#                     modified_coeffs.append(
#                         tuple(torch.zeros_like(d, dtype=torch.float32) for d in level)
#                     )

#         return ptwt.waverec2(modified_coeffs, wavelet_method)

#     def _clamp_high_grad_hook(self, grad: Tensor) -> Tensor:
#         if self.k is None or self.k <= 0:
#             return grad
#         return grad.clamp(min=-self.k, max=self.k)

#     def _build_input(self, x: Tensor, clamp_high_grad: bool = False) -> Tensor:
#         x_high, x_low = self.__decompose_high_low_domains(x)
#         if clamp_high_grad and self.training:
#             if not x_high.requires_grad:
#                 x_high = x_high.detach().requires_grad_(True)
#             x_high.register_hook(self._clamp_high_grad_hook)

#         x_recompose = x_high + x_low
#         return x_recompose

#     def _logits_from_input(self, x: Tensor) -> Tensor:
#         feat = self.backbone(x)
#         feat = self.avgpool(feat)
#         feat = torch.flatten(feat, 1)
#         logits = self.fc(feat)
#         return logits

#     def __jacobian_penalty(self, x: Tensor) -> Tensor:
#         device_type = x.device.type
#         with autocast(device_type=device_type, enabled=False):
#             x_fp32 = x.float()
#             noise = torch.randn_like(x_fp32) * self.sigma
#             x_noisy = (x_fp32 + noise).clamp(0.0, 1.0)

#             logits_1 = self._logits_from_input(x_fp32)
#             logits_2 = self._logits_from_input(x_noisy)

#             diff = logits_1 - logits_2
#             penalty = self.jacobian_delta * diff.pow(2).sum(dim=1).mean()
#         print(penalty)
#         return penalty

#     def forward(self, x: Tensor) -> Tensor:
#         x_recompose = self._build_input(x, clamp_high_grad=self.training)
#         logits = self._logits_from_input(x_recompose)

#         if not self.training:
#             self.__penalty = self.__jacobian_penalty(x_recompose)

#         return logits

#     @property
#     def penalty(self):
#         return self.pre_penalty()

#     def pre_penalty(self, *args, **kwargs):
#         return self.__penalty


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
