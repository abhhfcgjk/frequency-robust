from typing import Any, Tuple

import ptwt
import torch
import torch.nn as nn
from torch import Tensor

from ufbrp.attacks.autoattack import AutoAttack
from ufbrp.attacks.base import Attacker
from ufbrp.attacks.fgsm import FGSM
from ufbrp.attacks.pgd import PGD

from .resnet import ResNet34, ResNet50


class DBFTT(nn.Module):
    def __init__(self, wavelet_level=4, wavelet_method="haar", num_classes=10, jacobian_delta=0.005):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.wavelet_method = wavelet_method
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta

        model = ResNet50(self.num_classes)
        self.low_model = nn.Sequential(*list(model.children())[:-1])
        self.low_conv = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=True)
        model = ResNet34(self.num_classes)
        self.high_model = nn.Sequential(*list(model.children())[:-1])
        # self.high_conv = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=True)
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

        # Orthogonal wavelet decomposition
        coeffs = ptwt.wavedec2(batch, self.wavelet_method, level=self.wavelet_level, mode="constant")

        # Reconstruct domains
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
        with torch.no_grad():
            f_x = self.high_model(x)
            f_xn = self.high_model(x + noise)
        diff = f_x - f_xn
        return self.jacobian_delta * diff.norm(p="fro") ** 2 / bs

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
        feature_high = self.high_model(x_high)
        # feature_high = self.high_conv(feature_high)
        self.__penalty = self.__jacobian_penalty(x_high)
        feature_low = self.low_model(x_low)
        feature_low = self.low_conv(feature_low)
        beta = self.softmax(self.softmax_weights).repeat(batches, 1, 1, 1)
        feature = beta * feature_high + (1 - beta) * feature_low
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output


class AdvDBFTT(nn.Module):
    def __init__(self, wavelet_level=4, num_classes=10, jacobian_delta=0.005):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta

        model = ResNet50(self.num_classes)
        self.low_model = nn.Sequential(*list(model.children())[:-1])
        # self.low_conv = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=True)
        model = ResNet50(self.num_classes)
        self.high_model = nn.Sequential(*list(model.children())[:-1])
        self.__apply_spectral_norm(self.high_model)
        self.softmax_weights = nn.Parameter(torch.randn((512, 1, 1)))
        self.softmax = nn.Softmax(dim=1)  # (B, C, H, W)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

        attack_config = {"type": "fgsm", "params": {"eps": 8.0, "alpha": 2.5, "mode": "zero"}}
        self.attacker = self._init_attack(attack_config)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coeffs = ptwt.wavedec2(batch, "coif5", level=self.wavelet_level, mode="symmetric")
        high_freq = self.__reconstruct_from_coeffs(coeffs, keep_approx=False, keep_details=True)
        low_freq = self.__reconstruct_from_coeffs(coeffs, keep_approx=True, keep_details=False)

        return high_freq, low_freq

    @staticmethod
    def __reconstruct_from_coeffs(coeffs: list, keep_approx: bool = True, keep_details: bool = True) -> torch.Tensor:
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
        print(self.__penalty)
        return self.__penalty

    def __apply_spectral_norm(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, nn.utils.spectral_norm(module))
            else:
                self.__apply_spectral_norm(module)
        return model

    def _init_attack(self, attack_config: dict[str, Any], *args, **kwargs) -> Attacker:
        attack_name = attack_config["type"]
        if attack_name == "none":
            return None

        attackers = {"fgsm": FGSM, "pgd": PGD, "autoattack": AutoAttack}
        attacker_cls = attackers.get(attack_name)

        if attacker_cls is None:
            raise RuntimeError(f"Unknown attack `{attack_name}`")

        if attack_name == "autoattack":
            loss = nn.CrossEntropyLoss(reduction="none")

            def loss_computer(y, target):
                return loss(y, target)

        else:
            loss = nn.CrossEntropyLoss()

            def loss_computer(y, target):
                return loss(y, target)

        attacker = attacker_cls(
            model=nn.Sequential(self.high_model, self.avgpool, nn.Flatten(), self.fc),
            loss_computer=loss_computer,
            **attack_config["params"],
        )
        return attacker

    def forward(self, x: Tensor) -> Tensor:
        batches = x.shape[0]
        x_high, x_low = self.__decompose_high_low_domains(x)
        feature_high = self.high_model(x_high)
        self.__penalty = self.__jacobian_penalty(x_high)
        feature_low = self.low_model(x_low)
        # feature_low = self.low_conv(feature_low)
        beta = self.softmax(self.softmax_weights).repeat(batches, 1, 1, 1)
        feature = beta * feature_high + (1 - beta) * feature_low
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output, self.label.repeat(2)
