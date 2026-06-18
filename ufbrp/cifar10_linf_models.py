from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ufbrp.models import (
    LipReg,
    LipReg_aa,
    LipReg_aa_WideResNet,
    ResNet50,
    ResNet50Blur,
    WideResNet_L,
    create_model,
)

# CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR10_STD  = (0.2675, 0.2565, 0.2761)

DEFAULT_MODEL_DIR_CANDIDATES = (
    Path("models/Linf/cifar100"),
    Path("models/cifar100/Linf"),
)


@dataclass
class LoadedModel:
    model: nn.Module
    arch: str
    checkpoint: Path
    load_target: str
    load_variant: str
    missing_keys: list[str]
    unexpected_keys: list[str]
    state_dict_keys: int


def resolve_cifar10_linf_model_dir(models_dir: str | Path | None = None) -> Path:
    if models_dir is not None:
        path = Path(models_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {path}")
        return path

    for candidate in DEFAULT_MODEL_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate

    candidates = ", ".join(str(path) for path in DEFAULT_MODEL_DIR_CANDIDATES)
    raise FileNotFoundError(f"Could not find CIFAR-10 Linf model directory. Checked: {candidates}")


def iter_cifar10_linf_checkpoints(models_dir: str | Path | None = None) -> list[Path]:
    root = resolve_cifar10_linf_model_dir(models_dir)
    return sorted(root.glob("*.pt"))


def infer_cifar10_linf_arch(checkpoint_path: str | Path) -> str:
    name = Path(checkpoint_path).stem.lower().replace("-", "_")

    if name.startswith("wideresnet"):
        if "lipreg" in name:
            return "wideresnet_lipreg_aa"
        return "wideresnet"

    if "lipreg" in name and "_aa" in name:
        return "lipreg_aa"
    if "lipreg" in name:
        return "lipreg"
    if name.startswith("resnet_aa") or name.startswith("resnet50_aa"):
        return "resnet50_aa"
    return "resnet50"


def build_cifar10_linf_backbone(arch: str) -> nn.Module:
    if arch == "resnet50":
        return ResNet50(num_classes=100)
    if arch == "resnet50_aa":
        return ResNet50Blur(num_classes=100, filter_size=5, learnable=True)
    if arch == "lipreg":
        return LipReg(wavelet_level=4, wavelet_method="haar", num_classes=100, jacobian_delta=0.0, k=1.0)
    if arch == "lipreg_aa":
        return LipReg_aa(
            wavelet_level=2,
            wavelet_method="haar",
            num_classes=100,
            jacobian_delta=0.5,
            k=1.0,
            filter_size=5,
            learnable=True,
        )
    if arch == "wideresnet":
        return WideResNet_L(num_classes=100)
    if arch == "wideresnet_lipreg_aa":
        return LipReg_aa_WideResNet(
            wavelet_level=2,
            wavelet_method="haar",
            num_classes=100,
            jacobian_delta=0.5,
            k=1.0,
            filter_size=5,
            learnable=True,
        )

    raise ValueError(f"Unsupported CIFAR-10 Linf architecture: {arch}")


def build_cifar10_linf_model(arch: str) -> nn.Sequential:
    backbone = build_cifar10_linf_backbone(arch)
    return create_model(backbone, mean=CIFAR10_MEAN, std=CIFAR10_STD)


def load_cifar10_linf_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    arch: str | None = None,
) -> LoadedModel:
    checkpoint_path = Path(checkpoint_path)
    arch = arch or infer_cifar10_linf_arch(checkpoint_path)
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    state_dict = _strip_distributed_prefixes(state_dict)

    errors: list[str] = []
    variants = {
        "as_saved": state_dict,
        "prefixed_for_wrapper": _prefix_for_wrapper(state_dict),
        "stripped_for_backbone": _strip_for_backbone(state_dict),
    }

    for target_name, variant_names in (
        ("wrapper", ("as_saved", "prefixed_for_wrapper")),
        ("backbone", ("stripped_for_backbone", "as_saved")),
    ):
        for variant_name in variant_names:
            for strict in (True, False):
                wrapped_model = build_cifar10_linf_model(arch)
                target = wrapped_model if target_name == "wrapper" else wrapped_model.model

                try:
                    result = target.load_state_dict(variants[variant_name], strict=strict)
                except RuntimeError as exc:
                    errors.append(f"{target_name}/{variant_name}/strict={strict}: {exc}")
                    continue

                missing = list(result.missing_keys)
                unexpected = list(result.unexpected_keys)
                if strict or _is_acceptable_non_strict_load(target_name, missing, unexpected):
                    wrapped_model.to(device)
                    wrapped_model.eval()
                    return LoadedModel(
                        model=wrapped_model,
                        arch=arch,
                        checkpoint=checkpoint_path,
                        load_target=target_name,
                        load_variant=variant_name,
                        missing_keys=missing,
                        unexpected_keys=unexpected,
                        state_dict_keys=len(state_dict),
                    )

                errors.append(
                    f"{target_name}/{variant_name}/strict={strict}: "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )

    summary = "\n".join(errors[-6:])
    raise RuntimeError(f"Could not load {checkpoint_path.name} as {arch}. Last attempts:\n{summary}")


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        return dict(checkpoint)

    raise TypeError(f"Unsupported checkpoint object in {checkpoint_path}: {type(checkpoint)!r}")


def _strip_distributed_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        clean[new_key] = value
    return clean


def _prefix_for_wrapper(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    prefixed = {}
    for key, value in state_dict.items():
        if key.startswith("model.") or key.startswith("normalize."):
            prefixed[key] = value
        else:
            prefixed[f"model.{key}"] = value
    return prefixed


def _strip_for_backbone(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("normalize."):
            continue
        if key.startswith("model."):
            stripped[key[len("model.") :]] = value
        else:
            stripped[key] = value
    return stripped


def _is_acceptable_non_strict_load(target_name: str, missing: list[str], unexpected: list[str]) -> bool:
    if unexpected:
        return False
    if not missing:
        return True
    if target_name == "wrapper":
        return set(missing) <= {"normalize.mean", "normalize.std"}
    return False
