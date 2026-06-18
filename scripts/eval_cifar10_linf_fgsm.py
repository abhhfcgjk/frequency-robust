#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from ufbrp.attacks.fgsm import FGSM
from ufbrp.cifar10_linf_models import (
    infer_cifar10_linf_arch,
    iter_cifar10_linf_checkpoints,
    load_cifar10_linf_model,
)


FGSM_PRESETS = (
    {"eps": 8.0, "alpha": 8.0, "mode": "zero"},
    {"eps": 4.0, "alpha": 4.0, "mode": "zero"},
)

CSV_FIELDS = [
    "checkpoint",
    "arch",
    "attack",
    "eps",
    "alpha",
    "mode",
    "clean_accuracy",
    "robust_accuracy",
    "clean_correct",
    "robust_correct",
    "total",
    "load_target",
    "load_variant",
    "missing_keys",
    "unexpected_keys",
    "state_dict_keys",
    "device",
    "seconds",
    "status",
    "error",
]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    arch_overrides = parse_arch_overrides(args.arch)
    loader = build_cifar10_loader(args)
    checkpoints = iter_cifar10_linf_checkpoints(args.models_dir)
    if args.limit_models is not None:
        checkpoints = checkpoints[: args.limit_models]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for checkpoint in checkpoints:
            model_started_at = time.perf_counter()
            arch = arch_overrides.get(checkpoint.name) or arch_overrides.get(checkpoint.stem)
            arch = arch or infer_cifar10_linf_arch(checkpoint)

            try:
                loaded = load_cifar10_linf_model(checkpoint, device=device, arch=arch)
                clean_correct, total = evaluate_clean(loaded.model, loader, device, args.limit_batches)
                clean_accuracy = clean_correct / total
            except Exception as exc:
                writer.writerow(
                    base_row(
                        checkpoint=checkpoint,
                        arch=arch,
                        device=device,
                        status="load_failed",
                        seconds=time.perf_counter() - model_started_at,
                        error=repr(exc),
                    )
                )
                csv_file.flush()
                continue

            for preset in FGSM_PRESETS:
                row = base_row(
                    checkpoint=checkpoint,
                    arch=arch,
                    device=device,
                    status="ok",
                    seconds=0.0,
                    error="",
                )
                row.update(
                    {
                        "attack": "fgsm",
                        "eps": preset["eps"],
                        "alpha": preset["alpha"],
                        "mode": preset["mode"],
                        "clean_accuracy": f"{clean_accuracy:.6f}",
                        "clean_correct": clean_correct,
                        "total": total,
                        "load_target": loaded.load_target,
                        "load_variant": loaded.load_variant,
                        "missing_keys": ";".join(loaded.missing_keys),
                        "unexpected_keys": ";".join(loaded.unexpected_keys),
                        "state_dict_keys": loaded.state_dict_keys,
                    }
                )

                attack_started_at = time.perf_counter()
                try:
                    robust_correct, robust_total = evaluate_fgsm(
                        loaded.model,
                        loader,
                        device,
                        preset,
                        args.limit_batches,
                    )
                    row["robust_correct"] = robust_correct
                    row["robust_accuracy"] = f"{robust_correct / robust_total:.6f}"
                    row["total"] = robust_total
                except Exception as exc:
                    row["status"] = "attack_failed"
                    row["error"] = repr(exc)
                finally:
                    row["seconds"] = f"{time.perf_counter() - attack_started_at:.3f}"
                    writer.writerow(row)
                    csv_file.flush()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"FGSM evaluation results saved to {args.output}")


def build_cifar10_loader(args: argparse.Namespace) -> DataLoader:
    dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=args.download,
    )
    if args.limit_samples is not None:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[int, int]:
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="clean", leave=False)):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    if total == 0:
        raise RuntimeError("No samples were evaluated.")
    return correct, total


def evaluate_fgsm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    preset: dict[str, float | str],
    limit_batches: int | None,
) -> tuple[int, int]:
    correct = 0
    total = 0
    model.eval()
    loss = nn.CrossEntropyLoss()
    attacker = FGSM(model=model, loss_computer=loss, **preset)

    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"fgsm eps={preset['eps']}", leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        attacked_inputs = attacker.run(inputs, labels)

        with torch.no_grad():
            preds = model(attacked_inputs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    if total == 0:
        raise RuntimeError("No samples were evaluated.")
    return correct, total


def base_row(
    checkpoint: Path,
    arch: str,
    device: torch.device,
    status: str,
    seconds: float,
    error: str,
) -> dict[str, str | int | float]:
    return {
        "checkpoint": str(checkpoint),
        "arch": arch,
        "attack": "",
        "eps": "",
        "alpha": "",
        "mode": "",
        "clean_accuracy": "",
        "robust_accuracy": "",
        "clean_correct": "",
        "robust_correct": "",
        "total": "",
        "load_target": "",
        "load_variant": "",
        "missing_keys": "",
        "unexpected_keys": "",
        "state_dict_keys": "",
        "device": str(device),
        "seconds": f"{seconds:.3f}",
        "status": status,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all CIFAR-10 Linf checkpoints with two FGSM presets.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory containing .pt checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/CIFAR10"), help="CIFAR-10 root directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ufbrp/results-cifar10/linf_cifar10_fgsm_all_models.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-models", type=int, default=None, help="Optional smoke-test checkpoint limit.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Optional smoke-test batch limit.")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CIFAR-10 if it is missing.",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=[],
        metavar="CHECKPOINT=ARCH",
        help="Override inferred arch for a checkpoint name or stem.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_arch_overrides(items: list[str]) -> dict[str, str]:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --arch override {item!r}; expected CHECKPOINT=ARCH")
        checkpoint, arch = item.split("=", 1)
        overrides[checkpoint] = arch
    return overrides


if __name__ == "__main__":
    main()
