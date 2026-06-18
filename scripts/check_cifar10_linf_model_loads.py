#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from ufbrp.cifar10_linf_models import (
    infer_cifar10_linf_arch,
    iter_cifar10_linf_checkpoints,
    load_cifar10_linf_model,
)


CSV_FIELDS = [
    "checkpoint",
    "arch",
    "status",
    "output_shape",
    "load_target",
    "load_variant",
    "missing_keys",
    "unexpected_keys",
    "state_dict_keys",
    "device",
    "seconds",
    "error",
]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    arch_overrides = parse_arch_overrides(args.arch)
    checkpoints = iter_cifar10_linf_checkpoints(args.models_dir)
    if args.limit_models is not None:
        checkpoints = checkpoints[: args.limit_models]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for checkpoint in checkpoints:
            started_at = time.perf_counter()
            arch = arch_overrides.get(checkpoint.name) or arch_overrides.get(checkpoint.stem)
            arch = arch or infer_cifar10_linf_arch(checkpoint)

            row = {
                "checkpoint": str(checkpoint),
                "arch": arch,
                "status": "ok",
                "output_shape": "",
                "load_target": "",
                "load_variant": "",
                "missing_keys": "",
                "unexpected_keys": "",
                "state_dict_keys": "",
                "device": str(device),
                "seconds": "",
                "error": "",
            }

            try:
                loaded = load_cifar10_linf_model(checkpoint, device=device, arch=arch)
                row["load_target"] = loaded.load_target
                row["load_variant"] = loaded.load_variant
                row["missing_keys"] = ";".join(loaded.missing_keys)
                row["unexpected_keys"] = ";".join(loaded.unexpected_keys)
                row["state_dict_keys"] = loaded.state_dict_keys

                if not args.no_forward:
                    x = torch.rand(args.forward_batch_size, 3, 32, 32, device=device)
                    with torch.no_grad():
                        y = loaded.model(x)
                    row["output_shape"] = "x".join(str(dim) for dim in y.shape)
                    if tuple(y.shape) != (args.forward_batch_size, 10):
                        raise RuntimeError(f"Unexpected forward output shape: {tuple(y.shape)}")

            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
            finally:
                row["seconds"] = f"{time.perf_counter() - started_at:.3f}"
                writer.writerow(row)
                csv_file.flush()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"Load-check results saved to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that CIFAR-10 Linf model checkpoints load correctly.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory containing .pt checkpoints.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ufbrp/results-cifar10/linf_cifar10_model_load_check.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--limit-models", type=int, default=None, help="Optional smoke-test checkpoint limit.")
    parser.add_argument("--forward-batch-size", type=int, default=2, help="Batch size for the dummy forward pass.")
    parser.add_argument("--no-forward", action="store_true", help="Only load weights; skip dummy forward pass.")
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
