import os
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from ufbrp.attacks.autoattack import AutoAttack
from ufbrp.attacks.base import Attacker
from ufbrp.attacks.fgsm import FGSM
from ufbrp.attacks.pgd import PGD
from ufbrp.datasets import get_data_loader
from ufbrp.metrics import MetricPerformance
from ufbrp.models import (
    DBFTT,
    AdvDBFTT,
    Consistency,
    ConsistencyDBFTT,
    LipReg,
    LipReg_aa,
    ResNet50,
    ResNet50Blur,
    RevNet,
    RevNetDBFTT,
    RevNetSmall,
    create_model,
)
from ufbrp.utils import load_config

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["KMP_WARNINGS"] = "off"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

NORMALIZATION = {
    "CIFAR10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "ImageNet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}

SHAPES = {"CIFAR10": [3, 32, 32], "CIFAR100": [3, 32, 32], "ImageNet": [3, 224, 224]}


class AdversarialEvaluator:
    def __init__(self, gpu: int, config_path: Path, logs_dir: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.num_classes = self.config["data"]["num_classes"]
        self.dataset_name = self.config["data"]["dataset"]
        if logs_dir:
            self.logs_dir = Path(logs_dir)
            self.model_path = self.logs_dir / "best_model.pt"
        else:
            self.logs_dir = Path(self.config["logs_dir"])
            self.logs_dir.mkdir(exist_ok=True, parents=True)
            self.model_path = self.logs_dir / self.config["train"]["model"] / "best_model.pt"

        self.gpu = gpu
        print(f"I am gpu #{self.gpu}")

        if self.config["train"]["model"] in ("resnet50", "resnet50-adv", "resnet50-adv-reg"):
            model = ResNet50(num_classes=self.num_classes)
        elif self.config["train"]["model"] in ("resnet50_aa", "resnet50_aa-adv"):
            filt_size = self.config["train"].get("filt_size", 3)
            model = ResNet50Blur(num_classes=self.num_classes, filter_size=filt_size, learnable=True)
        elif self.config["train"]["model"] in ("dbftt", "dbftt-adv"):
            model = DBFTT(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=self.num_classes,
                jacobian_delta=self.config["train"]["jacobian_delta"],
            )
        elif self.config["train"]["model"] in ("lipreg", "lipreg-adv"):
            model = LipReg(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=self.num_classes,
                jacobian_delta=self.config["train"]["jacobian_delta"],
                k=self.config["train"]["k"],
            )
        elif self.config["train"]["model"] in ("lipreg_aa", "lipreg_aa-adv"):
            filt_size = self.config["train"].get("filt_size", 3)
            model = LipReg_aa(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=self.num_classes,
                jacobian_delta=self.config["train"]["jacobian_delta"],
                k=self.config["train"]["k"],
                filter_size=filt_size,
                learnable=True,
            )
        elif self.config["train"]["model"] in ("consistency-dbftt", "consistency-dbftt-adv"):
            model = ConsistencyDBFTT(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=self.num_classes,
                jacobian_delta=self.config["train"]["jacobian_delta"],
                beta=self.config["train"]["beta"],
            )
        elif self.config["train"]["model"] == "consistency":
            model = Consistency(wavelet_level=self.config["train"]["wavelet_level"], num_classes=self.num_classes)
        elif self.config["train"]["model"] == "revnet":
            model = RevNet(num_classes=self.num_classes, in_shape=SHAPES[self.dataset_name])
        elif self.config["train"]["model"] == "revnet-small":
            model = RevNetSmall(num_classes=self.num_classes, in_shape=SHAPES[self.dataset_name])
        elif self.config["train"]["model"] == "revnet-dbftt":
            model = RevNetDBFTT(
                wavelet_level=self.config["train"]["wavelet_level"],
                num_classes=self.num_classes,
                in_shape=SHAPES[self.dataset_name],
            )
        elif self.config["train"]["model"] == "resnet50-adv-dbftt":
            model = DBFTT(wavelet_level=self.config["train"]["wavelet_level"], num_classes=self.num_classes)
        elif self.config["train"]["model"] == "adv-dbftt":
            model = AdvDBFTT(wavelet_level=self.config["train"]["wavelet_level"], num_classes=self.num_classes)
            self.use_labels = True
        else:
            raise NotImplementedError(f"Model {self.config['train']['model']} is not supported")
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results_csv = self.results_dir
        self.model = create_model(
            model, mean=NORMALIZATION[self.dataset_name]["mean"], std=NORMALIZATION[self.dataset_name]["std"]
        )
        self.model.to(self.gpu)

    def _prepare_for_eval(self) -> None:
        (
            _,
            _,
            self.test_loader,
        ) = get_data_loader(
            dataset_name=self.config["data"]["dataset"],
            directory=self.config["data"]["directory"],
            batch_size=self.config["train"]["batch_size"],
            num_workers=self.config["train"]["num_workers"],
            val_split_ratio=0.9,
        )
        self.metric_computer = MetricPerformance(self.num_classes)

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
            model=self.model,
            loss_computer=loss_computer,
            **attack_config["params"],
        )
        return attacker

    def _val_step(self, data, attack: Attacker = None) -> None:
        inputs, label = (data[0].cuda(self.gpu, non_blocking=True), data[1].cuda(self.gpu, non_blocking=True))
        if attack:
            inputs = attack.run(inputs, label)
        model_out = self.model(inputs)
        logit = torch.argmax(model_out, dim=1)
        self.metric_computer.update(logit, label)

    def _get_labels(self, phase="test"):
        mapper = {"test": self.test_loader}
        loader = mapper[phase]
        labels = []
        for data in loader:
            logit = data[1]
            labels.extend(logit.cpu().detach().numpy())
        return labels

    def is_ddp_model(self, state_dict):
        for key in state_dict:
            if key.startswith("module."):
                return True
        return False

    def eval(self) -> None:
        self._prepare_for_eval()
        checkpoint = torch.load(self.model_path)
        if self.is_ddp_model(checkpoint["model"]):
            checkpoint["model"] = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
        self.model.load_state_dict(checkpoint["model"])
        self.metric_computer = MetricPerformance(self.num_classes)
        results = {}
        att_accuracys = []
        labels = self._get_labels()
        results["label"] = labels
        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.test_loader):
            self._val_step(data)
        orig_accuracy = self.metric_computer.accuracy
        orig_preds = self.metric_computer.preds.copy()
        orig_preds = np.array(orig_preds)
        results["origin_pred"] = orig_preds
        for attack_args in self.config["attack"]["test"]:
            attack = self._init_attack(attack_args, "test")
            self.metric_computer.reset()
            for step, data in tqdm(enumerate(self.test_loader)):
                self._val_step(data, attack)
            att_preds = np.array(self.metric_computer.preds)
            att_accuracys.append(self.metric_computer.accuracy.item())
            results[f'pred_eps={attack_args["params"]["eps"]}'] = att_preds

            print(f'Accuracy for eps={attack_args["params"]["eps"]}: {self.metric_computer.accuracy}')

            output_file = self.results_csv
            output_file /= f"{self.config['train']['model']}_{attack_args['type']}.csv"
            pd.DataFrame.from_dict(results).to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        print(f"Original accuracy: {orig_accuracy}")
        print(f"Attacked accuracys: {att_accuracys}")

    @classmethod
    def set_global_seed(cls, seed=20):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Global seed set to: {seed}")

    @classmethod
    def exec(cls, gpu, config_path, logs_dir):
        cls.set_global_seed()
        trainer = cls(gpu=gpu, config_path=config_path, logs_dir=logs_dir)
        trainer.eval()


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument(
    #     "--config", type=str, default="ufbrp/presets/dbftt.yaml", help="The path to the architectural configuration."
    # )
    parser.add_argument("--logs", type=str, default="logs/resnet50")
    args = parser.parse_args()
    args.config = os.path.join(args.logs, "presets.yaml")
    AdversarialEvaluator.exec(0, args.config, args.logs)
