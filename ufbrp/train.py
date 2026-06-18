import os
import random
import shutil
import time
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ufbrp.attacks.autoattack import AutoAttack
from ufbrp.attacks.base import Attacker
from ufbrp.attacks.fgsm import FGSM
from ufbrp.attacks.pgd import PGD
from ufbrp.datasets import get_data_loader
from ufbrp.log import dump_config
from ufbrp.losses.nmse import NMSERegularizationLoss
from ufbrp.losses.trades import TRADESLoss
from ufbrp.metrics import MetricPerformance, dump_scalar_metrics
from ufbrp.models import (
    DBFTT,
    AdvDBFTT,
    BlurPool,
    Consistency,
    ConsistencyDBFTT,
    LipReg,
    LipReg_aa,
    ResNet50,
    ResNet50Blur,
    RevNet,
    RevNetDBFTT,
    RevNetSmall,
    LipReg_aa_WideResNet,
    WideResNet_L,
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


class AdversarialTrainer:
    def __init__(self, gpu: int, config_path: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.num_classes = self.config["data"]["num_classes"]
        self.eval_only = self.config["eval_only"]
        self.base_logs_dir = Path(self.config["logs_dir"])
        self.base_logs_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_name = self.config["data"]["dataset"]
        self.use_labels = False
        self.adv_reg = self.config["attack"]["train"].get("regularization")

        self.learnable = False
        self.AntiAliasing = False

        self.gpu = gpu
        print(f"I am gpu #{self.gpu}")
        if self.config["train"]["model"] in ("resnet50", "resnet50-adv", "resnet50-adv-reg"):
            model = ResNet50(num_classes=self.num_classes)
        elif self.config["train"]["model"] in ("resnet50_aa", "resnet50_aa-adv"):
            self.AntiAliasing = True
            self.learnable = True
            filt_size = self.config["train"].get("filt_size", 3)
            model = ResNet50Blur(num_classes=self.num_classes, filter_size=filt_size, learnable=self.learnable)
        elif self.config["train"]["model"] in ("wideresnet", "wideresnet-adv"):
            model = WideResNet_L(num_classes=self.num_classes)
        elif self.config["train"]["model"] in ("wideresnet_lipreg_aa", "wideresnet_lipreg_aa-adv"):
            self.AntiAliasing = True
            self.learnable = True
            filt_size = self.config["train"].get("filt_size", 3)
            model = LipReg_aa_WideResNet(num_classes=self.num_classes, filter_size=filt_size, learnable=self.learnable)
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
        elif self.config["train"]["model"] == "lipreg_aa":
            self.AntiAliasing = True
            self.learnable = True
            filt_size = self.config["train"].get("filt_size", 3)
            model = LipReg_aa(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=self.num_classes,
                jacobian_delta=self.config["train"]["jacobian_delta"],
                k=self.config["train"]["k"],
                filter_size=filt_size,
                learnable=self.learnable,
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
        self.model = create_model(
            model, mean=NORMALIZATION[self.dataset_name]["mean"], std=NORMALIZATION[self.dataset_name]["std"]
        )
        print(self.model)
        self.model.to(self.gpu)
        self._init_logger()

    def _init_logger(self):
        # if self.gpu == 0:
        self.log_dir = self.config.get("checkpoints_path")
        if self.log_dir is None:
            self.log_dir = self.base_logs_dir / f"{datetime.today()}"
        else:
            self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        print(f"[Log dir] {self.log_dir}")
        # self.log_dir = self.base_logs_dir / f"{datetime.today()}"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        if self.eval_only:
            self.results_csv = Path(self.config["results_path"])
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"=> Logging in {self.log_dir}")
        dump_config(self.config, self.writer)
        shutil.copy(self.config_path, self.log_dir / "presets.yaml")
        self.start_training_time = time.time()

    def train(self) -> None:
        self._prepare_for_training()
        self._train_loop()

    def _prepare_for_training(self) -> None:
        self.current_epoch = 0
        self.end_epoch = self.config["train"]["epochs"]
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = get_data_loader(
            dataset_name=self.config["data"]["dataset"],
            directory=self.config["data"]["directory"],
            batch_size=self.config["train"]["batch_size"],
            num_workers=self.config["train"]["num_workers"],
            val_split_ratio=0.9,
        )

        self._init_optimizer()
        self.scaler = GradScaler()

        self._init_lr_scheduler()
        loss_name = self.config["train"].get("loss", {"name": "crossentropy"})
        self.loss_name = loss_name["name"]
        if self.loss_name == "crossentropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.loss_name == "trades":
            self.ce_loss = nn.CrossEntropyLoss()
            self.loss = TRADESLoss(self.ce_loss, beta=self.config["train"]["loss"]["beta"])
        else:
            raise NotImplementedError

        if self.adv_reg == "nmse":
            self.reg = NMSERegularizationLoss(p_clean=0.5)

        # if self.loss_name == "trades":
        #     self.attacker = None
        # else:
        self.attacker = self._init_attack(self.config["attack"]["train"])

        # if self.gpu == 0:
        self.val_criterion = self.config["train"]["val_criterion"]
        self.metric_computer = MetricPerformance(self.num_classes)
        self.best_val_criterion, self.best_epoch = -100, -1
        ckpt_path = self.log_dir / "last.pt"
        if ckpt_path.exists():
            map_location = {"cuda:0": f"cuda:{self.gpu}"}
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.current_epoch = ckpt["epoch"] + 1
            self.best_val_criterion = ckpt.get("best_acc", -100)
            print(f"[Current epoch]: {self.current_epoch}. Model and optimizer loaded from {self.log_dir}")
            print(f"[Accuracy]: {self.best_val_criterion}")

    def _init_optimizer(self) -> None:
        if self.AntiAliasing:
            aa_lr = self.config["optimizer"].get("aa_lr", None)
            lr = self.config["optimizer"].get("lr", None)
            params_main, params_lpf = [], []
            if aa_lr is not None:
                for m in self.model.model.modules():
                    if self.learnable:
                        if isinstance(m, BlurPool):
                            for p in m.parameters():
                                if p.requires_grad:
                                    params_lpf.append(p)

                lpf_set = set(params_lpf)
                for p in self.model.model.parameters():
                    if p.requires_grad and p not in lpf_set:
                        params_main.append(p)

                param_groups = []
                if params_main:
                    param_groups.append({"params": params_main, "lr": lr})
                if params_lpf:
                    param_groups.append({"params": params_lpf, "lr": float(aa_lr)})
            params = param_groups
        else:
            params = self.model.model.parameters()

        if self.config["optimizer"]["type"] == "sgd":
            self.optimizer = SGD(
                params,
                lr=self.config["optimizer"]["lr"],
                momentum=self.config["optimizer"]["momentum"],
                weight_decay=self.config["optimizer"]["weight_decay"],
            )
        elif self.config["optimizer"]["type"] == "adam":
            self.optimizer = Adam(
                params,
                lr=self.config["optimizer"]["lr"],
                weight_decay=self.config["optimizer"].get("weight_decay", 0.0),
            )
        elif self.config["optimizer"]["type"] == "adamw":
            self.optimizer = Adam(
                params,
                lr=self.config["optimizer"]["lr"],
                weight_decay=self.config["optimizer"]["weight_decay"],
            )
        else:
            raise NotImplementedError

    def _init_lr_scheduler(self) -> None:
        if self.config["lr_scheduler"]["type"] == "cosine":
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                eta_min=self.config["lr_scheduler"]["eta_min"],
                T_max=self.config["lr_scheduler"]["T_max"],
            )
        elif self.config["lr_scheduler"]["type"] == "step":
            self.lr_scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["lr_scheduler"]["step_size"],
                gamma=self.config["lr_scheduler"]["gamma"],
            )
        else:
            raise NotImplementedError

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
        elif self.loss_name == "trades":
            loss = nn.KLDivLoss(reduction="batchmean")

            def loss_computer(y, target):
                # print(target, y)
                target_one = F.one_hot(target.long(), num_classes=self.num_classes).float()
                y_one = F.softmax(y, dim=1)
                # print(y_one)
                return loss(F.log_softmax(target_one, dim=1), y_one)
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

    def _attack_step(self, input, label, use_clear=False):
        adv_input = self.attacker.run(input, label)
        if use_clear:
            out = torch.vstack((input, adv_input)).detach()
            return out.requires_grad_(True), torch.hstack((label, label))
        else:
            return adv_input.detach().requires_grad_(True), label

    def _trades_attack_step(self, input):
        attack_params = self.config["attack"]["train"]["params"]
        eps = attack_params["eps"] / 255
        step_size = attack_params.get("alpha", attack_params["eps"]) / 255
        num_steps = attack_params.get("iters", 10)
        init_std = attack_params.get("init_std", 0.001)
        lower = input.detach() - eps
        upper = input.detach() + eps

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                clean_probs = F.softmax(self.model(input), dim=1)
                attacked = input.detach() + init_std * torch.randn_like(input)
                attacked = torch.min(torch.max(attacked, lower), upper)
                attacked = torch.clamp(attacked, 0.0, 1.0)

            for _ in range(num_steps):
                attacked.requires_grad_()
                attacked_logit = self.model(attacked)
                loss_kl = F.kl_div(F.log_softmax(attacked_logit, dim=1), clean_probs, reduction="batchmean")
                grad = torch.autograd.grad(loss_kl, [attacked])[0]

                attacked = attacked.detach() + step_size * torch.sign(grad.detach())
                attacked = torch.min(torch.max(attacked, lower), upper)
                attacked = torch.clamp(attacked, 0.0, 1.0)

            return attacked.detach().requires_grad_(True)
        finally:
            if was_training:
                self.model.train()

    def lpf_smoothness_loss(self, module, alpha=1e-4, beta=5e-5, eps=1e-8):
        a = module.get_kernel()
        a = a / (a.sum() + eps)
        if a.numel() >= 3:
            d2 = a[:-2] - 2 * a[1:-1] + a[2:]
            smooth = (d2**2).mean()
        else:
            smooth = torch.zeros((), device=a.device)
        k = a.numel()
        idx = torch.arange(k, device=a.device) - (k - 1) / 2
        center = (a * idx.abs()).mean()
        return alpha * smooth + beta * center

    def _train_loop(self) -> None:
        train_data_len = len(self.train_loader)
        while self.current_epoch < self.end_epoch:
            self.model.train()

            done_steps = self.current_epoch * train_data_len
            batch_start_time = time.time()

            for step, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                metrics = self._train_step(data, step, batch_start_time)
                # if self.gpu == 0:
                # dump_scalar_metrics(
                #     metrics, self.writer, "train", global_step=done_steps + step, dataset=self.dataset_name
                # )
                batch_start_time = time.time()
            self.current_epoch += 1

            self.lr_scheduler.step()
            self.metric_computer.reset()
            self.model.eval()
            val_metrics = {}
            for step, data in enumerate(self.val_loader):
                self._val_step(data)

            val_metrics["Accuracy"] = self.metric_computer.accuracy
            val_metrics["Precision"] = self.metric_computer.precision
            val_metrics["Recall"] = self.metric_computer.recall
            val_metrics["F1Score"] = self.metric_computer.f1score
            val_criterion = val_metrics["Accuracy"]
            # dump_scalar_metrics(
            #     val_metrics, self.writer, "val", global_step=done_steps + step, dataset=self.dataset_name
            # )
            last_ckpt = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.current_epoch,
                "best_acc": self.best_val_criterion,
                "accuracy": val_criterion,
            }
            torch.save(last_ckpt, self.log_dir / "last.pt")

            if val_criterion >= self.best_val_criterion:
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.current_epoch,
                    "accuracy": val_criterion,
                }
                torch.save(checkpoint, self.log_dir / "best_model.pt")

                self.best_val_criterion = val_criterion
                self.best_epoch = self.current_epoch
                print(
                    f"Save current best model @best_val_criterion ({self.val_criterion}):\
                            {self.best_val_criterion:.3f} @epoch: {self.best_epoch}"
                )
            else:
                print(
                    f"Model is not updated @val_criterion ({self.val_criterion}):\
                            {val_criterion:.3f} @epoch: {self.current_epoch}"
                )

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
        }
        self.metric_computer = MetricPerformance(self.num_classes)
        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.train_loader):
            self._val_step(data)
        torch.save(checkpoint, self.log_dir / "final_model.pt")
        checkpoint = torch.load(self.log_dir / "best_model.pt")
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.train_loader):
            self._val_step(data)
        torch.save(checkpoint, self.log_dir / "best_model.pt")

    def _train_step(self, data, step: int, start_time: float) -> Dict[str, float]:
        metrics = {}
        inputs, label = data[0].cuda(self.gpu, non_blocking=True), data[1].cuda(self.gpu, non_blocking=True)
        inputs.requires_grad_(True)
        metrics["data_time"] = time.time() - start_time
        self.optimizer.zero_grad()

        if self.attacker is not None:
            attacked, label = self._attack_step(inputs, label)
            print(self.loss_name)
            if self.loss_name != "trades":
                inputs = attacked

        # if self.loss_name == "trades":
        #     attacked = self._trades_attack_step(inputs)
        # elif self.attacker is not None:
        #     attacked, label = self._attack_step(inputs, label)
        #     inputs = attacked

        if self.use_labels:
            self.model.model.label = label
            model_out, label = self.model(inputs)
        else:
            model_out = self.model(inputs)

        if self.loss_name == "trades":
            attacked_logit = self.model(attacked)
            loss = self.loss(model_out, attacked_logit, label)
            # print(loss)
        else:
            loss = self.loss(model_out, label)
        loss += self.model.model.penalty

        aa_reg_alpha = self.config["train"].get("aa_reg_alpha", 0.0)
        aa_reg_beta_alpha = self.config["train"].get("aa_reg_beta_alpha", 0.5)
        if (self.AntiAliasing) and (aa_reg_alpha > 0.0):
            reg = 0.0
            for m in self.model.model.modules():
                if self.learnable and isinstance(m, BlurPool):
                    _ = m.get_kernel()
                    reg = reg + self.lpf_smoothness_loss(m, alpha=aa_reg_alpha, beta=aa_reg_alpha * aa_reg_beta_alpha)
            loss = loss + reg

        # if self.adv_reg:
        #     bs = inputs.shape[0]
        #     loss += self.reg(model_out[: bs // 2, ...], model_out[bs // 2 :, ...])
        # print(loss)
        loss.backward()
        if hasattr(self.model.model, "post_penalty"):
            self.model.model.post_penalty(inputs)
        self.optimizer.step()

        metrics["total_loss"] = loss.cpu().detach().numpy()
        metrics["total_time"] = time.time() - start_time
        return metrics

    @torch.no_grad()
    def _val_step(self, data, attack: Attacker = None) -> None:
        inputs, label = data[0].cuda(self.gpu, non_blocking=True), data[1].cuda(self.gpu, non_blocking=True)
        if self.use_labels:
            self.model.model.label = None
            model_out, _ = self.model(inputs)
        else:
            model_out = self.model(inputs)
        logit = torch.argmax(model_out, dim=1)
        self.metric_computer.update(logit, label)

    def is_ddp_model(self, state_dict):
        for key in state_dict:
            if key.startswith("module."):
                return True
        return False

    def eval(self) -> None:
        checkpoint = torch.load(self.log_dir / "best_model.pt")
        self.model.load_state_dict(checkpoint["model"])
        self.metric_computer = MetricPerformance(self.num_classes)
        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.test_loader):
            self._val_step(data)
        metrics = {
            "Accuracy": self.metric_computer.accuracy,
            "Precision": self.metric_computer.precision,
            "Recall": self.metric_computer.recall,
            "F1Score": self.metric_computer.f1score,
        }
        for metric, metric_val in metrics.items():
            checkpoint[metric] = metric_val

        dump_scalar_metrics(metrics, self.writer, "test", dataset=self.dataset_name)
        print(f'Test Accuracy: {checkpoint["Accuracy"]}')
        torch.save(checkpoint, self.log_dir / "best_model.pt")

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
    def exec(cls, gpu, config_path):
        cls.set_global_seed()
        trainer = cls(gpu=gpu, config_path=config_path)
        if trainer.eval_only:
            trainer.test()
        else:
            trainer.train()
            print(str(datetime.now())[:-4])
            if gpu == 0:
                trainer.eval()

    def __del__(self):
        self.writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="ufbrp/presets/resnet50.yaml", help="The path to the architectural configuration."
    )
    args = parser.parse_args()
    AdversarialTrainer.exec(0, args.config)
