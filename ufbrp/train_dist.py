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
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm

from ufbrp.attacks.autoattack import AutoAttack
from ufbrp.attacks.base import Attacker
from ufbrp.attacks.fgsm import FGSM
from ufbrp.attacks.pgd import PGD

# from ufbrp.datasets.dali.imagenet import get_data_loader, get_train_loader_length
# from ufbrp.datasets.imagenet_dist import get_data_loader
from ufbrp.datasets.dist import get_data_loader
from ufbrp.log import dump_config
from ufbrp.metrics_dist import dump_scalar_metrics
from ufbrp.models import (
    DBFTT,
    AdvDBFTT,
    Consistency,
    ConsistencyDBFTT,
    LipReg,
    ResNet50,
    RevNet,
    RevNetDBFTT,
    RevNetSmall,
    create_model,
)
from ufbrp.utils import load_config

print("=== DIST DEBUG ===")
print(
    "RANK",
    os.environ.get("RANK"),
    "LOCAL_RANK",
    os.environ.get("LOCAL_RANK"),
    "WORLD_SIZE",
    os.environ.get("WORLD_SIZE"),
)
print("SLURM_NTASKS", os.environ.get("SLURM_NTASKS"), "SLURM_STEP_GPUS", os.environ.get("SLURM_STEP_GPUS"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count()", torch.cuda.device_count())

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["KMP_WARNINGS"] = "off"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
LOCAL_RANK = int(os.environ["LOCAL_RANK"])

# def setup(rank, world_size):
#     dist.init_process_group(
#         backend='nccl',  # fast for NVIDIA GPUs
#         init_method='env://',  # can also use tcp://IP:PORT
#         world_size=world_size,
#         rank=rank
#     )
#     torch.cuda.set_device(rank)


def setup():
    """
    Initialize process group using env://. Assumes torchrun / srun has set:
      - RANK, WORLD_SIZE, LOCAL_RANK
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # init process group using env:// (torchrun provides the right env variables)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Choose a local device index that is valid for this process
    visible_cuda_count = torch.cuda.device_count()
    if visible_cuda_count == 0:
        raise RuntimeError("No CUDA devices visible to this process (torch.cuda.device_count() == 0)")

    device_idx = local_rank
    if device_idx >= visible_cuda_count:
        # map into available devices (fail-safe)
        device_idx = local_rank % visible_cuda_count
        print(
            f"Warning: LOCAL_RANK {local_rank} >= visible device count {visible_cuda_count}, "
            f"mapping to device {device_idx}"
        )

    torch.cuda.set_device(device_idx)
    return rank, world_size, device_idx


def cleanup():
    dist.destroy_process_group()


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
        # self.config = config_path
        self.num_classes = self.config["data"]["num_classes"]
        self.eval_only = self.config["eval_only"]
        self.base_logs_dir = Path(self.config["logs_dir"])
        self.base_logs_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_name = self.config["data"]["dataset"]
        self.use_labels = False
        self.adv_reg = self.config["attack"]["train"].get("regularization")

        self.gpu = gpu
        self.world_size = dist.get_world_size()

        print(f"I am gpu #{self.gpu}")
        if self.config["train"]["model"] in ("resnet50", "resnet50-adv", "resnet50-adv-reg"):
            model = ResNet50(num_classes=self.num_classes)
        elif self.config["train"]["model"] == "dbftt":
            model = DBFTT(
                wavelet_level=4, num_classes=self.num_classes, jacobian_delta=self.config["train"]["jacobian_delta"]
            )
        elif self.config["train"]["model"] == "lipreg":
            model = LipReg(
                wavelet_level=4, num_classes=self.num_classes, jacobian_delta=self.config["train"]["jacobian_delta"]
            )
        elif self.config["train"]["model"] == "consistency":
            model = Consistency(wavelet_level=4, num_classes=self.num_classes)
        elif self.config["train"]["model"] == "revnet":
            model = RevNet(num_classes=self.num_classes, in_shape=SHAPES[self.dataset_name])
        elif self.config["train"]["model"] == "revnet-small":
            model = RevNetSmall(num_classes=self.num_classes, in_shape=SHAPES[self.dataset_name])
        elif self.config["train"]["model"] == "revnet-dbftt":
            model = RevNetDBFTT(wavelet_level=2, num_classes=self.num_classes, in_shape=SHAPES[self.dataset_name])
        elif self.config["train"]["model"] == "resnet50-adv-dbftt":
            model = DBFTT(wavelet_level=4, num_classes=self.num_classes)
        elif self.config["train"]["model"] == "consistency-dbftt":
            model = ConsistencyDBFTT(wavelet_level=4, num_classes=self.num_classes)
        elif self.config["train"]["model"] == "adv-dbftt":
            model = AdvDBFTT(wavelet_level=4, num_classes=self.num_classes)
            self.use_labels = True
        elif self.config["train"]["model"] == "lipreg-adv":
            model = LipReg(
                wavelet_level=4, num_classes=self.num_classes, jacobian_delta=self.config["train"]["jacobian_delta"]
            )
        elif self.config["train"]["model"] == "dbftt-adv":
            model = DBFTT(
                wavelet_level=4, num_classes=self.num_classes, jacobian_delta=self.config["train"]["jacobian_delta"]
            )
        elif self.config["train"]["model"] == "consistency-dbftt-adv":
            model = ConsistencyDBFTT(wavelet_level=4, num_classes=self.num_classes)
        else:
            raise NotImplementedError(f"Model {self.config['train']['model']} is not supported")
        model = create_model(
            model, mean=NORMALIZATION[self.dataset_name]["mean"], std=NORMALIZATION[self.dataset_name]["std"]
        )
        model.to(self.gpu)
        self.model = DDP(model, device_ids=[self.gpu])
        self._init_logger()

    def _init_logger(self):
        # if self.gpu == 0:
        self.log_dir = self.config.get("checkpoints_path")
        if dist.get_rank() == 0:
            if self.log_dir is None:
                self.log_dir = self.base_logs_dir / f"{datetime.today()}"
            else:
                self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)
            print(f"[Log dir] {self.log_dir}")
            if self.eval_only:
                self.results_csv = Path(self.config["results_path"])
            # self.writer = SummaryWriter(log_dir=self.log_dir)
            # print(f"=> Logging in {self.log_dir}")
            # dump_config(self.config, self.writer)
            # shutil.copy(self.config_path, self.log_dir / "presets.yaml")
            # if dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            dump_config(self.config, self.writer)
            shutil.copy(self.config_path, self.log_dir / "presets.yaml")
        else:
            self.writer = None
            self.log_dir = Path("") if self.log_dir is None else Path(self.log_dir)

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
            world_size=self.world_size,
            rank=self.gpu,
        )

        self._init_optimizer()
        self.scaler = GradScaler()

        self._init_lr_scheduler()
        self.loss = nn.CrossEntropyLoss()

        self.attacker = self._init_attack(self.config["attack"]["train"])

        self.val_criterion = self.config["train"]["val_criterion"]
        self.metric_computer = Accuracy(task="multiclass", num_classes=self.num_classes, dist_sync_on_step=True).to(
            self.gpu
        )
        self.best_val_criterion, self.best_epoch = -100, -1

        # if self.gpu == 0:
        if (self.log_dir / "best_model.pt").exists():
            map_location = {"cuda:0": f"cuda:{self.gpu}"}
            ckpt = torch.load(self.log_dir / "best_model.pt", map_location=map_location, weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.current_epoch = ckpt["epoch"] + 1
            self.best_val_criterion = ckpt["accuracy"]
            print(f"[Current epoch]: {self.current_epoch}. Model and optimizer loaded from {self.log_dir}")
            print(f"[Accuracy]: {self.best_val_criterion}")
        # dist.barrier()

    def _init_optimizer(self) -> None:
        if self.config["optimizer"]["type"] == "sgd":
            # self.optimizer = SGD(
            #     self.model.module.parameters(),
            #     lr=self.config["optimizer"]["lr"],
            #     momentum=self.config["optimizer"]["momentum"],
            #     weight_decay=self.config["optimizer"]["weight_decay"],
            # )
            pass
        elif self.config["optimizer"]["type"] == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config["optimizer"]["lr"])
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

    def _attack_step(self, input, label):
        adv_input = self.attacker.run(input, label)
        out = torch.vstack((input, adv_input)).detach()
        return out.requires_grad_(True), torch.hstack((label, label))

    def _train_loop(self) -> None:
        train_data_len = len(self.train_loader)
        while self.current_epoch < self.end_epoch:
            self.metric_computer.reset()
            self.train_loader.sampler.set_epoch(self.current_epoch)
            self.model.train()

            done_steps = self.current_epoch * train_data_len
            batch_start_time = time.time()

            for step, batch in tqdm(enumerate(self.train_loader), total=train_data_len):
                metrics = self._train_step(batch, step, batch_start_time)
                # if self.gpu == 0:
                #     dump_scalar_metrics(
                #         metrics, self.writer, "train", global_step=done_steps + step, dataset=self.dataset_name
                #     )
                batch_start_time = time.time()
            self.current_epoch += 1
            self.lr_scheduler.step()

            self.metric_computer.reset()
            self.model.eval()
            val_metrics = {}
            for step, data in enumerate(self.val_loader):
                self._val_step(data)

            compute = self.metric_computer.compute()
            val_metrics["Accuracy"] = compute
            val_criterion = val_metrics["Accuracy"]
            if self.gpu == 0:
                dump_scalar_metrics(
                    val_metrics, self.writer, "val", global_step=done_steps + step, dataset=self.dataset_name
                )
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
            dist.barrier()

    def _train_step(self, data, step: int, start_time: float) -> Dict[str, float]:
        metrics = {}
        inputs, label = data[0].cuda(self.gpu, non_blocking=True), data[1].cuda(self.gpu, non_blocking=True)
        inputs.requires_grad_(True)
        metrics["data_time"] = time.time() - start_time
        self.optimizer.zero_grad()
        if self.attacker is not None:
            inputs, label = self._attack_step(inputs, label)

        if self.use_labels:
            self.model.module.model.label = label
            model_out, label = self.model(inputs)
        else:
            model_out = self.model(inputs)
        loss = self.loss(model_out, label) + self.model.module.model.penalty
        if self.adv_reg:
            bs = inputs.shape[0]
            loss += self.reg(model_out[: bs // 2, ...], model_out[bs // 2 :, ...])
        loss.backward()
        if hasattr(self.model.module.model, "post_penalty"):
            self.model.module.model.post_penalty(inputs)
        self.optimizer.step()

        metrics["total_loss"] = loss.cpu().detach().numpy()
        metrics["total_time"] = time.time() - start_time
        return metrics

    def _val_step(self, data, attack: Attacker = None) -> None:
        inputs = data[0].to(self.gpu, non_blocking=True)
        label = data[1].to(self.gpu, non_blocking=True)
        model_out = self.model(inputs)
        logit = torch.argmax(model_out, dim=1)
        self.metric_computer.update(logit, label)

    def eval(self) -> None:
        checkpoint = torch.load(self.log_dir / "best_model.pt")
        self.model.load_state_dict(checkpoint["model"])
        self.metric_computer = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.gpu)
        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.test_loader):
            self._val_step(data)
        compute = self.metric_computer.compute()
        metrics = {
            "Accuracy": compute,
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
        # n_gpus = torch.cuda.device_count()
        # print(f"World size: {n_gpus}")
        # setup(gpu, world_size=n_gpus)
        setup()
        trainer = cls(gpu=gpu, config_path=config_path)
        if trainer.eval_only:
            trainer.test()
        else:
            trainer.train()
            print(str(datetime.now())[:-4])
            cleanup()

    def __del__(self):
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="ufbrp/presets/imagenet/resnet50.yaml",
        help="The path to the architectural configuration.",
    )
    args = parser.parse_args()
    AdversarialTrainer.exec(int(os.environ.get("LOCAL_RANK", "0")), args.config)
