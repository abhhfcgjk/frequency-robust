import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

# from torchvision import models
import torchmetrics
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List, Any
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from ufbrp.attacks.autoattack import AutoAttack
from ufbrp.attacks.base import Attacker
from ufbrp.attacks.fgsm import FGSM
from ufbrp.attacks.pgd import PGD

from ufbrp.utils import load_config
from ufbrp.models import (
    ResNet50,
    ResNet50Blur,
    LipReg,
    LipReg_aa,
    BlurPool,
    create_model
)

Section('model', 'model details').params(
    arch=Param(str, default='resnet50'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    config=Param(str, 'path to .yaml config file', default='ufbrp/presets/imagenet/resnet50.yaml'),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1),
    max_batches=Param(int, 'limit validation to N batches (-1 for full set)', default=-1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)


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
print("torch.cuda.device_count()", ch.cuda.device_count())
from numba import get_num_threads
print("NUMBA THREADS:", get_num_threads())


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CROP_RATIO = 224/256


from collections import OrderedDict

def _extract_state_dict(ckpt):
    # common patterns
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt  # already a state_dict

def _strip_prefix(state_dict, prefix="module."):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return OrderedDict((k[len(prefix):] if k.startswith(prefix) else k, v)
                       for k, v in state_dict.items())


def ddp_gather_1d(t: ch.Tensor, gpu: int) -> ch.Tensor:
    """
    Gather 1D tensor from all ranks (can have different lengths) and concat on rank0.
    Works with NCCL (requires CUDA tensors).
    Returns:
      - rank0: concatenated CPU tensor
      - other ranks: empty CPU tensor
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t.detach().cpu()

    t = t.detach()
    if not t.is_cuda:
        t = t.to(gpu)

    ws = dist.get_world_size()
    rank = dist.get_rank()

    local_n = ch.tensor([t.numel()], device=t.device, dtype=ch.long)
    sizes = [ch.zeros_like(local_n) for _ in range(ws)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    if t.numel() < max_n:
        pad = ch.empty(max_n, device=t.device, dtype=t.dtype)
        pad[: t.numel()] = t
        pad[t.numel() :] = 0
        t = pad

    gathered = [ch.empty(max_n, device=t.device, dtype=t.dtype) for _ in range(ws)]
    dist.all_gather(gathered, t)

    if rank != 0:
        return ch.empty(0, dtype=t.dtype, device="cpu")

    out = ch.cat([g[:n] for g, n in zip(gathered, sizes)], dim=0)
    return out.cpu()


@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


class ImageNetTrainer:
    @param('data.config')
    @param('training.distributed')
    def __init__(self, gpu, config, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.config = load_config(config)

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler(model_path=self.config["model_path"])
        # self.create_optimizer()
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.attacker = self._init_attack(self.config["attack"]["train"])
        self.initialize_logger()

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

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            # NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    def eval(self):
        results = {}

        for attack_args in self.config["attack"]["test"]:
            attack = self._init_attack(attack_args, "test")
            stats, predpack = self.val_loop(attack=attack)

            if dist.is_available() and dist.is_initialized():
                dist.barrier(device_ids=[self.gpu])

            full_pred = ddp_gather_1d(predpack["pred"].to(ch.long), self.gpu)
            full_gt = ddp_gather_1d(predpack["gt"].to(ch.long), self.gpu)

            if self.gpu == 0:
                eps = attack_args["params"]["eps"]
                results[f"pred_eps={eps}"] = full_pred.numpy().tolist()
                print(f"Accuracy for eps={eps} (rank-local meter): {stats['top_1']}")
                global_top1 = (full_pred == full_gt).float().mean().item()
                print(f"Accuracy for eps={eps} (global): {global_top1:.6f}")


        if self.gpu == 0:
            df = pd.DataFrame(results)
            output_file = self.log_folder / f"{self.config['train']['model']}_all_attacks.csv"
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")



    @param('training.distributed')
    def create_model_and_scaler(self, distributed, model_path=None):
        scaler = GradScaler()
        self.AntiAliasing = False
        self.learnable = False
        if self.config["train"]["model"] in ("resnet50", "resnet50-adv", "resnet50-adv-reg"):
            model = ResNet50(num_classes=1000)
        elif self.config["train"]["model"] in ("resnet50_aa", "resnet50_aa-adv"):
            self.AntiAliasing = True
            self.learnable = True
            filt_size = self.config["train"].get("filt_size", 3)
            model = ResNet50Blur(num_classes=1000, filter_size=filt_size, learnable=self.learnable)
        elif self.config["train"]["model"] in ("lipreg", "lipreg-adv"):
            model = LipReg(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=1000,
                jacobian_delta=self.config["train"]["jacobian_delta"],
                k=self.config["train"]["k"],
            )
        elif self.config["train"]["model"] in ("lipreg_aa", "lipreg_aa-adv"):
            self.AntiAliasing = True
            self.learnable = True
            filt_size = self.config["train"].get("filt_size", 3)
            model = LipReg_aa(
                wavelet_level=self.config["train"]["wavelet_level"],
                wavelet_method=self.config["train"]["wavelet_method"],
                num_classes=1000,
                jacobian_delta=self.config["train"]["jacobian_delta"],
                k=self.config["train"]["k"],
                filter_size=filt_size,
                learnable=self.learnable,
            )

        # checkpoint = ch.load(model_path)
        # model.load_state_dict(checkpoint)
        ckpt = ch.load(model_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        state_dict = _strip_prefix(state_dict, "module.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        model = create_model(model, 
                             mean=IMAGENET_MEAN, 
                             std=IMAGENET_STD)
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        else:
            # Attack needs gradients w.r.t. input, not model weights.
            for p in model.parameters():
                p.requires_grad_(False)

        return model, scaler

    def lpf_smoothness_loss(self, module, alpha=1e-4, beta=5e-5, eps = 1e-8):
        a = module.get_kernel()
        a = a / (a.sum() + eps)
        if a.numel() >= 3:
            d2 = a[:-2] - 2 * a[1:-1] + a[2:]
            smooth = (d2 ** 2).mean()
        else:
            smooth = ch.zeros((), device=a.device)
        k = a.numel()
        idx = ch.arange(k, device=a.device) - (k - 1) / 2
        center = (a * idx.abs()).mean()
        return alpha * smooth + beta * center
  
    @param('validation.lr_tta')
    @param('validation.max_batches')
    def val_loop(self, lr_tta, max_batches, attack=None):
        model = self.model
        model.eval()

        preds_chunks = []
        gt_chunks = []

        for batch_idx, (images, target) in enumerate(tqdm(self.val_loader)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(dtype=ch.float32).div_(255.0)
            target = target.long()

            if attack is not None:
                with ch.enable_grad():
                    # with autocast():
                    images = attack.run(images, target).detach()

            with ch.inference_mode():
                # with autocast():
                out = model(images)
                if lr_tta:
                    out = out + model(ch.flip(images, dims=[3]))

            # meters (these are per-rank unless you explicitly sync them)
            for k in ['top_1', 'top_5']:
                self.val_meters[k](out, target)
            self.val_meters['loss'](self.loss(out, target))

            preds_chunks.append(out.argmax(dim=1).detach().cpu())
            gt_chunks.append(target.detach().cpu())

        # local (per-rank) vectors
        local_pred = ch.cat(preds_chunks, dim=0)
        local_gt = ch.cat(gt_chunks, dim=0)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        for m in self.val_meters.values():
            m.reset()

        return stats, {"pred": local_pred, "gt": local_gt}


    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    def exec(cls, gpu, distributed):
        trainer = cls(gpu=gpu)
        trainer.eval()

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
