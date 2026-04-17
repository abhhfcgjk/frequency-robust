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

from ufbrp.attacks.base import Attacker
from ufbrp.attacks.pgd import ImageNetPGD

from ufbrp.utils import load_config
from ufbrp.models import (
    ResNet50,
    ResNet50Blur,
    LipReg,
    LipReg_aa,
    BlurPool
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
    train_dataset=Param(str, '.dat file to use for training', required=True),
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
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

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

        self.initialize_logger()
        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.start_epoch = 0
        self.load_checkpoint_if_exists()
        self.attacker = self._init_attack(self.config["attack"]["train"])
        

    def _init_attack(self, attack_config: dict[str, Any], *args, **kwargs) -> Attacker:
        attack_name = attack_config["type"]
        if attack_name == "none":
            return None

        attackers = {"pgd": ImageNetPGD}
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

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay, label_smoothing):
        assert optimizer == 'sgd'

        model = self.model.module
        params_main, params_lpf = [], []
        if self.AntiAliasing and self.learnable:
            aa_lr = self.config["optimizer"].get("aa_lr", None)

            if aa_lr is not None:
                for m in model.modules():
                    if isinstance(m, BlurPool):
                        params_lpf.extend(p for p in m.parameters() if p.requires_grad)
        lpf_set = set(params_lpf)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p in lpf_set:
                continue
            params_main.append((name, p))

        def split_bn(named_params):
            bn, other = [], []
            for name, p in named_params:
                if 'bn' in name.lower():
                    bn.append(p)
                else:
                    other.append(p)
            return bn, other
        main_bn, main_other = split_bn(params_main)
        lpf_bn, lpf_other = split_bn(
            [(n, p) for n, p in model.named_parameters() if p in lpf_set]
        )
        param_groups = []
        if main_bn:
            param_groups.append({
                "params": main_bn,
                "weight_decay": 0.0
            })
        if main_other:
            param_groups.append({
                "params": main_other,
                "weight_decay": weight_decay
            })
        if params_lpf:
            if lpf_bn:
                param_groups.append({
                    "params": lpf_bn,
                    "weight_decay": 0.0,
                    "lr": float(aa_lr),
                })
            if lpf_other:
                param_groups.append({
                    "params": lpf_other,
                    "weight_decay": weight_decay,
                    "lr": float(aa_lr),
                })

        self.optimizer = ch.optim.SGD(
            param_groups,
            momentum=momentum,
        )
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        # order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        use_os_cache = bool(in_memory)
        if use_os_cache:
            order = OrderOption.RANDOM
        else:
            order = OrderOption.QUASI_RANDOM
        # use_cache = bool(in_memory) and (not distributed or self.gpu == 0)
        # order = OrderOption.RANDOM if bool(in_memory) else OrderOption.QUASI_RANDOM

        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory, # use_cache, # in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

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
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
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

    # @param('training.epochs')
    # @param('logging.log_level')
    # def train(self, epochs, log_level):
    #     start = getattr(self, "start_epoch", 0)
    #     for epoch in range(start, epochs):
    #         res = self.get_resolution(epoch)
    #         self.decoder.output_size = (res, res)
    #         train_loss = self.train_loop(epoch)

    #         if log_level > 0:
    #             extra_dict = {
    #                 'train_loss': train_loss,
    #                 'epoch': epoch
    #             }
    #             self.eval_and_log(extra_dict)
    #         else:
    #             # still run validation if you want; otherwise remove this line
    #             self.eval_and_log({'epoch': epoch})
    #         if epoch % 10 == 0:
    #             self.save_checkpoint(epoch)

    #     self.eval_and_log({'epoch':epochs-1})
    #     if self.gpu == 0:
    #         ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def train(self, epochs, log_level):
        start = getattr(self, "start_epoch", 0)

        for epoch in range(start, epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)

            # 1) all ranks start epoch together
            self._barrier()

            train_loss = self.train_loop(epoch)

            # 2) all ranks finished backward/allreduces before validation starts
            self._barrier()

            if log_level > 0:
                extra_dict = {"train_loss": train_loss, "epoch": epoch}
                self.eval_and_log(extra_dict)
            else:
                self.eval_and_log({"epoch": epoch})

            # 3) all ranks finished torchmetrics distributed sync before next epoch
            self._barrier()

            # 4) checkpoint: rank0 writes, everyone waits
            if epoch % 10 == 0:
                if self.gpu == 0:
                    ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')
                self._barrier()

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))
        
        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool):
        scaler = GradScaler()
        # model = models[arch](pretrained=pretrained)
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

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    def attack_step(self, input, label):
        adv_input = self.attacker.run(input, label) #.detach()
        return adv_input.requires_grad_(True), label

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

    @param('logging.log_level')
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            images = images.requires_grad_(True)
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            if self.attacker is not None:
                images, target = self.attack_step(images, target)
            with autocast():
                output = self.model(images)
                # loss_train = self.loss(output, target)
                loss_train = self.loss(output, target) + self.model.module.penalty

                aa_reg_alpha = self.config["train"].get("aa_reg_alpha", 0.0)
                aa_reg_beta_alpha = self.config["train"].get("aa_reg_beta_alpha", 0.5)
                if (self.AntiAliasing) and (aa_reg_alpha > 0.0):
                    reg = 0.0
                    for m in self.model.module.modules():
                        if self.learnable and isinstance(m, BlurPool):
                            _ = m.get_kernel()
                            reg = reg + self.lpf_smoothness_loss(m, 
                                                                 alpha=aa_reg_alpha, 
                                                                 beta=aa_reg_alpha * aa_reg_beta_alpha)
                    loss_train = loss_train + reg

            self.scaler.scale(loss_train).backward()
            if hasattr(self.model.module, "post_penalty"):
                self.model.module.post_penalty(images)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end

    def _min_across_ranks(self, x: int) -> int:
        if dist.is_available() and dist.is_initialized():
            t = ch.tensor([x], device=f"cuda:{self.gpu}")
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            return int(t.item())
        return x

    # @param('logging.log_level')
    # def train_loop(self, epoch, log_level):
    #     self.model.train()

    #     module = self.model.module if hasattr(self.model, "module") else self.model

    #     # local_iters = len(self.train_loader)
    #     # iters = self._min_across_ranks(local_iters)
    #     iters = len(self.train_loader)

    #     lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
    #     lrs = np.interp(np.arange(iters), [0, max(iters - 1, 1)], [lr_start, lr_end])

    #     losses = []
    #     iterator = tqdm(self.train_loader, total=iters)

    #     for ix, (images, target) in enumerate(iterator):
    #         # if ix >= iters:
    #         #     break

    #         images = images.requires_grad_(True)

    #         for param_group in self.optimizer.param_groups:
    #             param_group["lr"] = float(lrs[ix])

    #         self.optimizer.zero_grad(set_to_none=True)

    #         if self.attacker is not None:
    #             images, target = self.attack_step(images, target)

    #         with autocast():
    #             output = self.model(images)
    #             loss_train = self.loss(output, target) + module.penalty

    #             aa_reg_alpha = self.config["train"].get("aa_reg_alpha", 0.0)
    #             aa_reg_beta_alpha = self.config["train"].get("aa_reg_beta_alpha", 0.5)
    #             if self.AntiAliasing and aa_reg_alpha > 0.0:
    #                 reg = 0.0
    #                 for m in module.modules():
    #                     if self.learnable and isinstance(m, BlurPool):
    #                         _ = m.get_kernel()
    #                         reg = reg + self.lpf_smoothness_loss(
    #                             m,
    #                             alpha=aa_reg_alpha,
    #                             beta=aa_reg_alpha * aa_reg_beta_alpha,
    #                         )
    #                 loss_train = loss_train + reg

    #         self.scaler.scale(loss_train).backward()

    #         if hasattr(module, "post_penalty"):
    #             module.post_penalty(images)

    #         self.scaler.step(self.optimizer)
    #         self.scaler.update()

    #         if log_level > 0:
    #             losses.append(loss_train.detach())
    #             group_lrs = [f'{g["lr"]:.3f}' for g in self.optimizer.param_groups]
    #             msg = f"ep={epoch}, iter={ix}, shape={tuple(images.shape)}, lrs={group_lrs}"
    #             if log_level > 1:
    #                 msg += f", loss={loss_train.item():.3f}"
    #             iterator.set_description(msg)


    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }

        base = Path(folder).absolute()
        run_dir_str = None

        if dist.is_available() and dist.is_initialized():
            obj = [None]
            if self.gpu == 0:
                base.mkdir(parents=True, exist_ok=True)
                if (base / "last.pt").is_file():
                    run_dir = base
                else:
                    run_dir = base / str(self.uid)
                    run_dir.mkdir(parents=True, exist_ok=True)
                obj[0] = str(run_dir)
            dist.broadcast_object_list(obj, src=0)
            run_dir_str = obj[0]
        else:
            base.mkdir(parents=True, exist_ok=True)
            if (base / "last.pt").is_file():
                run_dir_str = str(base)
            else:
                run_dir = base / str(self.uid)
                run_dir.mkdir(parents=True, exist_ok=True)
                run_dir_str = str(run_dir)

        self.log_folder = Path(run_dir_str).absolute()
        self.start_time = time.time()

        if self.gpu == 0:
            print(f'=> Logging in {self.log_folder}')
            params = {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}

            params_path = self.log_folder / 'params.json'
            if not params_path.exists():
                with open(params_path, 'w+') as handle:
                    json.dump(params, handle)
        

    def _model_state_dict(self):
        return self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

    def _load_model_state_dict(self, state):
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state, strict=True)
        else:
            self.model.load_state_dict(state, strict=True)

    def _optimizer_to_device(self, device: ch.device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if ch.is_tensor(v):
                    state[k] = v.to(device)

    def _ckpt_last_path(self) -> Path:
        return Path(self.log_folder).parent / "weights" / "last.pt"

    def _ckpt_epoch_path(self, epoch: int) -> Path:
        return Path(self.log_folder) / f"epoch_{epoch:04d}.pt"

    def save_checkpoint(self, epoch: int):
        state = {
            "epoch": int(epoch + 1),
            "model": self._model_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        if self.gpu == 0:
            last_path = self._ckpt_last_path()
            tmp_path = last_path.with_suffix(".pt.tmp")
            ch.save(state, tmp_path)
            os.replace(tmp_path, last_path)
            # ch.save(state, self._ckpt_epoch_path(epoch))


    def load_checkpoint_if_exists(self):
        ckpt_path = self._ckpt_last_path()
        print(ckpt_path)
        if not ckpt_path.is_file():
            self.start_epoch = 0
            return

        ckpt = ch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"Checkpoints loaded on rank {self.gpu}")

        self._load_model_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._optimizer_to_device(ch.device(f"cuda:{self.gpu}"))

        if (self.scaler is not None) and (ckpt.get("scaler", None) is not None):
            self.scaler.load_state_dict(ckpt["scaler"])

        self.start_epoch = int(ckpt.get("epoch", 0))

        if self.gpu == 0:
            print(f"=> Resumed from {ckpt_path} (start_epoch={self.start_epoch})")

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

    def _barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.gpu])

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
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

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