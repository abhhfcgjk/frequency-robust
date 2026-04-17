# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from functools import partial
from torch import optim as optim

try:
    from apex.optimizers import FusedAdam, FusedLAMB
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB or FusedAdam, please install apex.")

from utils import BlurPool

def build_optimizer(config, model, simmim=False, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if simmim:
        if is_pretrain:
            parameters = get_pretrain_param_groups(model, skip, skip_keywords)
        else:
            depths = config.MODEL.SWIN.DEPTHS if config.MODEL.TYPE == 'swin' else config.MODEL.SWINV2.DEPTHS
            num_layers = sum(depths)
            get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
            scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
            parameters = get_finetune_param_groups(model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, get_layer_func, scales, skip, skip_keywords)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    # print("OPTIMIZER:", opt_lower)
    if opt_lower == 'adamw_aa':
        param_group = get_aa_param_groups(model, AntiAliasing=True, learnable=True)
        optimizer = optim.AdamW(param_group, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_adam':
        optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_lamb':
        optimizer = FusedLAMB(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def get_aa_param_groups(model, AntiAliasing, learnable):
    # model = model.module
    params_main, params_lpf = [], []
    if AntiAliasing and learnable:
        aa_lr = 0.0003
        weight_decay = 0.0001

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
    return param_groups

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())
