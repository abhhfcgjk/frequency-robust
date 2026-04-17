# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=4, stride=2, pad_off=0, learnable=False):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        self.learnable = bool(learnable)

        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

        if self.learnable:
            base = torch.from_numpy(a)
            self.w = nn.Parameter(torch.log(torch.expm1(base + 1e-3)))  # shape [k]
        else:
            k2 = torch.from_numpy(a[:, None] * a[None, :])
            k2 = (k2 / k2.sum()).view(1, 1, self.filt_size, self.filt_size)
            self.register_buffer("filt", k2.repeat(self.channels, 1, 1, 1))

    def _make_kernel(self, x):
        a = F.softplus(self.w)  # ≥0, shape [k]
        a = a / (a.sum() + 1e-12)
        k2 = torch.outer(a, a)  # [k,k]
        k2 = k2 / (k2.sum() + 1e-12)
        return k2[None, None].to(dtype=x.dtype, device=x.device).repeat(x.size(1), 1, 1, 1)  # [channels, 1,k,k]

    def get_kernel(self):
        a = F.softplus(self.w)
        return a / (a.sum() + 1e-12)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]

        if self.learnable:
            filt = self._make_kernel(inp)
        else:
            filt = self.filt
        filt = filt.to(dtype=inp.dtype, device=inp.device)
        return F.conv2d(self.pad(inp), filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer
