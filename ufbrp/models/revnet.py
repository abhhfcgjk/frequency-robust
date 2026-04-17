"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR, 2018

(c) Joern-Henrik Jacobsen, 2018
"""

import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F


def split(x):
    n = int(x.size()[1] / 2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class FrequencyDecomposer(nn.Module):
    def __init__(self, wavelet_level=4):
        super(FrequencyDecomposer, self).__init__()
        self.wavelet_level = wavelet_level

    def forward(self, x):
        coeffs = ptwt.wavedec2(x, "haar", level=self.wavelet_level, mode="constant")
        high_freq = self.__reconstruct_from_coeffs(coeffs, keep_approx=False, keep_details=True)
        low_freq = self.__reconstruct_from_coeffs(coeffs, keep_approx=True, keep_details=False)
        return torch.cat((low_freq, high_freq), dim=1)

    def inverse(self, x):
        n = int(x.size()[1] / 2)
        x1 = x[:, :n, :, :].contiguous()
        x2 = x[:, n:, :, :].contiguous()
        return x1 + x2

    @staticmethod
    def __reconstruct_from_coeffs(coeffs: list, keep_approx: bool = True, keep_details: bool = True) -> torch.Tensor:
        modified_coeffs = []
        for i, level in enumerate(coeffs):
            if i == 0:  # Approximation coefficients
                modified_coeffs.append(level if keep_approx else torch.zeros_like(level))
            else:  # Detail coefficients
                if keep_details:
                    modified_coeffs.append(level)
                else:
                    modified_coeffs.append(tuple(torch.zeros_like(d) for d in level))
        return ptwt.waverec2(modified_coeffs, "haar")


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, : x.size(1) - self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = input.shape[0], input.shape[1] // bl_sq, input.shape[2], input.shape[3]
        return input.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)

    def forward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.0, affineBN=True, mult=4):
        """buid invertible bottleneck block"""
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print("")
            print("| Injective iRevNet |")
            print("")
        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_ch // 2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch // 2, int(out_ch // mult), kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(int(out_ch // mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch // mult), int(out_ch // mult), kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch // mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch // mult), out_ch, kernel_size=3, padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """bijective or injective block forward"""
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """bijective or injecitve block inverse"""
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = -self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class WaveNet(nn.Module):
    def __init__(
        self,
        nBlocks,
        nStrides,
        nClasses,
        nChannels=None,
        init_ds=2,
        dropout_rate=0.0,
        affineBN=True,
        in_shape=None,
        mult=4,
    ):
        super(WaveNet, self).__init__()
        self.inverse_nu = 0.01
        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + init_ds // 2)
        self.init_ds = init_ds
        self.in_ch = 2 * in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print("")
        print(" == Building iRevNet %d == " % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch // 2, self.in_ch // 2 * 4, self.in_ch // 2 * 4**2, self.in_ch // 2 * 4**3]

        self.freq_dec = FrequencyDecomposer()
        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(
            irevnet_block,
            nChannels,
            nBlocks,
            nStrides,
            dropout_rate=dropout_rate,
            affineBN=affineBN,
            in_ch=self.in_ch,
            mult=mult,
        )
        self.bn1 = nn.BatchNorm2d(nChannels[-1] * 2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1] * 2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate, affineBN, in_ch, mult):
        """Create stack of irevnet blocks"""
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1] * (depth - 1))
            channels = channels + ([channel] * depth)
        for channel, stride in zip(channels, strides):
            block_list.append(
                _block(
                    in_ch, channel, stride, first=self.first, dropout_rate=dropout_rate, affineBN=affineBN, mult=mult
                )
            )
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """irevnet forward"""
        self.x = x
        x = self.freq_dec(x)
        n = self.in_ch // 2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        self.out_bij = merge(out[0], out[1])
        out = F.relu(self.bn1(self.out_bij))
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def inverse(self):
        """irevnet inverse"""
        out = split(self.out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1 - i].inverse(out)
        out = merge(out[0], out[1])
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        return self.freq_dec.inverse(x)

    @property
    def penalty(self):
        inv = self.inverse()
        # print(inv)
        difference = inv - self.x
        norm = torch.norm(difference, p="fro")
        # print(self.inverse_nu * norm )
        return self.inverse_nu * norm


def WaveNetSmall(wavelet_level=4, num_classes=10):
    return WaveNet(
        nBlocks=[18, 18, 18],
        nStrides=[1, 2, 2],
        nChannels=[16, 64, 256],
        nClasses=10,
        init_ds=0,
        dropout_rate=0.1,
        affineBN=True,
        mult=4,
        in_shape=[3, 32, 32],
    )
