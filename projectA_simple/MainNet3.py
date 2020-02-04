import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NewAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.tensor(0.95, dtype=torch.float))
        self.param2 = nn.Parameter(torch.tensor(20, dtype=torch.float))
        self.param3 = nn.Parameter(torch.tensor(0.05, dtype=torch.float))

    def forward(self, x):
        return x * self.param1 + torch.sin(x * self.param2) * self.param3


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def DenseBnAct(in_ch, out_ch, act=nn.Identity()):
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False),
                         nn.BatchNorm1d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, ker_sz=5):
        super().__init__()
        pad = ker_sz // 2
        self.use_skip = in_ch == out_ch and stride == 1
        inter_ch = in_ch * 3
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, ker_sz, stride, pad, act, group=inter_ch)
        self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.use_skip:
            y = x + y
        return y


class MainNet(nn.Module):
    model_id = 3

    def __init__(self, in_dim, use_anti_ds=False):
        super().__init__()
        self.use_anti_ds = use_anti_ds
        act = nn.SELU()
        self.act = act
        self.conv1 = ConvBnAct(in_dim, 16, 3, 2, 1)
        self.conv2 = ConvBnAct(16, 24, 3, 1, 1, act)
        self.res = nn.Sequential(
            ResBlockA(24, 24, 1, act),
            ResBlockA(24, 48, 2, act),
            ResBlockA(48, 48, 1, act),
            ResBlockA(48, 48, 1, act),
            ResBlockA(48, 96, 2, act),
            ResBlockA(96, 96, 1, act),
            ResBlockA(96, 96, 1, act),
            ResBlockA(96, 96, 1, act),
            ResBlockA(96, 192, 2, act),
            ResBlockA(192, 192, 1, act),
            ResBlockA(192, 192, 1, act),
            ResBlockA(192, 386, 2, act),
            ResBlockA(386, 512, 2, act),
        )
        self.gmaxpool = nn.AdaptiveMaxPool2d(1)
        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        #self.den1 = DenseBnAct(256*8*8, 256)
        self.den2 = DenseBnAct(512, 2)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.res(y)
        y1 = self.gmaxpool(y)
        y2 = self.gavgpool(y)
        y = y1 + y2
        y = y.flatten(1)
        #y = self.den1(y)
        y = self.den2(y)
        return y


if __name__ == '__main__':
    a = torch.zeros(64, 1, 256, 256).cuda(1)
    net = MainNet(1).cuda(1)
    y = net(a)
    print(y.shape)
