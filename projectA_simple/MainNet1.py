import torch
import torch.nn as nn
import torch.nn.functional as F
from DownSample import Downsample


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def DenseBnAct(in_ch, out_ch, act=nn.Identity()):
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False),
                         nn.BatchNorm1d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        self.use_skip = in_ch == out_ch and stride == 1
        inter_ch = in_ch * 3
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, 1, 2, act, group=inter_ch)
        self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)
        self.ds1 = Downsample('zero', channels=inter_ch) if stride > 1 else nn.Identity()
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.ds1(y)
        y = self.conv3(y)
        if self.use_skip:
            y = x + y
        return y


class MainNet(nn.Module):
    model_id = 1

    def __init__(self, in_dim, use_anti_ds=False):
        super().__init__()
        self.use_anti_ds = use_anti_ds
        act = nn.ELU()
        self.conv1 = ConvBnAct(in_dim, 16, 1, 1, 0)
        self.rb1 = ResBlockA(16, 16, 1, act)
        self.rb2 = ResBlockA(16, 32, 2, act)
        self.rb3 = ResBlockA(32, 32, 1, act)
        self.rb4 = ResBlockA(32, 32, 1, act)
        self.rb5 = ResBlockA(32, 64, 2, act)
        self.rb6 = ResBlockA(64, 64, 1, act)
        self.rb7 = ResBlockA(64, 64, 1, act)
        self.rb8 = ResBlockA(64, 128, 2, act)
        self.rb9 = ResBlockA(128, 128, 1, act)
        self.rb10 = ResBlockA(128, 128, 1, act)
        self.rb11 = ResBlockA(128, 256, 2, act)
        self.rb12 = ResBlockA(256, 256, 1, act)
        self.gmaxpool = nn.AdaptiveMaxPool2d(1)
        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        # self.den1 = DenseBnAct(2, 256)
        self.den2 = DenseBnAct(256, 2)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.rb1(y)
        y = self.rb2(y)
        y = self.rb3(y)
        y = self.rb4(y)
        y = self.rb5(y)
        y = self.rb6(y)
        y = self.rb7(y)
        y = self.rb8(y)
        y = self.rb9(y)
        y = self.rb10(y)
        y = self.rb11(y)
        y = self.rb12(y)
        y1 = self.gmaxpool(y)
        y2 = self.gavgpool(y)
        y = y1 + y2
        y = y.flatten(1)
        #y = self.den1(y)
        y = self.den2(y)
        return y


if __name__ == '__main__':
    a = torch.zeros(64, 1, 64, 64).cuda(1)
    net = MainNet(1).cuda(1)
    y = net(a)
    print(y.shape)
