import torch
import torch.nn as nn
import torch.nn.functional as F
from DownSample import Downsample


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False),
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
        # inter_ch = in_ch * 3
        self.conv1 = ConvBnAct(in_ch, out_ch, 3, 1, 1, act)
        self.conv2 = ConvBnAct(out_ch, out_ch, 3, stride, 1, act, group=out_ch)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class ResBlockB(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        #inter_ch = in_ch * 3
        self.conv1 = ConvBnAct(in_ch, out_ch, 3, 1, 1, act)
        self.up = nn.Upsample(scale_factor=2, mode='area') if stride > 1 else nn.Identity()
        self.conv2 = ConvBnAct(out_ch, out_ch, 3, 1, 1, act, group=out_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = self.up(y)
        y = self.conv2(y)
        return y


class MainNet(nn.Module):
    model_id = 2

    def __init__(self, in_dim):
        super().__init__()
        act = nn.SELU()
        self.conv1 = ConvBnAct(in_dim, 8, 1, 1, 0)
        self.conv2 = ConvBnAct(8, 16, 3, 2, 1, act)
        self.rb1a = ResBlockA(16, 16, 1, act)
        self.rb1b = ResBlockA(16, 16, 1, act)
        self.rb2a = ResBlockA(16, 32, 2, act)
        self.rb2b = ResBlockA(32, 32, 1, act)
        self.rb3a = ResBlockA(32, 64, 2, act)
        self.rb3b = ResBlockA(64, 64, 1, act)
        self.rb4a = ResBlockA(64, 128, 2, act)
        self.rb4b = ResBlockA(128, 128, 1, act)
        self.rb5a = ResBlockA(128, 256, 2, act)
        self.rb5b = ResBlockA(256, 256, 1, act)

        self.rb6a = ResBlockB(256, 128, 2, act)
        self.rb6b = ResBlockB(128+128, 128, 1, act)
        self.rb7a = ResBlockB(128, 64, 2, act)
        self.rb7b = ResBlockB(64+64, 64, 1, act)
        self.rb8a = ResBlockB(64, 32, 2, act)
        self.rb8b = ResBlockB(32+32, 32, 1, act)

        self.rb9a = ResBlockB(32, 32, 2, act)
        self.rb9b = ResBlockB(32, 32, 1, act)

        self.out_conv1 = ConvBnAct(32, 32, 3, 1, 1, act)
        self.out_conv2 = ConvBnAct(32, 3, 3, 1, 1)

    def forward(self, x):
        # 256
        y = self.conv1(x)
        y = self.conv2(y)
        # 128
        y = self.rb1a(y)
        y = self.rb1b(y)
        y = self.rb2a(y)
        # 64
        y3 = y
        y = self.rb2b(y)
        y = self.rb3a(y)
        # 32
        y2 = y
        y = self.rb3b(y)
        y = self.rb4a(y)
        # 16
        y1 = y
        y = self.rb4b(y)
        y = self.rb5a(y)
        # 8
        y = self.rb5b(y)
        # -------------------
        y = self.rb6a(y)
        # 16
        y = torch.cat([y1, y], 1)
        y = self.rb6b(y)
        y = self.rb7a(y)
        # 32
        y = torch.cat([y2, y], 1)
        y = self.rb7b(y)
        y = self.rb8a(y)
        # 64
        y = torch.cat([y3, y], 1)
        y = self.rb8b(y)
        y = self.rb9a(y)
        # 128
        y = self.rb9b(y)
        y = self.out_conv1(y)
        y = self.out_conv2(y)
        return y


if __name__ == '__main__':
    a = torch.zeros(32, 1, 256, 256).cuda(1)
    net = MainNet(1).cuda(1)
    y = net(a)
    print(y.shape)
