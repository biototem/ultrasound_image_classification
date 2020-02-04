#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 05:58:23 2019

@author: root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv2 = ConvBnAct(inter_ch, inter_ch, 5, stride, 2, act, group=inter_ch)
        self.conv3 = ConvBnAct(inter_ch, out_ch, 1, 1, 0)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.use_skip:
            y = x + y
        return y


class MainNet(nn.Module):
    model_id = 2

    def __init__(self, in_dim):
        super().__init__()
        act = nn.ELU()
        self.conv1 = ConvBnAct(in_dim, 32, 1, 1, 0)
        self.conv2 = ConvBnAct(32, 64, 3, 2, 1, act)
        self.conv3 = ConvBnAct(64, 64, 3, 1, 1, act)
        self.conv4 = ConvBnAct(64, 128, 3, 2, 1, act)
        self.conv5 = ConvBnAct(128, 128, 3, 1, 1, act)
        self.conv6 = ConvBnAct(128, 256, 3, 2, 1, act)
        self.conv7 = ConvBnAct(256, 256, 3, 1, 1, act)
        self.conv8 = ConvBnAct(256, 256, 3, 2, 1, act)
        self.conv9 = ConvBnAct(256, 256, 3, 1, 1, act)
        self.gmaxpool = nn.AdaptiveMaxPool2d(1)
        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        #self.den1 = DenseBnAct(256 * 8 * 8, 256)
        self.den2 = DenseBnAct(256, 2)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)
        y = self.conv9(y)
        y1 = self.gmaxpool(y)
        y2 = self.gavgpool(y)
        y = (y1 + y2).flatten(1)
        #y = y.flatten(1)
        #y = self.den1(y)
        y = self.den2(y)
        return y


if __name__ == '__main__':
    a = torch.zeros(64, 1, 32, 32).cuda(1)
    net = MainNet(1).cuda(1)
    y = net(a)
    print(y.shape)
