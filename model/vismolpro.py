# -*- encoding: utf-8 -*-
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange


class ConvBNDepthWise(nn.Module):
    def __init__(self, cin, k, p=0, activation=True):
        super(ConvBNDepthWise, self).__init__()
        self.dwconv = nn.Conv3d(cin, cin, k, padding=p, groups=cin)
        self.dwbn = nn.BatchNorm3d(cin)
        self.dwrelu = nn.ReLU()
        self.pwconv = nn.Conv3d(cin, cin, (1, 1, 1), padding=0)
        self.pwbn = nn.BatchNorm3d(cin)
        self.pwrelu = None
        if activation:
            self.pwrelu = nn.ReLU()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.dwbn(x)
        x = self.dwrelu(x)
        x = self.pwconv(x)
        x = self.pwbn(x)
        if self.pwrelu:
            x = self.pwrelu(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k, s=(1, 1, 1), p=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(cin, cout, k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAvgPool(nn.Module):
    def __init__(self):
        super(ChannelAvgPool, self).__init__()
        self.pool = nn.AvgPool3d((1, 1, 5))
        self.dropout = nn.Dropout3d()

    def forward(self, x):
        out = self.pool(x.permute(0, 2, 3, 4, 1))
        out = self.dropout(out)
        return out.permute(0, 4, 1, 2, 3)


class VolConv(nn.Module):
    def __init__(self, cin, activation=True):
        super(VolConv, self).__init__()
        self.block1 = nn.Sequential(
            ConvBNDepthWise(cin, (3, 3, 3), 1, True),
            ConvBNDepthWise(cin, (3, 3, 3), 1, False),
        )
        self.block2 = nn.Sequential(
            ConvBNDepthWise(cin, (3, 3, 3), 1, True),
            ConvBNDepthWise(cin, (3, 3, 3), 1, False),
        )
        self.pool = None
        self.conv = None
        self.relu = None
        if activation:
            self.pool = nn.MaxPool3d((3, 3, 3), stride=2, padding=1)
            self.conv = nn.Conv3d(2 * cin, 4 * cin, (1, 1, 1))
            self.relu = nn.ReLU()

    def forward(self, x, y):
        """
        :param x: size of m * m * m * cin
        :param y: size of m * m * m * cin
        :return: size of n * n * n * cout, n = m / 2, cout = 2 * cin
        """
        left = self.block1(x)
        left += x
        right = self.block2(y)
        right += y
        out = torch.cat((left, right), 1)
        if self.relu:
            out = self.pool(out)
            out = self.conv(out)
            out = self.relu(out)
        return out
