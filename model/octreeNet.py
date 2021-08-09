# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self):
        """
        卷积单元，包含卷积、归一化、激活函数，以及池化层
        """
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv3d()
        self.bn = nn.BatchNorm3d()
        self.relu = F.relu()
        self.pooling = nn.MaxPool3d()

    def forward(self, x):
        """
        卷积单元模型训练
        :return: 训练结果
        """


class OctreeNet(nn.Module):
    def __init__(self):
        """
        定义八叉树网络节点
        """
        super(OctreeNet, self).__init__()
