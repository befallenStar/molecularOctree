# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
inputs: shape[128, 128, 128, 5]
x = pointwise(inputs)
x1 = conv(x, stripe=2)
x2 = conv(x, stripe=2)
x2 = conv(x2)
x3 = channel_pool(inputs)
x3 = conv(x3, stripe=2)
x4 = concat(x, x1, x2, x3)
x4 = conv(x4)
x += x4
x = pool(x)
x = fc(x) * n
"""


class OctreeInception(nn.Module):
    def __init__(self, channel_pooling: bool):
        super(OctreeInception, self).__init__()
        self.channel_pooling = channel_pooling
        self.point_wise = nn.Conv3d(5, 32, (1, 1, 1))
        self.max_pooling = nn.MaxPool3d(2)
        self.conv1 = nn.Conv3d(32, 32, (3, 3, 3), (2, 2, 2), 1)
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3), padding=1)
        self.avg_pooling = nn.AvgPool3d(5)
        self.fcn1 = nn.Linear(1048976, 1024)
        self.fcn2 = nn.Linear(1024, 15)

    def forward(self, inputs):
        point = self.point_wise(inputs)
        x1 = self.max_pooling(point)
        x2 = self.conv1(point)
        x3 = self.conv1(point)
        x3 = self.conv2(x3)
        x = torch.cat((x1, x2, x3), 3)
        if self.channel_pooling:
            x4 = self.avg_pooling(inputs)
            x4 = self.conv1(x4)
            x = torch.cat((x, x4), 3)
        x = self.conv1(x)
        x += self.max_pooling(point)
        x = self.max_pooling(x)
        return x


def main():
    x = torch.rand(16, 5, 128, 128, 128)
    y = torch.rand(16, 15)
    net = OctreeInception(True)
    pred = net(x)
    print(pred)


if __name__ == '__main__':
    main()
