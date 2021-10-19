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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./input32_2')
parser.add_argument('--diff', type=str, choices=['origin', 'gauss', 'wave', 'octree'], default='origin')
parser.add_argument('--sigma', type=int, default=1)
args = parser.parse_args()


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


class VolConvSingle(nn.Module):
    def __init__(self, cin, activation=True):
        super(VolConvSingle, self).__init__()
        self.block = nn.Sequential(
            ConvBNDepthWise(cin, (3, 3, 3), 1, True),
            ConvBNDepthWise(cin, (3, 3, 3), 1, False),
        )
        self.pool = None
        self.conv = None
        self.relu = None
        if activation:
            self.pool = nn.MaxPool3d((3, 3, 3), stride=2, padding=1)
            self.conv = nn.Conv3d(cin, 4 * cin, (1, 1, 1))
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out += x
        if self.relu:
            out = self.pool(out)
            out = self.conv(out)
            out = self.relu(out)
        return out


class GaussSmoothing(nn.Module):
    def __init__(self, channel, sigma):
        super(GaussSmoothing, self).__init__()
        self.channel = channel
        self.sigma = sigma
        self.weight = self.initiate_weight()

    def gauss(self, i, j, k):
        return np.exp(-(i ** 2 + j ** 2 + k ** 2) / (2 * self.sigma ** 2))

    def initiate_weight(self):
        weight = torch.zeros((self.channel, 7, 7, 7))
        for c in range(self.channel):
            for i in range(7):
                for j in range(7):
                    for k in range(7):
                        weight[c][i][j][k] = self.gauss(i, j, k)
            weight[c][3][3][3] = 1

        weight = weight.unsqueeze(0).expand(5, 5, 7, 7, 7)
        return weight

    def forward(self, x):
        return F.conv3d(x, self.weight, padding=(3, 3, 3))


class WaveDiffusion(nn.Module):
    def __init__(self, channel, sigma):
        super(WaveDiffusion, self).__init__()
        self.channel = channel
        self.sigma = sigma
        self.weight = self.initiate_weight()

    def wave(self, i, j, k):
        r2 = i ** 2 + j ** 2 + k ** 2
        gau = np.exp(-r2 / (2 * self.sigma ** 2))
        cos = np.cos(2 * np.sqrt(r2) * np.pi / self.sigma)
        return gau * cos

    def initiate_weight(self):
        weight = torch.zeros((self.channel, 7, 7, 7))
        for c in range(self.channel):
            for i in range(7):
                for j in range(7):
                    for k in range(7):
                        weight[c][i][j][k] = self.wave(i, j, k)
            weight[c][3][3][3] = 1

        weight = weight.unsqueeze(0).expand(5, 5, 7, 7, 7)
        return weight

    def forward(self, x):
        return F.conv3d(x, self.weight, padding=(3, 3, 3))


class VisMolDiffusion(nn.Module):
    def __init__(self, atoms):
        super(VisMolDiffusion, self).__init__()
        assert len(atoms) == 5
        self.atoms = atoms
        self.weight = self.initiate_weight()

    def wave100(self, a, r):
        a0 = 5.29
        y = np.exp(-(a * 2 * r) / a0) * np.sqrt(a ** 3 / (np.pi * (a0 ** 3)))
        return y ** 2

    def wave200(self, a, r):
        a0 = 5.29
        y = np.exp(-(a * r) / a0) * np.sqrt(
            a ** 3 / (32 * np.pi * (a0 ** 3))) * (
                    2 - a * 2 * r / a0)
        return y ** 2

    def wave(self, a, r):
        y = self.wave100(a, r)
        if a >= 3:
            y += self.wave200(a, r)
        return 4 * np.pi * (r ** 2) * y

    def initiate_weight(self):
        weight = torch.zeros((len(self.atoms), 7, 7, 7))
        for a in range(len(self.atoms)):
            for i in range(7):
                for j in range(7):
                    for k in range(7):
                        weight[a][i][j][k] = self.wave(self.atoms[a], (((
                                                                                i - 3) ** 2 + (
                                                                                j - 3) ** 2 + (
                                                                                k - 3) ** 2) ** 0.5) / 4)
            weight[a][3][3][3] = 1

        weight = weight.unsqueeze(0).expand(5, 5, 7, 7, 7)
        return weight

    def forward(self, x):
        return F.conv3d(x, self.weight, padding=(3, 3, 3))


class VisMol(nn.Module):
    def __init__(self, diffusion='origin', sigma=None):
        """
        网络模型
        :param diffusion: 扩散类型
            origin: 不扩散
            gauss: 高斯模糊
            wave: 加了cos的高斯模糊
            octree: 电子波函数扩散
        :param sigma:
        """
        super(VisMol, self).__init__()
        self.diffusion = None
        if diffusion == 'gauss' and sigma:
            self.diffusion = GaussSmoothing(5, sigma)
        elif diffusion == 'wave' and sigma:
            self.diffusion = WaveDiffusion(5, sigma)
        elif diffusion == 'octree':
            self.diffusion = VisMolDiffusion([1, 6, 7, 8, 9])
        self.conv = ConvBNReLU(5, 16, (1, 1, 1))
        self.cap = ChannelAvgPool()
        self.channelConv1 = ConvBNReLU(1, 16, (3, 3, 3), p=1)
        self.volConv1 = VolConv(16, activation=True)
        self.channelConv2 = ConvBNReLU(16, 64, (3, 3, 3), s=(2, 2, 2), p=1)
        self.volConv2 = VolConv(64, activation=True)
        self.channelConv3 = ConvBNReLU(64, 256, (3, 3, 3), s=(2, 2, 2), p=1)
        self.volConv3 = VolConv(256, activation=True)
        self.dropout = nn.Dropout3d()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fcn = nn.Linear(1024, 15)

    def forward(self, x):
        # 扩散
        if self.diffusion:
            x = self.diffusion(x)
        # 第一轮
        x_out = self.conv(x)
        x_channel_pool = self.cap(x)
        x_channel_pool = self.channelConv1(x_channel_pool)
        x_out = self.volConv1(x_out, x_channel_pool)
        # 第二轮
        x_channel_pool = self.channelConv2(x_channel_pool)
        x_out = self.volConv2(x_out, x_channel_pool)
        # 第三轮
        x_channel_pool = self.channelConv3(x_channel_pool)
        x_out = self.volConv3(x_out, x_channel_pool)
        # 通过全连接获得特征
        x_out = self.dropout(x_out)
        x_out = self.gap(x_out)
        x_out = torch.squeeze(x_out)
        x_out = self.dropout(x_out)
        x_out = self.fcn(x_out)
        return x_out


class VisMolA(nn.Module):
    def __init__(self, diffusion='origin', sigma=None):
        """
        网络模型
        :param diffusion: 扩散类型
            origin: 不扩散
            gauss: 高斯模糊
            wave: 加了cos的高斯模糊
            octree: 电子波函数扩散
        :param sigma:
        """
        super(VisMolA, self).__init__()
        self.diffusion = None
        if diffusion == 'gauss' and sigma:
            self.diffusion = GaussSmoothing(5, sigma)
        elif diffusion == 'wave' and sigma:
            self.diffusion = WaveDiffusion(5, sigma)
        elif diffusion == 'octree':
            self.diffusion = VisMolDiffusion([1, 6, 7, 8, 9])
        self.conv = ConvBNReLU(5, 16, (1, 1, 1))
        self.volConv1 = VolConvSingle(16, activation=True)
        self.volConv2 = VolConvSingle(64, activation=True)
        self.volConv3 = VolConvSingle(256, activation=True)
        self.dropout = nn.Dropout3d()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fcn = nn.Linear(1024, 15)

    def forward(self, x):
        # 扩散
        if self.diffusion:
            x = self.diffusion(x)
        # 第一轮
        x_out = self.conv(x)
        x_out = self.volConv1(x_out)
        # 第二轮
        x_out = self.volConv2(x_out)
        # 第三轮
        x_out = self.volConv3(x_out)
        # 通过全连接获得特征
        x_out = self.dropout(x_out)
        x_out = self.gap(x_out)
        x_out = torch.squeeze(x_out)
        x_out = self.dropout(x_out)
        x_out = self.fcn(x_out)
        return x_out


class VisMolB(nn.Module):
    def __init__(self, diffusion='origin', sigma=None):
        """
        网络模型
        :param diffusion: 扩散类型
            origin: 不扩散
            gauss: 高斯模糊
            wave: 加了cos的高斯模糊
            octree: 电子波函数扩散
        :param sigma:
        """
        super(VisMolB, self).__init__()
        self.diffusion = None
        if diffusion == 'gauss' and sigma:
            self.diffusion = GaussSmoothing(5, sigma)
        elif diffusion == 'wave' and sigma:
            self.diffusion = WaveDiffusion(5, sigma)
        elif diffusion == 'octree':
            self.diffusion = VisMolDiffusion([1, 6, 7, 8, 9])
        self.cap = ChannelAvgPool()
        self.channelConv = ConvBNReLU(1, 16, (3, 3, 3), p=1)
        self.volConv1 = VolConvSingle(16, activation=True)
        self.volConv2 = VolConvSingle(64, activation=True)
        self.volConv3 = VolConvSingle(256, activation=True)
        self.dropout = nn.Dropout3d()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fcn = nn.Linear(1024, 15)

    def forward(self, x):
        # 扩散
        if self.diffusion:
            x = self.diffusion(x)
        # 第一轮
        x_out = self.cap(x)
        x_out = self.channelConv(x_out)
        x_out = self.volConv1(x_out)
        # 第二轮
        x_out = self.volConv2(x_out)
        # 第三轮
        x_out = self.volConv3(x_out)
        # 通过全连接获得特征
        x_out = self.dropout(x_out)
        x_out = self.gap(x_out)
        x_out = torch.squeeze(x_out)
        x_out = self.dropout(x_out)
        x_out = self.fcn(x_out)
        return x_out


def train(model, optimizer, inputs, props, mode, losses):
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()

    model.train()
    if mode == 'train':
        # 训练
        print('*' * 25 + mode + '*' * 25)
        w_loss, w_mae = 0, 1
        start = time()
        train_x, train_y = Variable(inputs), Variable(props)

        # 梯度归零
        optimizer.zero_grad()
        # 开始训练
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        mae = MAE_fn(pred, train_y)
        # 反向传播
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        w_loss += loss.data
        w_mae += mae.data
        losses.append(loss.data)

        print(
            "loss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                w_loss, w_mae / len(inputs), time() - start))
    elif mode == 'valid':
        print('*' * 25 + mode + '*' * 25)
        # 计算验证误差
        w_loss, w_mae = 0, 1
        start = time()
        valid_x, valid_y = Variable(inputs), Variable(props)

        # 梯度归零
        optimizer.zero_grad()
        # 开始训练
        pred = model(valid_x)
        loss = loss_fn(pred, valid_y)
        mae = MAE_fn(pred, valid_y)

        w_loss += loss.data
        w_mae += mae.data

        print(
            "loss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                w_loss, w_mae / len(inputs), time() - start))
    elif mode == 'test':
        # 得到测试结果
        print('*' * 25 + mode + '*' * 25)
        w_loss, w_mae = 0, 1
        start = time()
        test_x, test_y = Variable(inputs), Variable(props)

        # 梯度归零
        optimizer.zero_grad()
        # 开始训练
        pred = model(test_x)
        loss = loss_fn(pred, test_y)
        mae = MAE_fn(pred, test_y)

        w_loss += loss.data
        w_mae += mae.data

        print(
            "loss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                w_loss, w_mae / len(inputs), time() - start))
    return w_mae / len(inputs)


def test():
    start = time()
    model = VisMolB()
    data = torch.rand(8, 5, 32, 32, 32)
    result = model(data)
    torch.save(model, '../vismolA.pth')
    print(result.shape)
    print(time() - start)


def main():
    # origin gauss wave octree
    # None   1 4   4    None
    model = VisMolA(diffusion=args.diff, sigma=args.sigma)
    epochs = 5
    losses = []
    lr = 0.01
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    mae_min = 10

    start = time()
    for e in range(epochs):
        for root, dirs, filenames in os.walk(args.path):
            print("loading...")
            inputs = []
            props = []
            for i in trange(len(filenames)):
                filename = filenames[i]
                path = os.path.join(root, filename)
                data = np.load(path, allow_pickle=True)
                atoms = data['voxel']
                properties = data['properties']
                atoms = atoms.transpose(3, 0, 1, 2)
                inputs.append(atoms)
                props.append(properties)
                if i % 8 == 7:
                    # 开始训练
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    props = torch.tensor(props, dtype=torch.float32)
                    props = torch.arctan(props) * 2 / np.pi

                    mode = ''
                    if i <= 107120:
                        mode = 'train'
                    elif 107120 < i <= 120496:
                        mode = 'valid'
                    elif 120496 < i <= 133885:
                        mode = 'test'

                    print("epoch: {}".format(e))
                    mae = train(model, optimizer, inputs, props, mode, losses)
                    mae_min = mae if mae < mae_min else mae_min
                    inputs = []
                    props = []

            print("min mae: {}".format(mae_min))

    print('total time: {}'.format(time() - start))
    plt.plot(range(len(losses)), losses)
    plt.savefig('./result_1.png')


if __name__ == '__main__':
    # main()
    test()
