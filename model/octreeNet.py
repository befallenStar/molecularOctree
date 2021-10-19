# -*- encoding: utf-8 -*-
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange


class ConvUnit3d(nn.Module):
    def __init__(self, cin, cout, k=(5, 5, 5)):
        """
        卷积单元，包含卷积、归一化、激活函数，以及池化层
        """
        super(ConvUnit3d, self).__init__()
        self.conv = nn.Conv3d(cin, cout, k)
        self.bn = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool3d(4, padding=2)

    def forward(self, x):
        """
        卷积单元模型训练
        :return: 训练结果
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class FCN(nn.Module):
    def __init__(self, cin, cout):
        """
        全连接层
        """
        super(FCN, self).__init__()
        self.fcn = nn.Linear(cin, cout)
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        全连接层训练数据
        :param x: 输入数据
        :return: 训练结果
        """
        x = self.fcn(x)
        x = self.dropout(x)
        return x


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


class OctreeNet(nn.Module):
    def __init__(self, cin, diffusion='origin', sigma=None):
        """
        定义八叉树网络节点
        :param cin: 网络输入通道数
        :param diffusion: 扩散方式，
            'origin': 不扩散
            'gauss': 高斯平滑
            'octree': 波函数扩散
        :param sigma: 扩散参数
        """
        super(OctreeNet, self).__init__()
        self.diffusion = None
        if diffusion == 'gauss' and sigma:
            self.diffusion = GaussSmoothing(5, sigma)
        elif diffusion == 'wave' and sigma:
            self.diffusion = WaveDiffusion(5, sigma)
        elif diffusion == 'octree':
            self.diffusion = VisMolDiffusion([1, 6, 7, 8, 9])
        self.conv3d_1 = ConvUnit3d(cin, 16)
        self.conv3d_2 = ConvUnit3d(16, 64)
        self.conv3d_3 = ConvUnit3d(64, 256, (1, 1, 1))
        self.conv3d = nn.Sequential(
            self.conv3d_1,
            self.conv3d_2,
            self.conv3d_3,
        )
        self.fcn_1 = FCN(2048, 512)
        self.fcn_2 = FCN(512, 128)
        self.fcn_3 = FCN(128, 15)
        self.fcn = nn.Sequential(
            self.fcn_1,
            self.fcn_2,
            self.fcn_3,
        )

    def forward(self, x):
        """
        八叉树网络训练数据
        :param x: 输入数据
        :return: 训练结果
        """
        if self.diffusion:
            x = self.diffusion(x)
        x = self.conv3d(x)
        x = self.fcn(x.view(x.shape[0], -1))
        return x


def load_train():
    model = OctreeNet(5, 'gauss', 1)
    for root, dirs, filenames in os.walk('../test16'):
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

                inputs = TensorDataset(inputs)
                props = TensorDataset(props)

                mode = ''
                if i <= 64:
                    mode = 'train'
                elif 64 < i <= 72:
                    mode = 'valid'
                elif 72 < i <= 80:
                    mode = 'test'

                epochs = 5

                train(model, inputs, props, mode, epochs)
                inputs = []
                props = []
        print("loading done.")


def train(model, inputs, props, mode, epochs=5):
    losses = []
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        MAE_fn = MAE_fn.cuda()
    lr = 0.1
    # 加载数据并划分
    batch_size = 8
    loader_x = torch.utils.data.DataLoader(inputs, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
    loader_y = torch.utils.data.DataLoader(props, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
    for e in range(epochs):
        model.train()
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        if mode == 'train':
            # 训练
            print('*' * 25 + mode + '*' * 25)
            w_loss, w_mae = 0, 1
            start = time()
            for i, data in enumerate(zip(loader_x, loader_y), 0):
                train_x, train_y = data
                train_x, train_y = Variable(train_x), Variable(train_y)
                if torch.cuda.is_available():
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
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

                if torch.cuda.is_available():
                    loss = loss.cpu()
                    mae = mae.cpu()
                w_loss += loss.data
                w_mae += mae.data
                losses.append(loss.data)

                if i % 100 == 0:
                    print(
                        "Epoch {:2d}, \titer: {:2d}, \tloss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                            e, i, w_loss, w_mae / len(inputs),
                                          time() - start))
        elif mode == 'valid':
            print('*' * 25 + mode + '*' * 25)
            # 计算验证误差
            w_loss, w_mae = 0, 1
            start = time()
            for i, data in enumerate(zip(loader_x, loader_y), 0):
                valid_x, valid_y = data
                valid_x, valid_y = Variable(valid_x[0]), Variable(valid_y[0])
                if torch.cuda.is_available():
                    valid_x = valid_x.cuda()
                    valid_y = valid_y.cuda()
                # 梯度归零
                optimizer.zero_grad()
                # 开始训练
                pred = model(valid_x)
                loss = loss_fn(pred, valid_y)
                mae = MAE_fn(pred, valid_y)

                if torch.cuda.is_available():
                    loss = loss.cpu()
                    mae = mae.cpu()
                w_loss += loss.data
                w_mae += mae.data
                losses.append(loss.data)

                if i % 100 == 0:
                    print(
                        "Epoch {:2d}, \titer: {:2d}, \tloss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                            e, i, w_loss, w_mae / len(inputs),
                                          time() - start))
        elif mode == 'test':
            # 得到测试结果
            print('*' * 25 + mode + '*' * 25)
            w_loss, w_mae = 0, 1
            start = time()
            for i, data in enumerate(zip(loader_x, loader_y), 0):
                test_x, test_y = data
                test_x, test_y = Variable(test_x[0]), Variable(test_y[0])
                if torch.cuda.is_available():
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                # 梯度归零
                optimizer.zero_grad()
                # 开始训练
                pred = model(test_x)
                loss = loss_fn(pred, test_y)
                mae = MAE_fn(pred, test_y)

                if torch.cuda.is_available():
                    loss = loss.cpu()
                    mae = mae.cpu()
                w_loss += loss.data
                w_mae += mae.data
                losses.append(loss.data)

                if i % 100 == 0:
                    print(
                        "Epoch {:2d}, \titer: {:2d}, \tloss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                            e, i, w_loss, w_mae / len(inputs),
                                          time() - start))


def main():
    # load_train()
    model = OctreeNet(5)
    inputs = torch.rand(8, 5, 64, 64, 64)
    props = torch.rand(8, 15)
    train(model, inputs, props, 'train', 5)

if __name__ == '__main__':
    main()
