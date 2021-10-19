# -*- encoding: utf-8 -*-
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange


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


class ConvBNDepthWise(nn.Module):
    def __init__(self, cin, cout, k=(1, 1, 1), s=(1, 1, 1), p=0, activation=True):
        super(ConvBNDepthWise, self).__init__()
        self.dwconv = nn.Conv3d(cin, cin, k, s, padding=p, groups=cin)
        self.dwbn = nn.BatchNorm3d(cin)
        self.dwrelu = nn.ReLU()
        self.pwconv = nn.Conv3d(cin, cout, (1, 1, 1))
        self.pwbn = nn.BatchNorm3d(cout)
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


class VoxelNet(nn.Module):
    def __init__(self, cin):
        super(VoxelNet, self).__init__()
        # entry flow
        self.conv1 = ConvBNReLU(cin, 64, (7, 7, 7), (2, 2, 2), 3)
        self.conv2 = ConvBNReLU(64, 128, (7, 7, 7), (2, 2, 2), 3)
        # middle flow
        self.conv3 = ConvBNDepthWise(128, 256, (7, 7, 7), (1, 1, 1), 3)
        self.convblock1 = ConvBNDepthWise(256, 256, (7, 7, 7), (1, 1, 1), 3)
        self.conv4 = ConvBNDepthWise(256, 512, (7, 7, 7), (1, 1, 1), 3)
        self.convblock2 = ConvBNDepthWise(512, 512, (7, 7, 7), (1, 1, 1), 3)
        # exit flow
        self.conv5 = ConvBNDepthWise(512, 128, (1, 1, 1), (2, 2, 2))
        self.fcn1 = nn.Linear(8192, 1024)
        self.fcn2 = nn.Linear(1024, 384)
        self.fcn3 = nn.Linear(384, 15)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # entry flow
        out = self.conv1(x)
        out = self.conv2(out)
        # middle flow
        out = self.conv3(out)
        for _ in range(8):
            out = self.convblock1(out)
        out = self.conv4(out)
        for _ in range(8):
            out = self.convblock2(out)
        # exit flow
        out = self.conv5(out)
        out = self.fcn1(out.view(out.shape[0], -1))
        out = self.relu(out)
        out = self.fcn2(out)
        out = self.tanh(out)
        out = self.fcn3(out)
        return out


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


def main():
    model = VoxelNet(5)
    epochs = 5
    losses = []
    lr = 0.01
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    mae_min = 10

    start = time()
    for e in range(epochs):
        for root, dirs, filenames in os.walk('./input32_2'):
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


def test():
    data = torch.rand(8, 5, 32, 32, 32)
    model = VoxelNet(5)
    result = model(data)
    print(result.shape)


if __name__ == '__main__':
    test()
