#!/usr/bin/env python
import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

def wave100(a, r):
    a0 = 5.29
    y = np.exp(-(a * 2 * r) / a0) * np.sqrt(a ** 3 / (np.pi * (a0 ** 3)))
    return y ** 2


def wave200(a, r):
    a0 = 5.29
    y = np.exp(-(a * r) / a0) * np.sqrt(a ** 3 / (32 * np.pi * (a0 ** 3))) * (
            2 - a * 2 * r / a0)
    return y ** 2


def wave(a, r):
    y = wave100(a, r)
    if a >= 3:
        y += wave200(a, r)
    return 4 * np.pi * (r ** 2) * y





def main():
    # inputs = []
    # for root, dirs, filenames in os.walk('./test'):
    #     for filename in tqdm(filenames):
    #         path = os.path.join(root, filename)
    #         data = np.load(path, allow_pickle=True)
    #         atoms = data['voxel']
    #         properties = data['properties']
    #         atoms = atoms.transpose(3, 0, 1, 2)
    #         inputs.append(atoms)
    #         # print(atoms.shape)
    # inputs = np.array(inputs)
    # inputs = torch.tensor(inputs)
    # print(data.shape)

    # conv = nn.Conv3d(5, 5, (5,))
    # result = conv(data)
    # print(result.shape)

    weights = torch.zeros((5, 7, 7, 7))
    # print(weights.shape)
    atoms = [1, 6, 7, 8, 9]
    for a in range(len(atoms)):
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    weights[a][i][j][k] = wave(atoms[a], (((i - 3) ** 2 + (j - 3) ** 2 + (
                            k - 3) ** 2) ** 0.5) / 4)
        weights[a][3][3][3] = 1

    weights = weights.unsqueeze(0).expand(5, 5, 7, 7, 7)
    print(weights.shape)


    # X_train, X_test, y_train, y_test = train_test_split(inputs, test_size=0.1)
    # start = time()
    # result = F.conv3d(inputs, weights, padding=(3, 3, 3))
    # print(result.shape)
    # print(time() - start)


if __name__ == '__main__':
    main()
