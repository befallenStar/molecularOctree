# -*- encoding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA

atoms_vector = [1, 6, 7, 8, 9]


class VoxelLoader:
    """
    体素类，将点云数据转化成为体素，并提供模糊化方法
    """

    def __init__(self, data):
        """
        加载点云数据
        :param data: 点云数据
        """
        self.d = np.array(data)

    def __call__(self, transformer='', sigma=2, omega=0.5):
        """
        点云数据体素化与模糊化操作
        :param transformer: 模糊化操作参数
        :param sigma: 高斯模糊参数
        :param omega: 小波变换参数
        :return: 模糊化的体素
        """
        voxel = self.__voxelize__()
        if transformer != '':
            voxel = self.__blur__(voxel, transformer=transformer, sigma=sigma,
                                  omega=omega)
        return voxel

    def __voxelize__(self):
        pca = PCA()
        sz = 0.2
        pca.fit_transform(self.d[:-1])
        lx, ly, lz, hx, hy, hz = self.__xyz_range__()
        D, W, H = (hx - lx) / sz, (hy - ly) / sz, (hz - lz) / sz
        D, W, H = int(np.ceil(D)), int(np.ceil(W)), int(np.ceil(H))

        voxel = np.zeros(shape=[D + 1, W + 1, H + 1, len(atoms_vector)],
                         dtype=np.float32)

        for itm in self.d:
            x, y, z = itm[[0, 1, 2]]
            idx = atoms_vector.index(int(itm[-1]))
            pos_x = int(np.ceil((x - lx) / sz))
            pos_y = int(np.ceil((y - ly) / sz))
            pos_z = int(np.ceil((z - lz) / sz))
            voxel[pos_x][pos_y][pos_z][idx] = 1.
        return voxel

    def __blur__(self, voxel, transformer='', sigma=2, omega=0.5):
        D, W, H, C = voxel.shape
        full = np.zeros([D + 8 * sigma, W + 8 * sigma, H + 8 * sigma, C])
        full[4 * sigma:D + 4 * sigma, 4 * sigma:W + 4 * sigma,
        4 * sigma:H + 4 * sigma] = voxel.copy()
        voxel = np.zeros([D + 8 * sigma, W + 8 * sigma, H + 8 * sigma, C])
        xx, yy, zz, cc = np.where(full != 0)
        for iter in range(len(xx)):
            if cc[iter] == 0:
                continue
            for i in range(-4 * sigma, 4 * sigma + 1):
                for j in range(-4 * sigma, 4 * sigma + 1):
                    for k in range(-4 * sigma, 4 * sigma + 1):
                        if i ** 2 + j ** 2 + k ** 2 > 16 * (sigma ** 2):
                            continue
                        pos = 200
                        if transformer == 'gauss':
                            pos *= np.exp(-(i ** 2 + j ** 2 + k ** 2) / (
                                    2 * sigma ** 2))
                        elif transformer == 'wave':
                            pos *= np.exp(-(i ** 2 + j ** 2 + k ** 2) / (
                                    2 * sigma ** 2))
                            cos = np.cos(2 * np.pi * omega * np.sqrt(
                                i ** 2 + j ** 2 + k ** 2))
                            pos *= cos
                        elif transformer == 'octree':
                            pos *= self.__wave__(atoms_vector[cc[iter]],
                                                 np.sqrt(
                                                     i ** 2 + j ** 2 + k ** 2))
                        voxel[xx[iter] + i][yy[iter] + j][zz[iter] + k][
                            cc[iter]] += \
                            full[xx[iter]][yy[iter]][zz[iter]][
                                cc[iter]] * self.__tanh__(pos)

        return voxel

    def __xyz_range__(self):
        x_min = self.d[:, 0].min()
        x_max = self.d[:, 0].max()
        y_min = self.d[:, 1].min()
        y_max = self.d[:, 1].max()
        z_min = self.d[:, 2].min()
        z_max = self.d[:, 2].max()
        return x_min, y_min, z_min, x_max, y_max, z_max

    def __sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __wave__(self, a, r):
        """
        波函数计算电子分布情况
        :param a: 原子序数
        :param r: 径向距离
        :return:
        """
        y = self.__wave100__(a, r)
        if 3 <= a:
            y += self.__wave200__(a, r)
        return 4 * np.pi * ((2 * a) ** 2) * (y ** 2)

    def __wave100__(self, a, r):
        """
        计算内层电子分布情况
        :param a: 原子序数
        :param r: 径向距离
        :return:
        """
        a0 = 5.29
        y = np.exp(-(a * 2 * r) / a0) * np.sqrt(a ** 3 / (np.pi * (a0 ** 3)))
        return y

    def __wave200__(self, a, r):
        """
        计算外层电子分布情况
        :param a: 原子序数
        :param r: 径向距离
        :return:
        """
        a0 = 5.29
        y = np.exp(-(a * r) / a0) * np.sqrt(
            a ** 3 / (32 * np.pi * (a0 ** 3))) * (2 - a * 2 * r / a0)
        return y
