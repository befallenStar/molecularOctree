# -*- encoding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA

atoms_vector = [1, 6, 7, 8, 9]
a0 = 5.29


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

    def __call__(self, sz=0.2, norm=2):
        """
        点云数据体素化与模糊化操作
        :param transformer: 模糊化操作参数
        :param sigma: 高斯模糊参数
        :param omega: 小波变换参数
        :return: 模糊化的体素
        """
        voxel = self.__voxelize__(sz=sz, norm=norm)
        voxel = self.__padding__(voxel, int(12.8/sz))
        # if transformer == 'octree':
        #     voxel = self.__octree__(voxel)
        # elif transformer != 'origin':
        #     voxel = self.__blur__(voxel, transformer=transformer, sigma=sigma,
        #                           omega=omega)
        return voxel

    def __voxelize__(self, sz=0.2, norm=2):
        """
        体素化
        :param sz: 初始体素大小
        :param norm: norm策略
            1: 对于体素格数均小于标准大小的一半的，将体素大小进行分割
            2: 正常
            3: 将分子中最长轴分割为标准体素格数，并以此确定体素大小
        :return:
        """
        pca = PCA()
        pca.fit_transform(self.d[:-1])
        lx, ly, lz, hx, hy, hz = self.__xyz_range__()
        if norm == 1:
            while max(hx - lx, hy - ly, hz - lz) < 6.4:
                self.d[:, :-1] *= 2
                lx, ly, lz, hx, hy, hz = self.__xyz_range__()
        elif norm == 3:
            scale = (12.8 - sz) / max(hx - lx, hy - ly, hz - lz)
            self.d[:, :-1] *= scale
            lx, ly, lz, hx, hy, hz = self.__xyz_range__()
        D, W, H = (hx - lx) / sz, (hy - ly) / sz, (hz - lz) / sz
        D, W, H = int(np.ceil(D)), int(np.ceil(W)), int(np.ceil(H))
        B = int(12.8 / sz)
        bias_D, bias_W, bias_H = int((B - D) / 2), int((B - W) / 2), int((B - H) / 2)
        voxel = np.zeros(shape=[B, B, B, len(atoms_vector)],
                         dtype=np.float32)

        for itm in self.d:
            x, y, z = itm[[0, 1, 2]]
            idx = atoms_vector.index(int(itm[-1]))
            # 画图时用来去除氢
            # if idx == 0:
            #     continue
            pos_x = int((x - lx) / sz)
            pos_y = int((y - ly) / sz)
            pos_z = int((z - lz) / sz)
            voxel[pos_x+bias_D][pos_y+bias_W][pos_z+bias_H][idx] = 1.
        return voxel

    def __padding__(self, voxel, B):
        """
        补0
        :param voxel: 待填充体素
        :param B: 边长
        :return: 填充结果
        """
        pad = np.zeros(shape=[B, B, B, len(atoms_vector)])
        D, W, H, C = voxel.shape
        bias_D, bias_W, bias_H = int((B - D) / 2), int((B - W) / 2), int((B - H) / 2)
        pad[bias_D:D + bias_D][bias_W:W+bias_W][bias_H:H+bias_H] = voxel
        return pad


    def __xyz_range__(self):
        x_min = self.d[:, 0].min()
        x_max = self.d[:, 0].max()
        y_min = self.d[:, 1].min()
        y_max = self.d[:, 1].max()
        z_min = self.d[:, 2].min()
        z_max = self.d[:, 2].max()
        return x_min, y_min, z_min, x_max, y_max, z_max

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
        return 4 * np.pi * (r ** 2) * y

    def __wave100__(self, a, r):
        """
        计算内层电子分布情况
        :param a: 原子序数
        :param r: 径向距离
        :return:
        """
        y = np.exp(-(a * 2 * r) / a0) * np.sqrt(a ** 3 / (np.pi * (a0 ** 3)))
        return y ** 2

    def __wave200__(self, a, r):
        """
        计算外层电子分布情况
        :param a: 原子序数
        :param r: 径向距离
        :return:
        """
        y = np.exp(-(a * r) / a0) * np.sqrt(
            a ** 3 / (32 * np.pi * (a0 ** 3))) * (2 - a * 2 * r / a0)
        return y ** 2
