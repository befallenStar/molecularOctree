# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
from ase import Atoms
from ase.db import connect
from matplotlib.pyplot import MultipleLocator
from tqdm import trange

from oct.octree import *


def atoms_from_file(path='./data/qm9_10000.db'):
    conn = connect(path)
    atomses = []
    for row in conn.select():
        atoms = row.toatoms()
        atomses.append(atoms)
    return atomses


def atoms2pcd(atoms: Atoms):
    pointcloud = []
    for step, number in enumerate(atoms.numbers):
        point = atoms.positions[step].tolist()
        # point.extend(colors[number])
        point.append(number)
        pointcloud.append(point)
    return pointcloud


def wave100(z, x):
    a0 = 5.29
    y = np.exp(-(z * 4 * x) / a0) * np.sqrt(z ** 3 / (np.pi * (a0 ** 3)))
    return y


def wave200(z, x):
    a0 = 5.29
    sigma = z * 4 * x / a0
    y = np.exp(-sigma / 2) * np.sqrt(z ** 3 / (32 * np.pi * (a0 ** 3))) * (
            2 - sigma)
    return y


def wave300(z, x):
    a0 = 5.29
    sigma = z * 4 * x / a0
    y = np.exp(-sigma / 3) * np.sqrt(
        z ** 3 / (81 * 81 * 3 * np.pi * (a0 ** 3))) * (
                27 - 18 * sigma + 2 * sigma ** 2)
    return y


def wave(z, x):
    y = wave100(z, x)
    y += 4 * wave200(z, x)
    y += 1 * wave300(z, x)
    return 4 * np.pi * ((2 * x) ** 2) * (y ** 2)


def gauss(x, sigma):
    return np.exp(-2 * x / (2 * sigma ** 2))


def main():
    atomses = atoms_from_file(r'./data/qm9_10000(所用数据库).db')
    for idx in trange(len(atomses)):
        if idx in [255, 258, 284, 519, 288, 592]:
            print(idx)
            print(atomses[idx].symbols)
        if idx > 600:
            break
        continue
        atoms = atomses[idx]
        if len(atoms.numbers) < 14 or len(set(atoms.numbers)) < 4:
            continue
        # if os.path.exists('./img/origin/{}.png'.format(idx)) and os.path.exists(
        #         './img/gauss/{}.png'.format(idx)) and os.path.exists(
        #     './img/wave/{}.png'.format(idx)):
        #     continue
        pcd = atoms2pcd(atoms)
        pcd = np.array(pcd)
        # pca = PCA(n_components=3)
        # pca.fit(pcd[:, :3])
        # pcd_pca = pcd.copy()
        # pcd_pca[:, :3] = pca.transform(pcd[:, :3])
        colors = [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1],
                  [0, 1, 0, 1]]
        pointcloud = []
        pointcloud_diff = []
        cmap_origin = []
        cmap_gauss = []
        cmap_wave = []
        scale = 2
        for point in pcd:
            # 原子序数
            a = int(point[3])
            if a == 1:
                continue
            # 核心原子颜色
            color = colors[0] if a == 1 else colors[a - 5]
            pointcloud.append(
                [point[0] * scale, point[1] * scale, point[2] * scale])
            pointcloud_diff.append(
                [point[0] * scale, point[1] * scale, point[2] * scale])
            cmap_origin.append(color)
            cmap_gauss.append(color)
            cmap_wave.append(color)
            r = 3
            for i in np.arange(-r, r + 0.1, 0.1):
                for j in np.arange(-r, r + 0.1, 0.1):
                    for k in np.arange(-r, r + 0.1, 0.1):
                        # 循环向外扩散
                        if i ** 2 + j ** 2 + k ** 2 > 9:
                            continue
                        pointcloud_diff.append(
                            [point[0] * scale + i, point[1] * scale + j,
                             point[2] * scale + k])
                        # wave_color = color.copy()
                        # wave_color[3] *= 10 * wave(a, (
                        #         i ** 2 + j ** 2 + k ** 2) ** 0.5)
                        wa = wave(a + 10, (i ** 2 + j ** 2 + k ** 2) ** 0.5)
                        wa = wa ** (3 / 8)
                        wave_color = []
                        for w in range(3):
                            if color[w] == 1:
                                wave_color.append(1)
                            else:
                                wave_color.append(1 - 3 * wa)
                        wave_color.append(wa)

                        gau = gauss(i ** 2 + j ** 2 + k ** 2, 1)
                        gau = gau ** 0.75
                        gauss_color = []
                        for c in range(3):
                            if color[c] == 1:
                                gauss_color.append(1)
                            else:
                                gauss_color.append(1 - gau)
                        gauss_color.append(gau)

                        cmap_wave.append(wave_color)
                        cmap_gauss.append(gauss_color)
        cmap_wave = np.array(cmap_wave, dtype='float32')
        cmin = np.min(cmap_wave[:, 0: 3])
        cmap_wave[:, 0: 3] -= cmin
        cmax = np.max(cmap_wave)
        cmap_wave = cmap_wave / cmax
        cmap_wave[:, 3] *= 0.75 # 目前最佳
        # cmap_wave *= 0.6

        cmap_gauss = np.array(cmap_gauss, dtype='float32')
        gmax = np.max(cmap_gauss[:, 0: 3])
        cmap_gauss[:, 0: 3] = cmap_gauss[:, 0: 3] / gmax

        pointcloud = np.array(pointcloud)
        pointcloud_diff = np.array(pointcloud_diff)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, projection='3d')
        # ax1.scatter(pointcloud[:, 0], pointcloud[:, 1],
        #             pointcloud[:, 2], s=2,
        #             c=cmap_origin)
        # ax1.xaxis.set_major_locator(MultipleLocator(1))
        # ax1.yaxis.set_major_locator(MultipleLocator(1))
        # ax1.zaxis.set_major_locator(MultipleLocator(1))
        # ax1.set_zticks([-3, -2, -1, 0, 1, 2, 3])
        # plt.xlim(-4.5, 4.5)
        # plt.ylim(-4.5, 4.5)
        # plt.axis('off')
        # plt.savefig('./img/origin/{}.png'.format(idx))
        # plt.show()
        # fig = plt.figure()
        # ax2 = fig.add_subplot(111, projection='3d')
        # ax2.scatter(pointcloud_diff[:, 0], pointcloud_diff[:, 1],
        #             pointcloud_diff[:, 2], s=2,
        #             c=cmap_gauss)
        # ax2.xaxis.set_major_locator(MultipleLocator(1))
        # ax2.yaxis.set_major_locator(MultipleLocator(1))
        # ax2.zaxis.set_major_locator(MultipleLocator(1))
        # ax2.set_zticks([-3, -2, -1, 0, 1, 2, 3])
        # plt.xlim(-4.5, 4.5)
        # plt.ylim(-4.5, 4.5)
        # plt.axis('off')
        # plt.savefig('./img/gauss/{}.png'.format(idx))
        # plt.show()
        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection='3d')
        ax3.scatter(pointcloud_diff[:, 0], pointcloud_diff[:, 1],
                    pointcloud_diff[:, 2], s=2,
                    c=cmap_wave)
        ax3.xaxis.set_major_locator(MultipleLocator(1))
        ax3.yaxis.set_major_locator(MultipleLocator(1))
        ax3.zaxis.set_major_locator(MultipleLocator(1))
        ax3.set_zticks([-3, -2, -1, 0, 1, 2, 3])
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        plt.axis('off')
        # plt.savefig('./img/wave/{}.png'.format(idx))
        plt.show()


if __name__ == '__main__':
    main()
