# -*- encoding: utf-8 -*-
from loader.loadData import *
from loader.pcd2voxel import *
import open3d as o3d
import numpy as np
from oct.octree import *
from oct.octreeNode import *
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import operator

'''
h   黑
c   白
n   蓝
o   红
f   绿
'''
atomColors = [[0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
colorMove = [[0, 0, 0], [-1, -1, -1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]


def main():
    atomses, propertieses, datas = atoms_from_file('./data', 'qm9.db')
    for atoms in tqdm(atomses):
        pcd = atoms2pcd(atoms)
        voxelLoader = VoxelLoader(pcd)
        trans = 'gauss'
        scale = 0.75
        voxel = voxelLoader(transformer=trans)
        x, y, z, d = np.where(voxel != 0)
        c = np.zeros((len(x), 3))
        for j in range(len(x)):
            grid = voxel[x[j]][y[j]][z[j]]
            cnt = np.count_nonzero(grid)
            k = d[j]
            c[j][0] += (scale * (1 - grid[k]) * colorMove[k][0] + atomColors[k][
                0]) / cnt
            c[j][1] += (scale * (1 - grid[k]) * colorMove[k][1] + atomColors[k][
                1]) / cnt
            c[j][2] += (scale * (1 - grid[k]) * colorMove[k][2] + atomColors[k][
                2]) / cnt

        # points = [(x[i], y[i], z[i]) for i in range(len(x))]
        # vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        dup = []
        for i in range(1, len(x)):
            if operator.eq((x[i - 1], y[i - 1], z[i - 1]), (x[i], y[i], z[i])):
                dup.append(i - 1)
        x = np.delete(x, dup)
        y = np.delete(y, dup)
        z = np.delete(z, dup)
        for i in dup:
            c[i + 1] += c[i]
        c = np.delete(c, dup, 0)

        # 颜色可能会超出范围，需要进行归一
        # c /= np.max(c)
        # c =1/(1+np.exp(-c))
        c *= 255

        points = [(x[i], y[i], z[i], c[i][0], c[i][1], c[i][2]) for i in
                  range(len(c))]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('red', 'uint8'), ('green', 'uint8'),
                                         ('blue', 'uint8')])
        elPoints = PlyElement.describe(vertex, 'vertex')
        # elColor = PlyElement.describe(colors, 'color')
        PlyData([elPoints]).write(
            './ply/{}_{}.ply'.format(atoms.symbols, trans))


if __name__ == '__main__':
    main()
