# -*- encoding: utf-8 -*-
from loader.loadData import *
from loader.pcd2voxel import *
import open3d as o3d
import numpy as np
from oct.octree import *
from oct.octreeNode import *


def main():
    atomses, propertieses, datas = atoms_from_file('./data', 'qm9.db')
    for atoms in atomses:
        pcd = atoms2pcd(atoms)
        voxelLoader = VoxelLoader(pcd)
        # voxel = voxelLoader(transformer='gauss')
        voxel = voxelLoader(transformer='octree')
        x, y, z, _ = np.where(voxel != 0)
        data = []
        for i in range(len(x)):
            d = [x[i], y[i], z[i]]
            d.extend(voxel[x[i]][y[i]][z[i]])
            data.append(d)
        D, W, H, _ = voxel.shape
        octree = Octree(data, (D // 2, W // 2, H // 2))
        octree.create()
        print(octree)


if __name__ == '__main__':
    main()
