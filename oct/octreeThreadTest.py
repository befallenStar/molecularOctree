# -*- encoding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from loader.loadData import *
from loader.pcd2voxel import *
from oct.octree import *
import os
from model.octreeNet import *
import threading


def backup():
    atomses, propertieses, datas = atoms_from_file('../data', 'qm9.db')
    for atoms in tqdm(atomses):
        pcd = atoms2pcd(atoms)
        voxelLoader = VoxelLoader(pcd)
        trans = 'gauss'
        voxel = voxelLoader(transformer=trans)
        octree = Octree(voxel)
        octree.create()
        print(octree)
        leaves = octree.get_leaves()
        print(leaves)
        print(octree.full_leaves_count())


def once():
    leaves_all = []
    properties_all = []
    trans = 'octree'
    for root, _, filenames in os.walk('../npz'):
        for filename in filenames:
            if trans in filename:
                print(filename)
                data = np.load(root + '/' + filename)
                # print(data['voxel'].shape)
                # print(data['properties'])
                voxel = data['voxel']
                properties = data['properties']
                octree = Octree(voxel)
                octree.create()
                # print(octree)
                leaves = octree.get_leaves()
                # print(leaves)
                # print(octree.full_leaves_count())
                # octreeNet = OctreeNet(5)
                # print(np.array(leaves))
                # result = octreeNet(np.array(leaves))
                # print(result)
                leaves_all.append(leaves)
                properties_all.append(properties)
    leaves_all = np.array(leaves_all)
    properties_all = np.array(properties_all)
    np.save('../data/qm9_{}.npz'.format(trans), leaves_all=leaves_all,
            properties_all=properties_all)


def main():
    leaves_all = []
    properties_all = []
    trans = 'octree'
    for root, _, filenames in os.walk('../npz'):
        indexes = np.arange()
    leaves_all = np.array(leaves_all)
    properties_all = np.array(properties_all)
    np.save('../data/qm9_{}.npz'.format(trans), leaves_all=leaves_all,
            properties_all=properties_all)


if __name__ == "__main__":
    main()
    # backup()
