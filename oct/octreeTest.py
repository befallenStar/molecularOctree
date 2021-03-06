# -*- encoding: utf-8 -*-

import numpy as np

from loader.loadData import *
from loader.pcd2voxel import *
from model.octreeNet import *
from oct.octree import *

MIN = 0.4


def backup():
    atomses, propertieses, _ = atoms_from_file('../data', 'qm9_10000.db')
    for i in trange(len(atomses)):
        atoms = atomses[i]
        properties = propertieses[i]
        prop = np.array(list(properties.values()))[:15]
        path = '../input32_3/{}.npz'.format(str(i + 1))
        pcd = atoms2pcd(atoms)
        voxelLoader = VoxelLoader(pcd)
        voxel = voxelLoader(sz=0.4, norm=3)
        np.savez_compressed(path, voxel=voxel, properties=prop)
        # data = torch.load('../inputs/{}.pth'.format(atoms.symbols))


def main():
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
    # np.save('../data/qm9_{}.npz'.format(trans), leaves_all=leaves_all,
    #         properties_all=properties_all)


if __name__ == "__main__":
    # main()
    backup()
