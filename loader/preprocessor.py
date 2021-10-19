# -*- encoding: utf-8 -*-

from tqdm import tqdm, trange

from loader.loadData import *
from loader.pcd2voxel import *
from oct.octree import *


def main():
    atomses, propertieses, datas = atoms_from_file('../data', 'qm9_10000.db')
    for i in trange(len(atomses)):
        atoms = atomses[i]
        properties = propertieses[i]
        # for atoms, properties in tqdm(zip(atomses, propertieses)):
        #     print(atoms.symbols)
        pcd = atoms2pcd(atoms)
        voxelLoader = VoxelLoader(pcd)
        for trans in ['origin', 'gauss', 'wave']:
            voxel = voxelLoader(transformer=trans)
            octree = Octree(voxel)
            octree.create()
            # print(octree)
            leaves = octree.get_leaves()
            # print(np.array(list(properties.values())))
            path = '../npz/{}/{}_{}.npz'.format(trans, str(i + 1), trans)
            np.savez(path, leaves=leaves,
                     properties=np.array(list(properties.values())))


if __name__ == "__main__":
    main()
