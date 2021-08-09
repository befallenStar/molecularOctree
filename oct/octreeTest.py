# -*- encoding: utf-8 -*-
from octreeNode import OctreeNode
from octree import Octree


def main():
    pcd = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [1, 2, 5],
        [4, 4, 4],
        [6, 5, 4],
        [7, 6, 5],
        [1, 3, 5],
        [2, 4, 6],
        [3, 6, 9],
        [2, 3, 2]
    ]
    octree = Octree(pcd)
    octree.create()
    print(octree)


if __name__ == "__main__":
    main()
