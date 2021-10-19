# -*- encoding: utf-8 -*-
import operator
from queue import Queue

import numpy as np
import torch

from .octreeNode import *

MIN = 0.4


class Octree:
    """
    八叉树类
    """

    def __init__(self, voxel):
        """
        初始化空八叉树
        """
        self.root = None
        self.voxel = voxel
        self.pcd = []
        self.voxel2pcd()
        self.pos = self.get_center()

    def __str__(self):
        return str(self.root)

    def voxel2pcd(self):
        """
        将体素转化为点云，每个点包含[x, y, z, [atom1, atom2, ..., atomN]]
        """
        x, y, z, _ = np.where(self.voxel != 0)
        dup = []
        for i in range(1, len(x)):
            if operator.eq((x[i - 1], y[i - 1], z[i - 1]), (x[i], y[i], z[i])):
                dup.append(i - 1)
        x = np.delete(x, dup)
        y = np.delete(y, dup)
        z = np.delete(z, dup)

        for i in range(len(x)):
            self.pcd.append(
                [x[i], y[i], z[i], self.voxel[x[i]][y[i]][z[i]].tolist()])

    def exists(self, node: OctreeNode):
        """
        判断体素格内是否存在原子，以及是否达到最大深度
        :return: 存在返回True
        """
        if node.width <= MIN:
            return False
        return True

    def create(self):
        """
        构建八叉树
        """
        self.root = OctreeNode(self.pos, None, 6.4, self.pcd)
        q = Queue()
        q.put(self.root)
        while not q.empty():
            node = q.get()
            if self.exists(node):
                node.split()
                if node.children:
                    for node in node.children:
                        q.put(node)

    def get_leaves(self):
        """
        获取八叉树所有最底层叶子节点
        :return:
        """
        leaves = []
        q = Queue()
        q.put(self.root)
        while not q.empty():
            node = q.get()
            if node.width <= MIN:
                for d in node.data:
                    leaves.append(d[3])
            else:
                for child in node.children:
                    q.put(child)
        return leaves

    def get_center(self):
        """
        获取体素中心位置
        :return:
        """
        D, W, H, _ = self.voxel.shape
        return D // 2, W // 2, H // 2

    def full_leaves_count(self):
        """
        统计八叉树最底层叶子节点中包含原子的节点个数
        :return:
        """
        cnt = 0
        q = Queue()
        q.put(self.root)
        while not q.empty():
            node = q.get()
            if node.width == MIN:
                for d in node.data:
                    if sum(d[3]) > 0:
                        cnt += 1
            else:
                if not node.isLeaf:
                    for child in node.children:
                        q.put(child)
        return cnt

    def to_voxel(self):
        """
        将八叉树转成网络输入，5 * 128 * 128 * 128
        :return:
        """
        q = Queue()
        q.put(self.root)
        center = (6.4, 6.4, 6.4)
        L = int(12.8 / MIN)
        data = torch.zeros(L, L, L, 5)
        bias_x, bias_y, bias_z = [center[i] - self.root.center[i] for i in
                                  range(3)]
        while not q.empty():
            node = q.get()
            if node.width == MIN:
                x, y, z = node.center
                data[round((x + bias_x) / MIN)][round((y + bias_y) / MIN)][
                    round((z + bias_z) / MIN)] += torch.tensor(node.data[0][3])
            else:
                if not node.isLeaf:
                    for child in node.children:
                        q.put(child)
        return data
