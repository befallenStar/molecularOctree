# -*- encoding: utf-8 -*-
from .octreeNode import *
from queue import LifoQueue


class Octree:
    """
    八叉树类
    """

    def __init__(self, pcd, center):
        """
        初始化空八叉树
        """
        self.root = None
        self.pcd = pcd
        self.pos=center

    def __str__(self):
        return str(self.root)

    def exists(self, node: OctreeNode):
        """
        判断体素格内是否存在原子
        :return: 存在返回True
        """
        pos = node.center
        w = node.width/2
        for point in self.pcd:
            if pos[0]-w <= point[0] <= pos[0] + w and pos[1]-w <= point[1] <= pos[
                1] + w and pos[2]-w <= point[2] <= pos[2] + w:
                return True
        return False

    def create(self):
        """
        构建八叉树
        """
        self.root = OctreeNode(self.pos, None, None, None, 16, self.pcd)
        q = LifoQueue()
        q.put(self.root)
        while not q.empty():
            node = q.get()
            if self.exists(node):
                node.split()
                if node.firstChild is not None:
                    q.put(node.firstChild)
                    node = node.firstChild
                    while node.nextSibling:
                        q.put(node.nextSibling)
                        node = node.nextSibling

    def get_leaves(self):
        """
        获取八叉树所有最底层叶子节点
        :return:
        """
        leaves = []

        return leaves