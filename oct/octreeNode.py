# -*- encoding: utf-8 -*-
MIN = 4


class OctreeNode:
    """
    八叉树节点类，包含节点基本信息
    """

    def __init__(self, pos=(8, 8, 8), p=None, s=None, c=None, w=16.0, d=None):
        """
        初始化八叉树节点
        """
        if d is None:
            d = []
        self.center = pos
        self.parent = p
        self.nextSibling = s
        self.firstChild = c
        self.width = w
        self.data = d
        self.isLeaf = True

    def __str__(self):
        s = 'pos: ' + str(self.center) + ' data: ' + str(
            self.data) + ' isLeaf:' + str(self.isLeaf)
        node = self.firstChild
        if node:
            s += '\n' + 'parent center: ' + str(
                node.parent.center) + ' parent width: ' + str(
                node.parent.width) + ' ' + str(node)
            while node.nextSibling:
                s += 'parent center: ' + str(
                    node.nextSibling.parent.center) + ' parent width: ' + str(
                    node.parent.width) + ' ' + str(node.nextSibling)
                node = node.nextSibling
        return s + '\n'

    def vertexes(self, node):
        vertex = []
        pos = node.center
        w = node.width / 4
        vertex.append((pos[0] + w, pos[1] + w, pos[2] - w))
        vertex.append((pos[0] + w, pos[1] - w, pos[2] + w))
        vertex.append((pos[0] - w, pos[1] + w, pos[2] + w))
        vertex.append((pos[0] + w, pos[1] - w, pos[2] - w))
        vertex.append((pos[0] - w, pos[1] - w, pos[2] + w))
        vertex.append((pos[0] - w, pos[1] + w, pos[2] - w))
        vertex.append((pos[0] + w, pos[1] + w, pos[2] + w))
        vertex.append((pos[0] - w, pos[1] - w, pos[2] - w))
        return vertex

    def split(self):
        """
        对包含原子的体素进行一次划分，生成八个子节点，
        self.firstChild指向首个子节点，
        每个子节点的nextSibling指向下一个子节点，
        最后一个子节点的nextSibling为None
        :return:
        """
        if self.width > MIN and len(self.data)>0:
            vertexes = self.vertexes(self)
            child = OctreeNode(vertexes[0], self, None, None,
                               self.width / 2, None)
            for d in self.data:
                if self.in_bound(child, d):
                    child.data.append(d)
            self.firstChild = child
            self.isLeaf = False
            for i in range(1, len(vertexes)):
                sibling = OctreeNode(vertexes[i], self, None, None, child.width,
                                     None)
                for d in self.data:
                    if self.in_bound(sibling, d):
                        sibling.data.append(d)
                child.nextSibling = sibling
                child = sibling

    def in_bound(self, node, data):
        """
        判断原子是否存在在节点范围内
        :param node: 节点
        :param data: 原子数据
        :return: 原子在节点内部返回True
        """
        pos = node.center
        w = node.width / 2
        return pos[0] - w <= data[0] < pos[0] + w and pos[1] - w <= data[1] < \
               pos[1] + w and pos[2] - w <= data[2] < pos[2] + w

