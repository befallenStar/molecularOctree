# -*- encoding: utf-8 -*-
MIN = 0.4


class OctreeNode:
    """
    八叉树节点类，包含节点基本信息
    """

    def __init__(self, pos=(8, 8, 8), p=None, w=6.4, d=None):
        """
        初始化八叉树节点
        """
        self.center = pos
        self.parent = p
        self.children = []
        self.width = w
        self.data = d
        self.isLeaf = True

    def __str__(self):
        s = 'pos: ' + str(self.center) + ' data: ' + str(
            self.data) + ' isLeaf:' + str(self.isLeaf)
        for node in self.children:
            s += '\n' + 'parent center: ' + str(
                node.parent.center) + ' parent width: ' + str(
                node.parent.width) + ' ' + str(node)
        return s + '\n'

    def vertexes(self, node):
        """
        获取节点体素的八个顶点坐标
        :param node: 当前节点
        :return: 包含八个坐标的列表
        """
        vertex = []
        pos = node.center
        w = node.width / 2
        pos_f = round((pos[0] + w) * 10) / 10.0
        pos_b = round((pos[0] - w) * 10) / 10.0
        pos_r = round((pos[1] + w) * 10) / 10.0
        pos_l = round((pos[1] - w) * 10) / 10.0
        pos_t = round((pos[2] + w) * 10) / 10.0
        pos_d = round((pos[2] - w) * 10) / 10.0

        vertex.append((pos_f, pos_r, pos_d))
        vertex.append((pos_f, pos_r, pos_t))
        vertex.append((pos_f, pos_l, pos_d))
        vertex.append((pos_f, pos_l, pos_t))
        vertex.append((pos_b, pos_r, pos_d))
        vertex.append((pos_b, pos_r, pos_t))
        vertex.append((pos_b, pos_l, pos_d))
        vertex.append((pos_b, pos_l, pos_t))
        return vertex

    def split(self):
        """
        对包含原子的体素进行一次划分，生成八个子节点，添加到self.children
        """
        if self.width > MIN:
            vertexes = self.vertexes(self)
            for i in range(len(vertexes)):
                child = OctreeNode(vertexes[i], self, self.width / 2, [])
                for d in self.data:
                    if self.in_bound(child, d):
                        child.data.append(d)
                if not child.data:
                    child.data.append([0, 0, 0, [0, 0, 0, 0, 0]])
                self.children.append(child)
            self.isLeaf = False

    def in_bound(self, node, data):
        """
        判断原子是否存在在节点范围内
        :param node: 节点
        :param data: 原子数据，点云
        :return: 原子在节点内部返回True
        """
        pos = node.center
        w = node.width
        for _ in data:
            if pos[0] - w <= data[0] < pos[0] + w and pos[1] - w <= data[1] < \
                    pos[1] + w and pos[2] - w <= data[2] < pos[2] + w:
                return True
        return False
