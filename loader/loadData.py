# -*- encoding: utf-8 -*-
import os

from ase import Atoms
from ase.db import connect


def atoms_from_file(path='..\\data', ase_db='ase.db'):
    """
    从db文件中读取分子信息
    :param path: db文件路径
    :param ase_db: db文件名称
    :return: 返回分子列表，分子属性以及分子数据
    """
    ase_path = os.path.join(path, ase_db)
    conn = connect(ase_path)
    atomses, propertieses, datas = [], [], []
    for row in conn.select():
        atoms = row.toatoms()
        properties = row.key_value_pairs
        data = row.data
        atomses.append(atoms)
        propertieses.append(properties)
        datas.append(data)
    return atomses, propertieses, datas


def atoms2pcd(atoms: Atoms):
    """
    将分子转化成为点云数据
    :param atoms: Atoms类型分子
    :return: 点云数据
    """
    pointcloud = []
    for step, number in enumerate(atoms.numbers):
        point = atoms.positions[step].tolist()
        # point.extend(colors[number])
        point.append(number)
        pointcloud.append(point)
    return pointcloud
