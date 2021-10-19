# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold


def train():
    """
    网络训练
    :return: 测试结果
    """
    # 加载数据
    data_all = np.load('./data/qm9.npz')
    leaves_all = data_all['leaves_all']
    properties_all = data_all['properties_all']
    # 划分训练集、验证集、测试集

    # 初始化网络
    # 训练开始
    # 计算测试集误差
    # 返回结果



def main():
    result = train()
    print(result)


if __name__ == '__main__':
    main()