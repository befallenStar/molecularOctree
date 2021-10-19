# -*- encoding: utf-8 -*-
"""
input: log文件
根据train、valid、test三个阶段统计分析
不同阶段最小mae
不同阶段time
"""
import re


def load(path='./voxelnet.log'):
    tmp_path = './tmp.log'
    with open(path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            for p in ['train', 'valid', 'test']:
                if p in line:
                    lines.append(p + "\t")
            p, result = process(line)
            if 3 == len(result):
                lines.append(
                    'loss: {}, \tmae: {}, \ttime: {}\n'.format(result[0],
                                                               result[1],
                                                               result[2]))
        with open(tmp_path, 'w', encoding='utf-8') as tmp:
            tmp.writelines(lines)

    train_loss, valid_loss, test_loss = [], [], []
    train_mae, valid_mae, test_mae = [], [], []
    train_time, valid_time, test_time = 0, 0, 0
    train_iter, valid_iter, test_iter = 0, 0, 0
    with open(tmp_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p, result = process(line)
            if len(result) == 3:
                if p == 'train':
                    train_loss.append(result[0])
                    train_mae.append(result[1])
                    train_time += float(result[2])
                    train_iter += 1
                elif p == 'valid':
                    valid_loss.append(result[0])
                    valid_mae.append(result[1])
                    valid_time += float(result[2])
                    valid_iter += 1
                elif p == 'test':
                    test_loss.append(result[0])
                    test_mae.append(result[1])
                    test_time += float(result[2])
                    test_iter += 1
    print(min(train_mae))
    print(min(valid_mae))
    print(min(test_mae))
    print(train_time)


def process(line):
    p = re.match(r'[a-zA-Z]{4,6}', line)
    if p:
        p = p.group()
    pattern = re.compile(r'\d+\.\d{3,}')
    result = pattern.findall(line)
    return p, result


def main():
    load('./voxelnet2.0.log')


if __name__ == '__main__':
    main()
