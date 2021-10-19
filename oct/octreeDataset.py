# -*- encoding: utf-8 -*-
import torch
import torch.utils.data.dataset as Dataset
import numpy as np


class OctreeDataset(Dataset.Dataset):
    def __init__(self, data: torch.Tensor, label: torch.Tensor):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

