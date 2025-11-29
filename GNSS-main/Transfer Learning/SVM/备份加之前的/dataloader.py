import torch
from torch.utils.data import Dataset
import numpy as np

class SeqDataset(Dataset):
    """
    用于时间序列的自定义 PyTorch Dataset。
    
    它被设计为可以同时处理:
    1. 有标签数据 (X, y) - 用于源域 (Source Domain)
    2. 无标签数据 (X) - 用于目标域 (Target Domain)
    """
    def __init__(self, X, y=None):
        """
        初始化 Dataset。
        
        参数:
        - X: (Numpy Array) 特征数据。
             [你传入的形状将是 (142005, 5, 4) 或 (2517, 5, 4)]
        - y: (Numpy Array, 可选) 标签数据。
             [你传入的形状将是 (142005,)]
        """
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.y_exists = y is not None
        
        # 如果提供了标签 y，也将其转换为 Tensor
        if self.y_exists:
            self.y = torch.from_numpy(y).long()
        else:
            # 如果没有y，创建一个虚拟的y，只是为了让代码能跑通
            # 这样 __getitem__ 总是可以返回两个值
            self.y = torch.zeros(len(self.X), dtype=torch.long)
            
    def __len__(self):
        """返回数据集中的样本总数 (例如 142005)"""
        return len(self.X)
    
    def __getitem__(self, i):
        """
        根据索引 i 获取一个样本。
        
        返回: (数据, 标签)
        - 对于源域: (self.X[i], self.y[i]) -> (一个 (5,4) 的张量, 一个标签)
        - 对于目标域: (self.X[i], 0)     -> (一个 (5,4) 的张量, 一个占位符0)
        """
        return self.X[i], self.y[i]