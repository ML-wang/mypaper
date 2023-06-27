#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 9:59
# @Author  : Wang Jixin
# @File    : mlp.py
# @Software: PyCharm

import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
# from config import Config

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP,self).__init__()
        self.linear = nn.Sequential(
           nn.Linear(12,10),
           nn.ReLU(),
           nn.Dropout(),
           nn.Linear(10,8),
           nn.Dropout(),
           nn.ReLU(),
           nn.Linear(8,5),
           nn.Softmax()
        )

    def forward(self,x):
        out_put = self.linear(x)
        return out_put


if __name__ == '__main__':
    config = Config()
    mlp = MLP()
    output = mlp(config.smote_df)
    print(output)