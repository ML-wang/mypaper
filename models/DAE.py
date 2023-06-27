#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 16:15
# @Author  : Wang Jixin
# @File    : DAE.py
# @Software: PyCharm

import torch
import torch.nn as nn
from datasets import smote_x
import warnings
warnings.filterwarnings('ignore')
class DAE(nn.Module):
    def __init__(self) -> None:
        super(DAE,self).__init__()
        self.encoder = nn.Sequential(
           nn.Linear(12,10),
           nn.ReLU(),
           nn.Dropout(),
           nn.Linear(10,10),
           nn.Dropout(),
           nn.ReLU(),
           nn.Linear(10,6),
           # nn.ReLU()
        )

        self.decoder = nn.Sequential(
           nn.Linear(6,10),
           nn.ReLU(),
            nn.Dropout(),
           nn.Linear(10,10),
           nn.ReLU(),
            nn.Dropout(),
           nn.Linear(10,12),
        )
        self.reset_weight

        self.mlp = nn.Sequential(
           nn.Linear(12,10),
           nn.ReLU(),
           nn.Dropout(),
           nn.Linear(10,8),
           nn.Dropout(),
           nn.ReLU(),
           nn.Linear(8,5),
           nn.Softmax()
           )

    def forward(self,train_x):
        self.train_x = train_x
        noise = nn.init.normal(torch.empty(self.train_x.shape))
        input = noise*0.01 + self.train_x
        encoder = self.encoder(input)
        decoder = self.decoder(encoder)
        # print(decoder)
        # print(self.train_x)
        # out_put = self.mlp(torch.cat((decoder,self.train_x),1))
        out_put = self.mlp(decoder)
        return out_put

    def reset_weight(self):
        for module in self.encoder:
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.normal_(module.bias,0,1)
        for module in self.decoder:
            # print(type(module.bias))
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.normal_(module.bias,0,1)


if __name__ == '__main__':
    dae = DAE()
    output = dae(smote_x)
    print(output)