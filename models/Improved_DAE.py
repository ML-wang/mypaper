#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 16:15
# @Author  : Wang Jixin
# @File    : DAE.py
# @Software: PyCharm

import torch
import torch.nn as nn
class Imporved_DAE(nn.Module):
    def __init__(self) -> None:
        super(Imporved_DAE,self).__init__()
        self.linear1 = nn.Linear(12,10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 8)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(8, 6)
        self.relu3 = nn.ReLU()


        self.linear4 = nn.Linear(6,8)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(8, 10)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(10, 12)

        self.reset_weight

        self.mlp = nn.Sequential(
           nn.Linear(2*12,10),
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
        # if isinstance(train_x,np.ndarray):
        #     self.train_x = torch.tensor(train_x,dtype=torch.float)
        # else:
        #     self.train_x = torch.tensor(train_x.values,dtype=torch.float)
        noise = nn.init.normal(torch.empty(self.train_x.shape))
        input = noise*0.01 + self.train_x
        # print(input)
        encoder1 = self.relu1(self.linear1(input))  #12dao  10
        encoder2 = self.relu2(self.linear2(encoder1))  #10dao 8
        encoder3 = self.relu3(self.linear3(encoder2))  # 8dao6

        decoder1 = self.relu4(self.linear4(encoder3))  # 6
        decoder2 = self.relu5(self.linear5(decoder1 +encoder2))  #8到10
        decoder3 = self.linear6(decoder2+encoder1)  # 10到12
        # print(decoder)
        # print(self.train_x)
        out_put = self.mlp(torch.cat((decoder3,self.train_x),1))
        # out_put = self.mlp(decoder + self.train_x)
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