#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 10:06
# @Author  : Wang Jixin
# @File    : train.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torch import optim
import random
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datasets import process_data,split_train_test
from models.mlp import MLP
from models.prepare import AE
import time
from tqdm import tqdm
import argparse


def training(model,train_x,train_y,epochs):
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epoch_loss = []
    beast_loss = 1.2
    for epoch in tqdm(range(1,epochs+1)):
        optimizer.zero_grad()
        set_seed(42)
        pre_y = model(train_x)
        loss = criterion(pre_y,train_y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss)
        # print(loss)
        if loss < beast_loss:
            beast_loss = loss
            checkpoint = {
                'epoch': epoch,
                'loss':loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if isinstance(model, MLP):
                torch.save(checkpoint, './checkpoint/MLP_ckpt_train_model2.pt')
            elif isinstance(model, AE):
                torch.save(checkpoint, './checkpoint/AE_ckpt_train_model4.pt')



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(model):
    data = process_data()
    smote_df,test_df = split_train_test(data)
    train_x,train_y = torch.tensor(smote_df.iloc[:,:-1].values,dtype=torch.float),\
                      torch.tensor(smote_df.iloc[:,-1].values)


    set_seed(42)
    training(model,train_x,train_y,1500)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
    args = parser.parse_args()
    print(args.is_action)

    model_time = time.time()
    ae = AE()
    # mlp = MLP()
    main(ae)
    # main(mlp)