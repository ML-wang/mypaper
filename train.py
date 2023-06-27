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
from models.mlp import MLP
from models.DAE import DAE
from models.Improved_DAE import Imporved_DAE
from tqdm import tqdm
import argparse
from datasets import smote_x, smote_y
import time

model_time = time.strftime('%m-%d_%H-%M-%S', time.localtime())


def training(model, train_x, train_y, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epoch_loss = []
    beast_loss = 2
    for epoch in tqdm(range(1, epochs + 1)):
        optimizer.zero_grad()
        set_seed(42)
        pre_y = model(train_x)
        loss = criterion(pre_y, train_y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if loss < beast_loss:
            beast_loss = loss
            checkpoint = {
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if isinstance(model, (MLP, DAE, Imporved_DAE)):
                # assert model in ['mlp','dae','idae']
                save_path = f'./checkpoint/{type(model).__name__}_ckpt_{model_time}.pt'
                torch.save(checkpoint, save_path)
    return epoch_loss

    # elif isinstance(model, DAE):
    #     save_path = f'./checkpoint/DAE_ckpt_{model_time}.pt'
    #     torch.save(checkpoint, save_path)
    #     return ['dae',f'./checkpoint/DAE_ckpt_{model_time}.pt']
    # elif isinstance(model, Imporved_DAE):
    #     save_path = f'./checkpoint/IDAE_ckpt_{model_time}.pt'
    #     torch.save(checkpoint, save_path)
    #     return ['idae',f'./checkpoint/IDAE_ckpt_{model_time}.pt']


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
    set_seed(42)
    epochs_loss = training(model, smote_x, smote_y, 1500)
    print(epochs_loss)
    return epochs_loss


# if __name__ == '__main__':
mlp = MLP()
mlp_loss = main(mlp)
#
dae = DAE()
dae_loss = main(dae)
# print(model_name)
# print(model_path)

idae = Imporved_DAE()
idae_loss = main(idae)

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# parser.add_argument("--model_name", default='Improved_DAE', action="store_true", help="model_name")
# args = parser.parse_args()
# print(args.is_action)
