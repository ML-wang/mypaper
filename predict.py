#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 9:59
# @Author  : Wang Jixin
# @File    : predict.py
# @Software: PyCharm

import torch
from train import set_seed
import numpy as np
from models.mlp import MLP
from models.DAE import DAE
from models.Improved_DAE import Imporved_DAE
from datasets import test_x,test_y,config
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from train import model_name,model_path


def load_model_predict(model_name,model_path, test_x):
    set_seed(42)
    if model_name == 'mlp':
        model = MLP()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        output = model(test_x)
    return output


def models_metrics(label_result:torch.tensor,test_y):
    acc = accuracy_score(label_result, test_y)
    f1 = f1_score(label_result, test_y, average='weighted')
    precision = precision_score(label_result, test_y, average='weighted')
    recall = recall_score(label_result, test_y, average='weighted')
    model_metrics = [acc,f1,precision,recall]
    model_metrics_name = ['acc', 'f1', 'precision', 'recall']
    for met_name,met in zip(model_metrics_name,model_metrics):
        print('{}\t{:.4f}'.format(met_name,met))
    print("*" * 20)

if __name__ == '__main__':

    # mlp = MLP()
    # result_mlp = predict(mlp, 'save_models/MLP_ckpt_06-27_10-34-37.pt', test_x)
    # label_result_mlp = torch.argmax(result_mlp,dim=1)
    # models_metrics(label_result_mlp, test_y)

    dae = DAE()
    result_dae = predict(dae,'./checkpoint/DAE_ckpt_06-27_10-43-50.pt',test_x)
    label_result_dae = torch.argmax(result_dae,dim=1)
    models_metrics(label_result_dae, test_y)


    # idae = Imporved_DAE()
    # result_idae = predict(idae, 'save_models/IDAE_ckpt_06-27_10-30-35.pt', test_x)
    # label_result_idae = torch.argmax(result_idae,dim=1)
    # models_metrics(label_result_idae, test_y)







# print("*"*20)
# print(accuracy_score(r1,test_y))
# print(f1_score(np.argmax(r1.detach().numpy(), axis=1),test_y,average='weighted'))
# print(precision_score(np.argmax(r1.detach().numpy(), axis=1), test_y, average='weighted'))
# print(recall_score(np.argmax(r1.detach().numpy(), axis=1), test_y, average='weighted'))
#
# print("*" * 20)
# print(accuracy_score(np.argmax(r2.detach().numpy(), axis=1), test_y))
# print(f1_score(np.argmax(r2.detach().numpy(), axis=1), test_y, average='weighted'))
# print(precision_score(np.argmax(r2.detach().numpy(), axis=1), test_y, average='weighted'))
# print(recall_score(np.argmax(r2.detach().numpy(), axis=1), test_y, average='weighted'))

