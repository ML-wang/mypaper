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
from models.prepare import AE
from datasets import split_train_test,process_data
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


def predict(model,model_path, test_x):
    set_seed(42)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(test_x)
    # return output.detach().numpy()
    # return np.argmax(output[0].detach().numpy(), axis=1)
    if isinstance(model,AE):
        return output[0]
    elif isinstance(model,MLP):
        return output


# if __name__ == '__main__':
data = process_data()
smote_df,test_df = split_train_test(data)
test_x,test_y = torch.tensor(test_df.iloc[:,:-1].values,dtype=torch.float),\
                  torch.tensor(test_df.iloc[:,-1].values)
mlp = MLP()
ae = AE()
# r1 = predict(ae,'./checkpoint/AE_ckpt_train_model2.pt',test_x)
r1 = predict(ae,'./checkpoint/AE_ckpt_train_model4.pt',test_x)
r2 = predict(mlp,'./checkpoint/MLP_ckpt_train_model.pt',test_x)
# print(r2)
#
#
#
# print(r2)
print(test_y)
# label_r1 = label_binarize(r1, classes=[0, 1, 2, 3, 4])
# label_test_y = label_binarize(test_y, classes=[0, 1, 2, 3, 4])

print("*"*20)
print(accuracy_score(r1,test_y))
print(f1_score(np.argmax(r1.detach().numpy(), axis=1),test_y,average='weighted'))
print(precision_score(np.argmax(r1.detach().numpy(), axis=1), test_y, average='weighted'))
print(recall_score(np.argmax(r1.detach().numpy(), axis=1), test_y, average='weighted'))

print("*" * 20)
print(accuracy_score(np.argmax(r2.detach().numpy(), axis=1), test_y))
print(f1_score(np.argmax(r2.detach().numpy(), axis=1), test_y, average='weighted'))
print(precision_score(np.argmax(r2.detach().numpy(), axis=1), test_y, average='weighted'))
print(recall_score(np.argmax(r2.detach().numpy(), axis=1), test_y, average='weighted'))

