#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 14:24
# @Author  : Wang Jixin
# @File    : vis.py
# @Software: PyCharm



from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
#编码
from sklearn.preprocessing import label_binarize
from predict import test_y,r1,r2
# 绘制多分类ROC曲线


def plt_roc(n_classes,r):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test_y[:, i], r.detach().numpy()[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.figure(dpi=512)
    lw = 1
    colors = ['blue', 'red', 'green', 'yellow', 'black']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (AUC = {1:0.3f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()


# label_r1 = label_binarize(r1, classes=[0, 1, 2, 3, 4])
# print(label_r1)
# label_r2 = label_binarize(r2, classes=[0, 1, 2, 3, 4])
# print(label_r2)

label_test_y = label_binarize(test_y, classes=[0, 1, 2, 3, 4])


plt_roc(5,r1)

plt_roc(5,r2)