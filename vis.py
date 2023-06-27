#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 14:24
# @Author  : Wang Jixin
# @File    : vis.py
# @Software: PyCharm



from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize


# from train import mlp_loss,dae_loss,idae_loss


def loss_line(loss_list):
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    x = np.array([i for i in range(len(loss_list[0]))])
    mlp_loss = np.array(loss_list[0])
    dae_loss = np.array(loss_list[1])
    idae_loss = np.array(loss_list[2])

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5),dpi=500)
    plt.grid()  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框


    plt.plot(x, mlp_loss, marker='v',ms=10, color="blue", label="MLP_loss", linewidth=0.7)
    plt.plot(x, dae_loss, marker='o', ms=10, color="green", label="DAE_loss", linewidth=0.7)
    plt.plot(x, idae_loss, marker='d', ms=10, color="red", label="IDAE_loss", linewidth=0.7)

    # group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
    plt.xticks(x, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=14, fontweight='bold')
    plt.title("MODELS_LOSS", fontsize=18, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("EPOCHS", fontsize=14, fontweight='bold')
    plt.ylabel("LOSS", fontsize=14, fontweight='bold')
    plt.xlim(-1, 30)  # 设置x轴的范围
    plt.ylim(1.1, 1.67)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1,labelspacing=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    # plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()

# if __name__ =='__main__':
#     loss_line([mlp_loss[0:1500:50],dae_loss[0:1500:50],idae_loss[0:1500:50]])
#
#
#




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
    plt.xlabel("MODELS_LOSS", fontsize=18, fontweight='bold')
    plt.ylabel("MODELS_LOSS", fontsize=18, fontweight='bold')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()



def bar_metrics():
    size = 8
    x = np.arange(size)
    # 有a/b两种类型的数据，n设置为2
    total_width, n = 0.6, 2
    # 每种类型的柱状图宽度
    width = total_width / n

    list1 = [88.59, 86.23, 87.79, 84.35, 89.97, 88.36, 85.51, 89.99]
    list2 = [79.92, 76.53, 79.32, 76.17, 79.78, 80.92, 77.51, 81.23]
    # 重新设置x轴的坐标
    x = x - (total_width - width) / 2
    # print(x)
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(16, 10),dpi=500)
    # 画柱状图
    plt.bar(x, list1, width=width, label="Coarse", color='#0066cc')
    plt.bar(x + width, list2, width=width, label="Fine", color='#9ACD32')
    # plt.bar(x + 2*width, c, width=width, label="c")
    # plt.xticks(np.arange(8), ('MLP', 'DAE', 'IDAE'))
    plt.xticks(np.arange(8), ('a','b','c','d','e','f','g','h'),fontsize=18, fontweight='bold')
    plt.yticks(np.arange(100), fontsize=18, fontweight='bold')
    # 显示图例
    # plt.figure(dpi=300,figsize=(24,24))
    plt.legend(loc='lower right', prop={"family": "Times New Roman"})
    plt.xlabel("Comparision   Experiments", fontname="Times New Roman",fontsize=18, fontweight='bold')
    plt.ylabel("Dice  Score", fontname="Times New Roman",fontsize=18, fontweight='bold')
    # plt.savefig('plot123_2.png', dpi=500)
    # 显示柱状图
    plt.show()


bar_metrics()
#
# label_r1 = label_binarize(r1, classes=[0, 1, 2, 3, 4])
# print(label_r1)
# label_r2 = label_binarize(r2, classes=[0, 1, 2, 3, 4])
# print(label_r2)
#
# label_test_y = label_binarize(test_y, classes=[0, 1, 2, 3, 4])
# #
#
# plt_roc(5,r1)
#
# plt_roc(5,r2)