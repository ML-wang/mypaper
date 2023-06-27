#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 14:07
# @Author  : Wang Jixin
# @File    : pca.py
# @Software: PyCharm


from sklearn.decomposition import PCA,KernelPCA
import numpy as np
import matplotlib.pyplot as plt



pca = PCA(n_components=3)
kpca = KernelPCA(n_components=3)
pcadata = kpca.fit_transform(de_features.detach().numpy())

pca_data = kpca.transform(features.detach().numpy())
df = pd.DataFrame(pca_data,columns=['pca1','pca2','pca3'])
df['label'] = test_y.values
df

x1 = df[df['label'] == 0]['pca1']
y1 = df[df['label'] == 0]['pca2']
z1 = df[df['label'] == 0]['pca3']

x2 = df[df['label'] == 1]['pca1']
y2 = df[df['label'] == 1]['pca2']
z2 = df[df['label'] == 1]['pca3']

x3 = df[df['label'] == 2]['pca1']
y3 = df[df['label'] == 2]['pca2']
z3 = df[df['label'] == 2]['pca3']

x4 = df[df['label'] == 3]['pca1']
y4 = df[df['label'] == 3]['pca2']
z4 = df[df['label'] == 3]['pca3']

x5 = df[df['label'] == 4]['pca1']
y5 = df[df['label'] == 4]['pca2']
z5 = df[df['label'] == 4]['pca3']

fig = plt.figure(figsize=(8, 8), dpi=512)  # 初始化一张画布

ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
ax.set_title('PCA_3d_image')  # 设置本图名称
ax.scatter(x1, y1, z1, c='b')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
ax.scatter(x2, y2, z2, c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
ax.scatter(x3, y3, z3, c='g')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
ax.scatter(x4, y4, z4, c='y')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
ax.scatter(x5, y5, z5, c='k')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

plt.legend(['0=Blood Donor', '3=Cirrhosis', '1=Hepatitis', '2=Fibrosis',
            '0s=suspect Blood Donor'])

ax.set_xlabel('X')  # 设置x坐标轴
ax.set_ylabel('Y')  # 设置y坐标轴
ax.set_zlabel('Z')  # 设置z坐标轴

plt.show()