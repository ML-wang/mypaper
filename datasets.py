#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 9:58
# @Author  : Wang Jixin
# @File    : datasets.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import os
from config import Config


def process_data(config):
    data = pd.read_csv(config.data_path,encoding='utf-8')
    replace_map = {'0=Blood Donor':0,'1=Hepatitis':1,'2=Fibrosis':2,'3=Cirrhosis':3,'0s=suspect Blood Donor':4}
    gender_map = {'m':1,'f':0}
    data['Category'] = data['Category'].map(replace_map)
    data['Sex'] = data['Sex'].map(gender_map)
    data.drop(labels='Unnamed: 0',axis=1,inplace=True)
    data['ALB'].fillna(data['ALB'].mean(),inplace=True)
    data['ALP'].fillna(data['ALP'].mean(),inplace=True)
    data['ALT'].fillna(data['ALT'].mean(),inplace=True)
    data['CHOL'].fillna(data['CHOL'].mean(),inplace=True)
    data['PROT'].fillna(data['PROT'].mean(),inplace=True)
    if not os.path.exists(os.path.dirname(config.data_path)+'/proccessed_data.csv'):
        data.to_csv(os.path.dirname(config.data_path)+'/proccessed_data.csv',index=False)
    return data


def split_train_test(config):
    data = pd.read_csv(config.pro_data_path,encoding='utf-8',index_col=False)
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42, test_size=0.25, stratify=y)
    smote_train_x, smote_train_y = SMOTE(k_neighbors=4, random_state=42).fit_resample(train_x, train_y)
    # smote_train_x,smote_train_y = train_x,train_y
    sc = StandardScaler()
    cols = smote_train_x.columns
    smote_train_x = sc.fit_transform(smote_train_x)
    smote_train_x = pd.DataFrame(smote_train_x, columns=cols)
    smote_df = smote_train_x
    smote_df['label'] = smote_train_y.values
    test_x = pd.DataFrame(sc.transform(test_x), columns=cols)
    test_df = test_x
    test_df['label'] = test_y.values
    if not os.path.exists(os.path.dirname(config.data_path)+'/smote_df.csv'):
        data.to_csv(os.path.dirname(config.data_path)+'/smote_df.csv',index=False)
        data.to_csv(os.path.dirname(config.data_path)+'/test_df.csv',index=False)
    return smote_df,test_df


if __name__ == '__main__':
    config = Config()
    data = process_data(config)
    smote_df,test_df, = split_train_test(config)
    print('123')

