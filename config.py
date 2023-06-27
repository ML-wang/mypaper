#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 9:59
# @Author  : Wang Jixin
# @File    : config.py
# @Software: PyCharm

import os
import pandas as pd
import argparse



CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class Config:
    def __init__(self):
        self.model_name = ['mlp','DAE','Improved_DAE']
        self.data_path = os.path.join(CURRENT_DIR_PATH,'data\HepatitisCdata.csv')
        self.pro_data_path = os.path.join(CURRENT_DIR_PATH, 'data\proccessed_data.csv')
        self.classifier_hid = [10,8,6]
        self.lr = 0.05
        self.pic_save_dir = os.path.join(CURRENT_DIR_PATH,'output')
        self.beast_loss = 1.2
        self.model_save_dir = os.path.join(CURRENT_DIR_PATH, 'output')
#         self.smote_df = pd.read_csv(os.path.join(CURRENT_DIR_PATH, 'data\smote_df.csv'))
#         self.test_df = pd.read_csv(os.path.join(CURRENT_DIR_PATH, 'data\\test_df.csv'))
#
# #
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--is_action",default=False, action="store_true", help="discribe is_action")
# args = parser.parse_args()
# print(args.is_action)

if __name__ == '__main__':
    con = Config()
    print(con.test_df)

