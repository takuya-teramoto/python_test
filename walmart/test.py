#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:07:48 2020

@author: takuya.teramoto
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy as sqa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

df = {}
df_head = {}
file_names = ['calendar', 'sales_train_validation', 'sales_train_evaluation', 'sample_submission', 'sell_prices']
for file_name in file_names:
    input_data_path = os.path.join('/Users/takuya.teramoto/Documents/kaggle_dataset/walmart', file_name + '.csv')
    df[file_name] = pd.read_csv(input_data_path)
    df_head[file_name] = pd.read_csv(input_data_path).head(100)
    print(file_name, df[file_name].shape)