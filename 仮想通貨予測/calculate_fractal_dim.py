#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:28:36 2020

@author: takuya.teramoto
"""

import os
import random
import numpy as np
import pandas as pd
import math
import sqlalchemy as sqa
import json
import websocket
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fftpack import fft
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 初期設定
# データサイズは 1/len_window になる。
column = 'ask'
len_window = 2048*4  # フラクタル次元を求める際の区間
windows = [12, 24, 36, 48]  # 移動平均を算出する際の区間

# データの読み込み
df_origin = pd.read_csv(os.path.join(os.getcwd(), 'all.csv'))
# df_origin = df_origin.head(1000)


# # 価格がランダムな場合を想定
# df_origin['rand'] = pd.Series(np.random.random(len(df_origin)), index=df_origin.index)
# # 価格がランダムウォークな場合を想定
# num = len(df_origin)
# rand_walk = np.zeros(num)
# rand_walk[0] = 1
# for i in range(num-1):
#     rand_walk[i+1] = rand_walk[i] + random.normalvariate(0, 1)
# df_origin['rand_walk'] = pd.Series(rand_walk)


# 指定したcolumnのフラクタル次元の計算
# フラクタル次元が1.5で完全ランダムウォーク。2.0では長期的記憶を持っていないランダム。1.0なら長期記憶あり。
model = LinearRegression()
fractal_dims = {}
df_div_num = math.floor(len(df_origin) / len_window)
df_origin['idx'] = pd.qcut(df_origin.index, df_div_num, labels=[i for i in range(df_div_num)])
for df_num in range(df_div_num):
    df = df_origin.loc[df_origin['idx'] == df_num, :]
    tail_idx = df.index[-1]
    df = df.reset_index()
    N = len(df) # data number
    max_log_k = math.floor(math.log(N/2, 2))
    scores = np.zeros([max_log_k, 2])
    for idx, log_k in enumerate(range(1, max_log_k+1)):
        k = 2**log_k
        sum_Lmk = 0
        for m in range(1, k+1):
            j = math.floor((N-m)/k)
            sum_X = 0
            for i in range(1, j+1):
                sum_X += abs(df[column][m+i*k-1] - df[column][m+(i-1)*k-1])
            Lmk = (sum_X*(N-1)/(j*k))/k
            sum_Lmk += Lmk
        
        Lk = sum_Lmk/k
        scores[idx, 0], scores[idx, 1] = k, Lk
    
    x = np.log2(scores[:, 0]).reshape(-1, 1)
    y = np.log2(scores[:, 1]).reshape(-1, 1)
    
    # フラクタル次元を求める際の散布図のプロット(k, Lk)
    # fig = plt.figure()
    # plt.scatter(x, y)
    
    model.fit(x, y)
    y_pred = model.predict(x)
    R2 = r2_score(y, y_pred)
    fractal_dim = abs(model.coef_[0][0])
    
    print("R2: {:.4f}, fractal_dim: {:.3f}, tail_idx: {}, len(df): {}".format(R2, fractal_dim, tail_idx, len(df)))
    fractal_dims[(df_num, 'R2')] = R2
    fractal_dims[(df_num, 'fractal_dim')] = fractal_dim
    fractal_dims[(df_num, 'tail_idx')] = tail_idx

df_fractal_dim = pd.Series(fractal_dims).unstack()
# df_fractal_dim = df_fractal_dim[df_fractal_dim['R2'] > 0.999]

# フラクタル次元とR2をグラフ
y1_label = 'fractal_dim'
y2_label = 'R2'
y1 = df_fractal_dim[y1_label]
y2 = df_fractal_dim[y2_label]
x = df_fractal_dim.index

fig, ax1 = plt.subplots()
plt.title(column)
# ax1とax2を関連させる
ax2 = ax1.twinx()

# それぞれのaxesオブジェクトのlines属性にLine2Dオブジェクトを追加
ax1.plot(x, y1, label=y1_label)
ax2.plot(x, y2, label=y2_label, color='red')

ax1.set_xlabel('time [{}min]'.format(len(df)))
ax1.set_ylabel(y1_label)
ax2.set_ylabel(y2_label)
# 凡例
# グラフの本体設定時に、ラベルを手動で設定する必要があるのは、barplotのみ。plotは自動で設定される＞
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
# 凡例をまとめて出力する
ax1.legend(handler1 + handler2, label1 + label2, borderaxespad=0, bbox_to_anchor=(0, -0.1), loc='upper left')

# ax2.set_xlim([0, 100])
ax1.set_ylim([0.6, 2.4])
ax2.set_ylim([0.9, 1.01])

# fig = plt.figure()
# plt.xlabel(y1_label)
# plt.ylabel(y2_label)
# plt.scatter(y1, y2)

# フラクタル次元を求めたミニバッチ(df)の最後の値を代表値として、それ以外をdrop
# 元のdf_originにフラクタル次元とR2を結合する
mask = df_fractal_dim['tail_idx'].values
df_concat = df_origin.loc[df_origin.index.isin(mask), :].reset_index(drop=True)
df_concat = pd.concat([df_concat, df_fractal_dim], axis=1)

# 平均線を算出する
df_rollings = {}
for window in windows:
    df_rollings[window] = df_concat[column].rolling(window=window).mean()
    df_rollings[window].name = df_rollings[window].name + "_{}".format(window)

# データの結合
for window in windows:
    df_concat = pd.concat([df_concat, df_rollings[window]], axis=1)

# nan drop
# df_concat = df_concat.dropna().reset_index(drop=True)


# データの保存
# df_concat.to_csv(os.path.join(os.getcwd(), 'all_with_frac_dim.csv'), index=False)

fig = plt.figure()
plt.plot(df_concat.index, df_concat['ask'])
plt.plot(df_concat.index, df_concat['ask_12'])
plt.plot(df_concat.index, df_concat['ask_24'])
plt.plot(df_concat.index, df_concat['ask_36'])
plt.plot(df_concat.index, df_concat['ask_48'])

fig = plt.figure()
plt.scatter(df_concat['ask'].pct_change()[1:], df_concat['fractal_dim'][:-1])
