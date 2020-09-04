#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:02:24 2020

@author: takuya.teramoto
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import keras
import math

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

from sklearn.metrics import r2_score

# ライブラリの重複によるエラーを回避するための環境変数の設定
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# データ読み込み
df = pd.read_csv(os.path.join(os.getcwd(), 'all_with_frac_dim.csv'))
df_columns = ['ask', 'bid', 'high', 'last', 'low', 'volume', 'fractal_dim', 'R2']
df = df.loc[:, df_columns]
mean = df.mean().values.reshape(-1, df.shape[1])
std = df.std().values.reshape(-1, df.shape[1])
df = df.dropna(how='any', axis=0)
df = (df - mean) / std # 標準化（仮）
array = df.values

# 時系列方向に切り出す長さの設定
time_len = 64

# 予測を何分先にするか
delay = 1

# 各パラメータの設定
input_dim = len(df_columns)  # 入力データの次元数
output_dim = 2               # 出力データの次元数
num_hidden_units = 100       # 隠れ層のユニット数
batch_size = 100             # ミニバッチサイズ
num_of_training_epochs = 50  # 学習エポック数
learning_rate = 0.01        # 学習率
train_rate = 0.9             # 学習データとテストデータの割合
validation_split=0.05        # kerasの学習過程におけるvalidatoin dataの割合


# データ整形
data_size = array.shape[0] - time_len + 1 - delay
X = np.zeros((data_size, time_len, input_dim))
y = np.zeros((data_size, output_dim))
for i in range(data_size):
    # 説明変数
    tmp_X = array[i:i+time_len, :]
    # tmp_X = (tmp_X - array[i+time_len-1, :]) / std
    shape_X = tmp_X.shape
    tmp_X = tmp_X.reshape(-1, shape_X[0], shape_X[1])
    
    # 目的変数
    # tmp_y = ((array[i+time_len+delay-1, 0:2] / array[i+time_len-1, 0:2]) - 1) * 100
    tmp_y = array[i+time_len+delay-1, 0:2] - array[i+time_len-1, 0:2]
    shape_y = tmp_y.shape
    tmp_y = tmp_y.reshape(-1, shape_y[0])
    
    # 結合
    X[i, :, :] = tmp_X
    y[i, :] = tmp_y
    # 前処理プロセスの経過時間確認
    progress = i / data_size * 100
    if i % (data_size//10) == 0:
        print("data preprocessing progress is {:.0f}%".format(progress))

# (サンプルサイズ, 時系列, 特徴量の次元)になっていることを確認する
print("X shape: {}, y shape: {}".format(X.shape, y.shape))

# 学習用データとテスト用データに切り分ける
train_size = math.floor(X.shape[0] * train_rate)
X_train, X_test = X[0:train_size, :, :], X[train_size:-1, :, :]
y_train, y_test = y[0:train_size, :], y[train_size:-1, :]

# モデル構築
model = Sequential()
model.add(LSTM(
    num_hidden_units,
    input_shape=(time_len, input_dim),
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
model.summary()

# 学習
result = model.fit(
    X_train, y_train, 
    batch_size=batch_size, 
    epochs=num_of_training_epochs, 
    validation_split=validation_split, 
)

# 学習過程プロット
train_loss = result.history['loss']
val_loss = result.history['val_loss']
epoc = range(len(train_loss))
fig = plt.figure()
plt.title(model.loss)
plt.xlabel('epoc')
plt.ylabel('loss')
plt.plot(epoc, train_loss, label='train_loss')
plt.plot(epoc, val_loss, label='val_loss')
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)

# 学習データに対する予測のr2とグラフ描写
pred_y_train = model.predict(X_train)
print("train r2 score(ask): {}".format(r2_score(y_train[:, 0], pred_y_train[:, 0])))
print("train r2 score(bid): {}".format(r2_score(y_train[:, 1], pred_y_train[:, 1])))

t = range(y_train.shape[0])
fig = plt.figure()
plt.title('train')
plt.plot(t, y_train[:, 0], label='ask')
plt.plot(t, pred_y_train[:, 0], label='ask_pred')
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)

fig = plt.figure()
plt.title('train')
plt.plot(t, y_train[:, 1], label='bid')
plt.plot(t, pred_y_train[:, 1], label='bid_pred')
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)

# テストデータに対する予測のr2とグラフ描写
pred_y = model.predict(X_test)
print("keras test r2 score(ask): {}".format(r2_score(y_test[:, 0], pred_y[:, 0])))
print("keras test r2 score(bid): {}".format(r2_score(y_test[:, 1], pred_y[:, 1])))

t = range(y_test.shape[0])
fig = plt.figure()
plt.title('test')
plt.plot(t, y_test[:, 0], label='ask')
plt.plot(t, pred_y[:, 0], label='ask_pred')
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)

fig = plt.figure()
plt.title('test')
plt.plot(t, y_test[:, 1], label='bid')
plt.plot(t, pred_y[:, 1], label='bid_pred')
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)