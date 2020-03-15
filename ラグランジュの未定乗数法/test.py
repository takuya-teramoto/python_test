#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:31:24 2020

@author: takuya.teramotox
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import *
from scipy.optimize import minimize

seed(0)

#予測させる関数の定義
def ans_f(x1, x2, rand_level=5):
    a = [-2, -1, 1, -10, 1, 10]
    y = a[0] * x1**2 + a[1] * x2**2 + a[2] * x1 * x2 + a[3] * x1 + a[4] * x2 + a[5]
    y = np.vectorize(lambda x: x * (1 + rand() * rand_level))(y)
    return y

x1, x2 = np.meshgrid(np.arange(-3, 3, 1), np.arange(-3, 3, 1))
shape = x1.shape
y = ans_f(x1, x2)
plt.contour(x1, x2, y)
plt.show()

x1, x2, y = x1.reshape(-1, 1), x2.reshape(-1, 1), y.reshape(-1, 1)

class Ellipse2DModel():
    def __init__(self):
        self.x1, self.x2, self.y = np.nan, np.nan, np.nan
        self.coef_ = np.nan
        self.intercept_ = np.nan
        self.a = np.nan
        
    def f(self, a, x1, x2):
        return a[0] * x1**2 + a[1] * x2**2 + a[2] * x1 * x2 + a[3] * x1 + a[4] * x2 + a[5]
    
    def L1_norm(self, a):
        return np.sum(np.abs(a))
    
    def L2_norm(self, a):
        return np.sum(np.square(a))
    
    def minimize_E(self, a):
        return np.sum(np.abs(self.y - self.f(a[:-1], self.x1, self.x2))) + a[-1] * self.L1_norm(a[:-1])

    def fit(self, X, y):
        self.x1, self.x2, self.y = X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), y
        a = [-1, -1, 0, 0, 0, 0, 0.5]
        # a = [1, 1, 1, 1, 1, 1, 1]
        eps = 1e-4
        cons = (
            {'type': 'ineq', 'fun': lambda a: 4 * a[0] * a[1] - a[2]**2},
            {'type': 'ineq', 'fun': lambda a: -a[0] - eps}, 
            {'type': 'ineq', 'fun': lambda a: -a[1] - eps}, 
            {'type': 'ineq', 'fun': lambda a: a[-1] - eps}, 
            {'type': 'ineq', 'fun': lambda a: 1 - a[-1]}, 
        )
        res = minimize(self.minimize_E, x0=a, constraints=cons, method="SLSQP")
        self.a = res.x
        self.coef_ = res.x[:5]
        self.intercept_ = res.x[5]
        
    def predict(self, X):
        return self.f(self.a, X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))

X = np.concatenate([x1, x2], 1)
model = Ellipse2DModel()
model.fit(X, y)

y_pred = model.predict(X)

plt.contour(x1.reshape(shape), x2.reshape(shape), y_pred.reshape(shape))
plt.show()

print(model.a)