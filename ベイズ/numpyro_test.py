#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:09:49 2021

@author: takuya.teramoto
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random

from sklearn.linear_model import LinearRegression

color = ["red", "blue", "green", "orange"]

# データの準備
np.random.seed(seed=2)

a = np.random.rand(4) + 0.3
b = np.random.rand(4)
class_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])

x = np.random.rand(len(class_list))
y = a[class_list] * x + b[class_list] + np.random.randn(len(class_list))/10

x_line = np.linspace(0, 1, 100)
fig = plt.figure()
for i in range(len(a)):
    plt.scatter(x[class_list == i], y[class_list == i], label=str(i), color=color[i])
    plt.plot(x_line, a[i]*x_line+b[i], label="ans:{}".format(str(i)), color=color[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("ans")
plt.ylim([0.2, 1.8])


# 個体差を考慮せずに線形回帰
model = LinearRegression()

model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

fig = plt.figure()
for i in range(len(a)):
    plt.scatter(x[class_list == i], y[class_list == i], label=str(i), color=color[i])
plt.plot(x_line, model.predict(x_line.reshape(-1, 1)), color = 'red', label="predict")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("liner regression")
plt.ylim([0.2, 1.8])


# 完全に別個体として線形回帰
fig = plt.figure()
for i in range(len(a)):
    tmp_x, tmp_y = x[class_list == i], y[class_list == i]
    model.fit(tmp_x.reshape(-1, 1), tmp_y.reshape(-1, 1))
    
    plt.scatter(tmp_x, tmp_y, label=str(i), color=color[i])
    plt.plot(x_line, model.predict(x_line.reshape(-1, 1)), label="predict_{}".format(str(i)), color=color[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("individual liner regression")
plt.ylim([0.2, 1.8])


# 階層ベイズでモデル化

def hierarchical_model(x, y, class_list):
    mu_a = numpyro.sample("mu_a", dist.Uniform(-5, 5))
    sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(0.1))
    mu_b = numpyro.sample("mu_b", dist.Uniform(-5, 5))
    sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(0.1))
    
    with numpyro.plate("plate_x", len(np.unique(class_list))):
        a = numpyro.sample("a", dist.Normal(mu_a, sigma_a))
        b = numpyro.sample("b", dist.Normal(mu_b, sigma_b))
    
    sigma_y = numpyro.sample("sigma_y", dist.Uniform(0, 5))
    mu_y = a[class_list] * x + b[class_list]
    
    with numpyro.plate("plate_y", len(y)):
        numpyro.sample("y", dist.Normal(mu_y, sigma_y), obs=y)
    # numpyro.deterministic('y', y)


nuts_kernel = NUTS(hierarchical_model)
mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=1000)
rng_key = random.PRNGKey(0)

mcmc.run(rng_key, x, y, class_list)

posterior_samples = mcmc.get_samples()

fig = plt.figure()
for i in range(len(a)):
    tmp_a = posterior_samples["a"][:, i]
    tmp_b = posterior_samples["b"][:, i]
    for j in range(len(tmp_a)):
        plt.plot(x_line, tmp_a[j]*x_line+tmp_b[j], alpha=0.01, color=color[i])
    plt.scatter(x[class_list == i], y[class_list == i], label=str(i), color=color[i])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("hierarchical")
plt.ylim([0.2, 1.8])