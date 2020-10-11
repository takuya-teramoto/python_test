#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:32:10 2020

@author: takuya.teramoto
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from jax import ops,random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam

def model(x_obs, y_obs=None):
    isnan = np.isnan(x_obs)
    x_nanidx = np.nonzero(isnan)[0]
    x_mu = numpyro.sample("x_mu", dist.Uniform(-100, 100))
    x_sigma = numpyro.sample('x_sigma', dist.Uniform(0, 10))
    x_impute = numpyro.sample("x_impute", dist.Normal(x_mu, x_sigma).mask(False))
    x_obs = ops.index_update(x_obs, x_nanidx, x_impute)
    numpyro.sample('x', dist.Normal(x_mu, x_sigma), obs=x_obs)
    
    a_mu = numpyro.sample("a_mu", dist.Uniform(-100, 100))
    a_sigma = numpyro.sample('a_sigma', dist.Uniform(0, 10))
    a = numpyro.sample('a', dist.Normal(a_mu, a_sigma))
    
    b_mu = numpyro.sample("b_mu", dist.Uniform(-100, 100))
    b_sigma = numpyro.sample('b_sigma', dist.Uniform(0, 10))
    b = numpyro.sample('b', dist.Normal(b_mu, b_sigma))
    
    y_mu = a * x_obs + b
    y_sigma = numpyro.sample('y_sigma', dist.Uniform(0, 10))
    numpyro.sample('y', dist.Normal(y_mu, y_sigma), obs=y_obs)
    
    numpyro.deterministic('test', a*x_obs) # deterministicを使えば、勾配などを算出しておくことも可能なはず

# memo
# type(x_obs)
# <class 'numpy.ndarray'>
# type(x_nanidx)
# <class 'numpy.ndarray'>
# type(x_impute)
# <class 'jax.interpreters.xla.DeviceArray'>
# type(x_obs)
# <class 'jax.interpreters.xla.DeviceArray'> 27行目でx_obsのtypeが変わる

# 初期設定
num_samples = 2000
num_warmup = 200
num_chains = 1
num_x_missings = 1500
num_y_missings = 1500
rng_key = random.PRNGKey(0)
numpyro.set_platform('cpu')
numpyro.set_host_device_count(num_chains)

# 適当な関数生成
np.random.seed(0)
n = 1000
origin_x = np.random.randn(n)
origin_y = 3*origin_x + 2*np.random.randn(n)
plt.scatter(origin_x, origin_y)

# 欠損値を発生させる
x_missing_list = np.random.choice(list(range(n)), num_x_missings)
y_missing_list = np.random.choice(list(range(n)), num_y_missings)

x = origin_x.copy()
y = origin_y.copy()
x[x_missing_list] = np.nan
# y[y_missing_list] = np.nan

# numpyroでサンプリング
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=num_chains, progress_bar=True)
mcmc.run(rng_key, x, y)
mcmc.print_summary()
samples = mcmc.get_samples()
pred = Predictive(model, samples)(rng_key, x)

plt.scatter(x, y)
plt.scatter(pred['x'].mean(axis=0), pred['y'].mean(axis=0))

