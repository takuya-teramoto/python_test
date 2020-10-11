#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:38:37 2020

@author: takuya.teramoto
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam


def model(X, y=None):
    sd_randwalk = numpyro.sample('sd_randwalk', dist.Uniform(low=0, high=100.0))
    randwalk = numpyro.sample('mu', dist.Normal(loc=0, scale=sd_randwalk))
    value = X + randwalk
    sd_value = numpyro.sample('sd', dist.Uniform(low=0.0, high=100.0))
    pred_y = numpyro.sample('pred_y', dist.Normal(loc=value, scale=sd_value), obs=y)

if __name__ == "__main__":
    # 初期設定
    num_samples = 1000
    num_warmup = 100
    num_chains = 1
    device = 'cpu'
    rng_key = random.PRNGKey(0)
    numpyro.set_platform(device)
    numpyro.set_host_device_count(num_chains)
    
    # ランダムウォーク波形生成
    random_walk = jnp.cumsum(0.5*np.random.randn(100000))
    
    # 価格がランダムな場合を想定
    # random_walk = 0.5*np.random.randn(100000)
    
    random_walks = {
        1: random_walk, 
        2: random_walk[::2], 
        4: random_walk[::2][::2], 
        8: random_walk[::2][::2][::2], 
        16: random_walk[::2][::2][::2][::2], 
        32: random_walk[::2][::2][::2][::2][::2], 
        64: random_walk[::2][::2][::2][::2][::2][::2], 
        128: random_walk[::2][::2][::2][::2][::2][::2][::2], 
        256: random_walk[::2][::2][::2][::2][::2][::2][::2][::2], 
        512: random_walk[::2][::2][::2][::2][::2][::2][::2][::2][::2], 
        1024: random_walk[::2][::2][::2][::2][::2][::2][::2][::2][::2][::2], 
        }
    results = {}
    for key, data in random_walks.items():
        X = data[:-1]
        y = data[1:]
    
        # カーネル生成、モデル定義、学習、サンプル取得
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=num_chains, progress_bar=True)
        mcmc.run(rng_key, X, y)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        
        results[key] = samples['sd_randwalk'].mean()
    
        # collect deterministic sites
        pred = Predictive(model, samples)(rng_key, X)["pred_y"]
    
        pred_mean = pred.mean(axis=0)
        pred_std = pred.std(axis=0)
        time = range(len(pred_mean))
        
        fig = plt.figure()
        plt.title(key)
        plt.plot(time, y)
        plt.plot(time, pred_mean)
        plt.fill_between(time, pred_mean-pred_std, pred_mean+pred_std, facecolor='y', alpha=0.5)
        
    fig = plt.figure()
    plt.scatter(results.keys(), results.values())
    plt.xlabel('minute chart')
    plt.ylabel('sd_randomwalk')