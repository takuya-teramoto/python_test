#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:32:10 2020

@author: takuya.teramoto
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam

def model(x_obs, y_obs=None):
    x_mu = numpyro.sample("x_mu", dist.Uniform(-100, 100))
    x_sigma = numpyro.sample('x_sigma', dist.Uniform(0, 10))
    x = numpyro.sample('x', dist.Normal(x_mu, x_sigma), obs=x_obs)
    a_mu = numpyro.sample("a_mu", dist.Uniform(-100, 100))
    a_sigma = numpyro.sample('a_sigma', dist.Uniform(0, 10))
    a = numpyro.sample('a', dist.Normal(a_mu, a_sigma))
    b_mu = numpyro.sample("b_mu", dist.Uniform(-100, 100))
    b_sigma = numpyro.sample('b_sigma', dist.Uniform(0, 10))
    b = numpyro.sample('b', dist.Normal(b_mu, b_sigma))
    y_mu = a * x + b
    y_sigma = numpyro.sample('y_sigma', dist.Uniform(0, 10))
    y = numpyro.sample('y_mu', dist.Normal(y_mu, y_sigma), obs=y_obs)


# 初期設定
num_samples = 2000
num_warmup = 200
num_chains = 2
rng_key = random.PRNGKey(0)
numpyro.set_platform('cpu')
numpyro.set_host_device_count(num_chains)

# 適当な関数生成
n = 1000
x = np.random.randn(n)
y = 3*x + 2*np.random.randn(n)
plt.scatter(x, y)

# numpyroでサンプリング
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=num_chains, progress_bar=True)
mcmc.run(rng_key, x, y)
mcmc.print_summary()
samples = mcmc.get_samples()
pred = Predictive(model, samples)(rng_key, x)

