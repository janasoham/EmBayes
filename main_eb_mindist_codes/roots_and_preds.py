### The code below stores the necessary functions for pruning and merging
# the roots of the equation for minimum-distance estimates,
# functions for calculating EB estimates of theta given data and the estimate of prior #####


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def prune_roots(lambdas, mu):
    new_lam = list(lambdas);
    new_mu = list(mu);

    i = 0;
    while i < len(new_mu):
        if new_mu[i] < 1e-3:
            del (new_mu[i]);
            del (new_lam[i]);
        else:
            i = i + 1;

    return np.array(new_lam), np.array(new_mu);


def merge_roots(lambdas, mu, new_roots):
    new_mu = np.zeros(len(new_roots));
    all_roots = list(lambdas) + list(new_roots);
    all_mu = list(mu) + list(new_mu);

    # Sort
    b = sorted(zip(all_roots, all_mu));
    sort_roots = [x[0] for x in b];
    sort_mu = [x[1] for x in b];

    # delete all roots closer than slack;
    i = 1;
    slack = 1e-2;
    while i < len(sort_roots):
        if (sort_roots[i] - sort_roots[i - 1]) < slack:
            sort_mu[i - 1] += sort_mu[i];
            del sort_mu[i];
            del sort_roots[i];
        else:
            i = i + 1;

    return np.array(sort_roots), np.array(sort_mu);


def prediction_error(observation, estimate):
    N = len(observation);
    rmse_new = np.sqrt(np.sum((observation - estimate)**2)/N);
    l1_new = np.sum(np.abs(observation - estimate)) / N;

    return (rmse_new,l1_new);
