

##### This code computes NPMLE based EB estimating functions, given the training datasets ######

import numpy as np;
from main_poisson_mindist import *;
import scipy as sp;


def eval_regfunc(lambdas, mu, newXs):
    newXs = np.append([], newXs);
    m = len(lambdas);
    loglam = np.zeros(m);
    loglam[lambdas > 0] = np.log(lambdas[lambdas > 0]);
    loglam[lambdas == 0] = -1e100;

    ret = np.zeros(len(newXs));

    for i in range(len(newXs)):
        x = newXs[i];
        # density at x and x+1
        fdens = np.sum(mu * np.exp(-lambdas + x * loglam - sp.special.gammaln(x + 1)));
        fdensp1 = np.sum(mu * np.exp(-lambdas + (x + 1) * loglam - sp.special.gammaln(x + 2)));
        ret[i] = fdensp1 * (x + 1) / fdens;

    return ret;


def poisson_eb_npmle(train,test):
    theta, mu = mindist_npmle(train);
    return eval_regfunc(theta, mu, test);


def poisson_eb_sqH(train,test):
    theta, mu = mindist_sqH(train);
    return eval_regfunc(theta, mu, test);


def poisson_eb_chisq(train,test):
    theta, mu = mindist_chisq(train);
    return eval_regfunc(theta, mu, test);


def poisson_eb_robbins_comp(train):
    val = 0. * train;
    for i in range(len(train)):
        y = train[i];
        Ny = np.sum(train == y);
        Ny1 = np.sum(train == (y + 1));
        val[i] = (y + 1) * Ny1 / Ny;
    return val;


def poisson_eb_robbins_eb(train):
    val = 0. * train;

    for i in range(len(train)):
        y = train[i];

        Ny = np.sum(train == y);

        Ny1 = np.sum(train == (y + 1));

        val[i] = (y + 1) * Ny1 / (Ny+1);

    return val;
