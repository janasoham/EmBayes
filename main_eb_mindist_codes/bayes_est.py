

##### This code plots the predictions for different prior distributions, such as Uniform, mixture of Poisson, Gamma etc. ######


import scipy.integrate as integrate;
import scipy as sp;
from scipy.stats import poisson;
import numpy as np;


def best_expo(scale0,values0):
    values0= np.append([],values0);
    fdens1 = 0.*values0; fdens = 0.*values0;
    for i in range(len(values0)):
        l=np.arange(0,100);
        fdens1[i] = np.sum(np.exp(-l/scale0)/scale0 *
                           poisson.pmf(values0[i]+1,mu=l));
        fdens[i] = np.sum(np.exp(-l/scale0)/scale0 *
                          poisson.pmf(values0[i],mu=l));
    return (values0+1)*fdens1/fdens;


def best_unif(lb0, ub0, values0):
    values0 = np.append([], values0);
    v0= ub0-lb0;
    fdens1 = 0. * values0;
    fdens = 0. * values0;
    for i in range(len(values0)):
        l = np.arange(0, 100);
        fdens1[i] = np.sum(poisson.pmf(values0[i] + 1, mu=l))/v0;
        fdens[i] = np.sum(poisson.pmf(values0[i], mu=l))/v0;
    return (values0+1) * fdens1 / fdens;


def best_gamma(shape0, scale0, values0):
    values0 = np.append([], values0);
    fdens1 = 0. * values0;
    fdens = 0. * values0;
    for i in range(len(values0)):
        l = np.arange(0, 100);
        y1 = values0[i] + 1;
        y = values0[i];
        ################
        #Below we calculate densities without the normalizing factors
        ################
        fdens1[i] = np.sum(l ** (shape0 - 1) * np.exp(-l / scale0)/
                           (scale0 ** (shape0 - 1))* poisson.pmf(y1, mu=l));
        fdens[i] = np.sum(l ** (shape0 - 1) * np.exp(-l / scale0)/
                           (scale0 ** (shape0 - 1))* poisson.pmf(y, mu=l));
    return (values0+1) * fdens1 / fdens;


def best_poimix(sup0, prob0, values0):
    values0 = np.append([], values0);
    fdens1 = 0. * values0;
    fdens = 0. * values0;
    for i in range(len(values0)):
        l = np.arange(0, 100);
        prior_at_l= 0.*l;
        for j in range(len(l)):
            prior_at_l[j]= np.sum(prob0 * poisson.pmf(l[j], mu=sup0));
        fdens1[i] = np.sum(poisson.pmf(values0[i] + 1, mu=l)*prior_at_l);
        fdens[i] = np.sum(poisson.pmf(values0[i], mu=l)*prior_at_l);
    return (values0+1) * fdens1 / fdens;