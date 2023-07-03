#### The code below generates data from Poisson mixture given the prior distribution #####


import numpy as np;
from scipy.stats import poisson;
import random;


def data_expo(scale0,size):
    theta = np.zeros(size);
    data = np.zeros(size);
    for i in range(size):
        theta[i] = np.random.exponential(scale=scale0);
        data[i] = np.random.poisson(lam=theta[i]);
    data = data.astype("int32");
    return (theta,data);

def data_gamma(shape0,scale0, size):
    theta = np.zeros(size);
    data = np.zeros(size);
    for i in range(size):
        theta[i] = np.random.gamma(shape=shape0,scale=scale0);
        data[i] = np.random.poisson(lam=theta[i]);
    data = data.astype("int32");
    return (theta, data);

def data_unif(low0, high0, size):
    theta = np.zeros(size);
    data = np.zeros(size);
    for i in range(size):
        theta[i] = np.random.uniform(low0, high0);
        data[i] = np.random.poisson(lam=theta[i]);
    data = data.astype("int32");
    return (theta, data);

def data_poimix(sup0, prob0, size):
    theta = np.zeros(size);
    data = np.zeros(size);
    for i in range(size):
        a_temp = np.random.choice(sup0, p=prob0, size=1);
        theta[i] = np.random.poisson(lam=a_temp, size=1);
        data[i] = np.random.poisson(lam=theta[i], size=1);
    data = data.astype("int32");
    return (theta, data);

       

