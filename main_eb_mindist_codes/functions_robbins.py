
#### The code below stores the necessary functions for calculating the Robbins estimates #####

import scipy as sp
import scipy.optimize as optimize
from eval_hockey_robbins import *;
from scipy.stats import poisson;

def eval_robbins(train_size,train_pmf,predictor):
    N = len(predictor);
    def get_from_train_pmf(x):
        if x in train_pmf:
            return train_pmf[x];
        else:
            return 0.;
    test_robbins = 1.*predictor;
    for i in range(0, N):
        x = predictor[i];
        if (get_from_train_pmf(x) > 0):
            test_robbins[i] = (x + 1) * (0. + get_from_train_pmf(x + 1)) / (0. + get_from_train_pmf(x));
        else:
            test_robbins[i]=(x + 1) * (0. + get_from_train_pmf(x + 1)*train_size) /\
                            (0. + get_from_train_pmf(x)*train_size+1);
    return test_robbins;