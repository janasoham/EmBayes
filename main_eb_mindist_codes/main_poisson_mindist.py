import numpy as np;
from functions_chisq import *;
from functions_npmle import *;
from functions_sqH import *;
from roots_and_preds import *;


# Given a training sample, the following function generates
# minimum nonparametric Chi square estimate of G (support and probability) based on the Poisson mixture model


def mindist_chisq(train_data):
    grid_size = len(np.unique(train_data));
    PX = {};
    for i in train_data:
        if i in PX:
            PX[i] += 1;
        else:
            PX[i] = 1;

    Xs = np.array(list(PX.keys()));
    Phat = np.array(list(PX.values()));
    Phat = Phat / np.sum(Phat);

    theta_max = np.max(Xs);

    # Initialize location of values
    theta = np.linspace(1e-3, theta_max, grid_size);
    theta_chisq=np.copy(theta);


    mu0 = np.random.rand(grid_size);
    mu0 = mu0/np.sum(mu0);
    mu0_chisq=np.copy(mu0);

    for iter in range(1,15,1):
        if iter % 2 == 1:
            mu0_chisq = get_mu_chisq(theta_chisq, mu0_chisq, Xs, Phat);
            (theta_chisq, mu0_chisq) = prune_roots(theta_chisq, mu0_chisq);

        else:
            if iter % 7 == 0:
                new_roots_chisq = get_roots_chisq(theta_chisq, mu0_chisq, Xs, Phat);
                (theta_chisq, mu0_chisq) = merge_roots(theta_chisq, mu0_chisq, new_roots_chisq);
                mu0_chisq = get_mu_chisq(theta_chisq, mu0_chisq, Xs, Phat);

            else:
                theta_chisq = theta_gradient_step_chisq(theta_chisq, mu0_chisq, Xs, Phat);
    support = np.copy(theta_chisq);
    probs = np.copy(mu0_chisq);
    return support,probs;


def mindist_npmle(train_data):
    grid_size = len(np.unique(train_data));
    PX = {};
    for i in train_data:
        if i in PX:
            PX[i] += 1;
        else:
            PX[i] = 1;

    Xs = np.array(list(PX.keys()));
    Phat = np.array(list(PX.values()));
    Phat = Phat / np.sum(Phat);

    theta_max = np.max(Xs);

    # Initialize location of values
    theta = np.linspace(1e-3, theta_max, grid_size);
    theta_npmle=np.copy(theta);


    mu0 = np.random.rand(grid_size);
    mu0 = mu0/np.sum(mu0);
    mu0_npmle=np.copy(mu0);

    for iter in range(1,15,1):
        if iter % 2 == 1:
            mu0_npmle = get_mu_npmle(theta_npmle, mu0_npmle, Xs, Phat);
            (theta_npmle, mu0_npmle) = prune_roots(theta_npmle, mu0_npmle);

        else:
            if iter % 7 == 0:
                new_roots_npmle = get_roots_npmle(theta_npmle, mu0_npmle, Xs, Phat);
                (theta_npmle, mu0_npmle) = merge_roots(theta_npmle, mu0_npmle, new_roots_npmle);
                mu0_npmle = get_mu_npmle(theta_npmle, mu0_npmle, Xs, Phat);

            else:
                theta_npmle = theta_gradient_step_npmle(theta_npmle, mu0_npmle, Xs, Phat);
    support = np.copy(theta_npmle);
    probs = np.copy(mu0_npmle);
    return support, probs;


def mindist_sqH(train_data):
    grid_size = len(np.unique(train_data));
    PX = {};
    for i in train_data:
        if i in PX:
            PX[i] += 1;
        else:
            PX[i] = 1;

    Xs = np.array(list(PX.keys()));
    Phat = np.array(list(PX.values()));
    Phat = Phat / np.sum(Phat);

    theta_max = np.max(Xs);

    # Initialize location of values
    theta = np.linspace(1e-3, theta_max, grid_size);
    theta_sqH=np.copy(theta);

    mu0 = np.random.rand(grid_size);
    mu0 = mu0/np.sum(mu0);
    mu0_sqH=np.copy(mu0);

    for iter in range(1,15,1):
        if iter % 2 == 1:
            mu0_sqH = get_mu_sqH(theta_sqH, mu0_sqH, Xs, Phat);
            (theta_sqH, mu0_sqH) = prune_roots(theta_sqH, mu0_sqH);

        else:
            if iter % 7 == 0:
                new_roots_sqH = get_roots_sqH(theta_sqH, mu0_sqH, Xs, Phat);
                (theta_sqH, mu0_sqH) = merge_roots(theta_sqH, mu0_sqH, new_roots_sqH);
                mu0_sqH = get_mu_sqH(theta_sqH, mu0_sqH, Xs, Phat);

            else:
                theta_sqH = theta_gradient_step_sqH(theta_sqH, mu0_sqH, Xs, Phat);
    support = np.copy(theta_sqH);
    probs = np.copy(mu0_sqH);
    return support,probs;

