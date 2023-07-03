#### The code below stores the necessary functions for calculating the minimum-Chi square distance estimates #####

import numpy as np;
import scipy as sp;
import scipy.optimize as optimize;
from scipy.stats import poisson;
from scipy.optimize import LinearConstraint;

################### Optimization routine for Chi square ##########################

def val_chisq(theta, mu, Xs, Phat):
    m = len(theta);
    logtheta = np.zeros(m);
    logtheta[theta > 0] = np.log(theta[theta > 0]);
    logtheta[theta == 0] = -1e100;
    ret = 0.;
    for j in range(len(Xs)):
        x = Xs[j];
        # density at x
        fdens = np.sum(mu * np.exp(-theta + x * logtheta - sp.special.gammaln(x + 1)));
        ret += (Phat[j])**2 / fdens;
    return (ret-1);


# Optimize atoms given locations
def get_mu_chisq(theta, mu0, Xs, Phat):
    m = len(theta);

    if (len(mu0) != m):
        # Ok so the number of theta changed, reinit mu0
        mu0 = np.ones(m) / m;

    logtheta = np.zeros(m);
    logtheta[theta > 0] = np.log(theta[theta > 0]);
    logtheta[theta == 0] = -1e100;

    def target(mu):
        ret = 0.;
        for j in range(len(Xs)):
            x = Xs[j];
            # density at x
            fdens = np.sum((mu+1e-10)/np.sum(mu+1e-10) * np.exp(-theta + x * logtheta - sp.special.gammaln(x + 1)));
            ret += (Phat[j])**2/ fdens;
        return (ret-1);

    def jac(mu):
        ret = np.zeros(m);
        fdens = np.zeros(len(Xs));
        for i in range(len(Xs)):
            x = Xs[i];
            # density at x
            fdens[i] = np.sum((mu+1e-10)/(np.sum(mu+1e-10)) * np.exp(-theta + x * logtheta - sp.special.gammaln(x + 1)));

        for j in range(m):
            comp_j = np.zeros(len(Xs));
            for i in range(len(Xs)):
                x = Xs[i];
                # density at x
                nu = -(mu+1e-10)/np.sum(mu+1e-10)** 2;
                nu[j] = (np.sum(mu+1e-10) - (mu[j]+1e-10)) / np.sum(mu+1e-10)**2;
                comp_j[i] = np.sum(nu * np.exp(-theta + x * logtheta - sp.special.gammaln(x + 1)));
            ret[j] = -np.sum(np.square(Phat / fdens) * comp_j);

        return ret;

    # Without equality constraints
    result = optimize.minimize(target, mu0, jac=jac, method='L-BFGS-B',
                               bounds=sp.optimize.Bounds(np.zeros(m), np.inf + np.zeros(m), True));

    if (not result.success):
        print(result);
        print("sample size=%g" % (len(Xs)));
        print();
        print("sample values=");
        print(Xs);
        raise NameError('Optimization failed');

    return result.x / np.sum(result.x);


def theta_grad_chisq(theta, mu, Xs, Phat):
    m = len(theta);
    logtheta = np.zeros(m);
    logtheta[theta > 0] = np.log(theta[theta > 0]);
    logtheta[theta == 0] = -1e100;
    fdens = np.zeros(len(Xs));
    for i in range(len(Xs)):
        x = Xs[i];
        # density at x
        fdens[i] = np.sum(mu * np.exp(-theta + x * logtheta - sp.special.gammaln(x + 1)));

    ret = np.zeros(m);
    for j in range(m):
        if (theta[j] == 0):
            ret[j] = -np.sum(np.square(Phat[Xs == 0] / fdens[Xs == 0])) * mu[j];
        else:
            fdensj = np.zeros(len(Xs));
            for i in range(len(Xs)):
                x = Xs[i];
                # density at x
                fdensj[i] = np.exp(-theta[j] + (x - 1) * logtheta[j] - sp.special.gammaln(x + 1));
            ret[j] = -np.sum(np.square(Phat / fdens) * fdensj * (Xs - theta[j])) * mu[j];

    return ret;


def theta_gradient_step_chisq(theta, mu, Xs, Phat):
    step = 0.1;
    theta_max = np.max(Xs);
    theta_new = theta + step * theta_grad_chisq(theta, mu, Xs, Phat);
    theta_new = np.maximum(theta_new, 0);
    theta_new = np.minimum(theta_new, theta_max);
    return theta_new;


def get_roots_chisq(theta, mu, Xs, Phat):
    xmax = np.max(Xs);
    Wx = np.zeros(xmax + 2);
    m = len(theta);
    logtheta = np.zeros(m);
    logtheta[theta > 0] = np.log(theta[theta > 0]);
    logtheta[theta == 0] = -1e100;
    for j in range(len(Xs)):
        x = Xs[j];
        # density at x
        fdens = np.sum(mu * np.exp(-theta + x * logtheta - sp.special.gammaln(x + 1)));
        Wx[Xs[j]] = np.square(Phat[j]/fdens);

    # Coeffs of poly: \sum_j \theta^j p[xmax - j]
    p = np.zeros(xmax + 1);
    for x in range(xmax + 1):
        p[xmax - x] = (Wx[x + 1] - Wx[x]) * np.exp(-sp.special.gammaln(x + 1));

    roots = np.roots(p);

    # zero is always a root
    new_theta = [0, ];
    slack = 1e-2;
    for r in roots:
        ima = np.imag(r);
        if (ima <= 0) & (ima > -slack) & (np.real(r) >= 0):
            new_theta += [np.real(r)];


    return np.sort(new_theta);



