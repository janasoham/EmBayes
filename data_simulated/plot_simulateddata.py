

##### This code plots the predictions for different prior distributions, such as Uniform, mixture of Poisson, Gamma etc. ######

import matplotlib.pyplot as plt;
import scipy as sp;
import scipy.integrate as integrate;


# Add parent directory to the Python module search path
sys.path.append("../main_eb_mindist_codes")

from main_poisson_eb import *;
from bayes_est import *;
from generate_data import *;


#### Bayes estimator calculation ####

sup0=[1,2,8];
prob0=[0.2,0.3,0.5];
shape0 = 4;
scale0 = 2;
lb0=0
ub0= 3;

# plot_title="Prior: Gamma [shape=%g,scale=%g]"%(shape0,scale0);
# plot_title="Prior on Theta: Unif [0,%g]" % (ub);
# plot_title="Prior on theta: Exponential [scale=%g]"%(scale0);
plot_title="Prior: 0.2*Poi(1)+0.3*Poi(2)+0.5*Poi(8)";


est_len=60; # The estimator will calculated for (0,1,...,est_len)

support=np.arange(0,est_len);
# est_bayes=best_gamma(shape0,scale0,support)
# est_bayes=best_unif(lb0,ub0,support)
# est_bayes=best_expo(scale0,support)
est_bayes=best_poimix(sup0,prob0,support)

################### Data points #######################

train_size = 600;

np.random.seed(100);


# train_theta,train_data = data_gamma(scale0,shape0,train_size);
# train_theta,train_data = data_unif(low0,high0,train_size);
# train_theta,train_data = data_expo(scale0,train_size);
train_theta,train_data = data_poimix(sup0,prob0,train_size);


################ Test data ##################

test_data = np.copy(train_data)
test_theta = np.copy(train_theta);

################ min-dist and Robbins calculation ##############

plot_Xs = np.arange(0, np.max(test_data) + 1)

eb_sqH = poisson_eb_sqH(train_data,plot_Xs);
eb_npmle = poisson_eb_npmle(train_data,plot_Xs);
eb_chisq = poisson_eb_chisq(train_data,plot_Xs);
eb_robbins = poisson_eb_robbins_comp(train_data); # We apply Robbins for compound estimation, so use train data


############## Prediction error computation ###############

rmse_robbins, l1_robbins = prediction_error(test_theta, eb_robbins);

rmse_npmle, l1_npmle = prediction_error(test_theta, eb_npmle[test_data]);

rmse_sqH, l1_sqH = prediction_error(test_theta, eb_sqH[test_data]);

rmse_chisq, l1_chisq = prediction_error(test_theta, eb_chisq[test_data]);

print()
print("Prediction errors : rmse_Robbins=%g, rmse_Hellinger=%g, rmse_NPMLE=%g, mse_Chisquare=%g"
      % (rmse_robbins, rmse_sqH, rmse_npmle, rmse_chisq));
print()
print("Prediction errors absolute: Robbins=%g, Hellinger=%g, NPMLE=%g, Chisquare=%g"
      % (l1_robbins, l1_sqH, l1_npmle, l1_chisq));
print()


############# Plot data ##############

plt.figure(num=4,figsize=(7,5));
plt.clf();

asort = np.argsort(-test_data);
plt.plot(test_data[asort], test_theta[asort],
         'o',fillstyle="none",color="dodgerblue",
         label='True Theta');
plt.xlabel('Data');
plt.ylabel('Theta');
plt.title(','.join([plot_title,"n=%g"%(train_size)]))

plt.xlim(-1,22);
plt.ylim(-1,45)


################## Bayes est plot ##############

plt.plot(plot_Xs,est_bayes[plot_Xs],'--',color="black",label="Bayes estimator");

############################################################

plt.plot(test_data[asort], eb_robbins[asort], 'o--',
             color="orange",label="Robbins estimator");

plt.plot(plot_Xs, eb_sqH, '^--',
             color="blue",label="Hellinger estimator");

plt.plot(plot_Xs, eb_npmle, '*--',
             color="green",label="NPMLE estimator");

plt.plot(plot_Xs, eb_chisq, 'x--',
             color="red",label="Chi square estimator");


plt.legend(bbox_to_anchor=(0,1), loc='upper left');
plt.savefig("pred_plot_unif%g_w_robbin.png"%(train_size));




plt.show()
