##### The code below generates (theta,data) from the Exponential distribution with scale 1.05, ###
#### performs minimum-distance and Robbins EB estimation for theta based on the data, ####
#### calculates the corresponding training error values, and saves them in a CSV type file #######

#### All the data from different runs of this code are combined separately as all_105_data.csv #####



import pandas as pd;


# Add parent directory to the Python module search path
sys.path.append("../main_eb_mindist_codes")

from main_poisson_eb import *;
from generate_data import *;
from bayes_est import *;
from tqdm import tqdm;
import time;
from random import randrange;



#################### Bayes estimator  ####################


scale0 = 1.05;


plot_title = "Prior: Exponential [scale=%g]"%(scale0);

rand_number= randrange(1000);

est_len = 60; # The estimator will calculated for (0,1,...,est_len)

support = np.arange(0, est_len);
est_bayes = best_expo(scale0, support)


#########################################################################



size_min=50;
incr=25;
total_iter=11;
size_max=size_min+total_iter*incr;

size_names = ["" for i in range(total_iter)];


for i in range(total_iter):
    temp_size = size_min+i*incr;
    size_names[i] = "size%g"%(temp_size);


####### Creating empty csv file to add regret values #######


all_regret = pd.DataFrame(columns=["method"]+size_names);


################################################################



sub_iter=300;


for k in tqdm(range(sub_iter)):
    if k % 50 == 0:
        all_regret.to_csv('all_regret_105_number%g.csv' % rand_number, index=False);

    temp_regret_robbins = np.zeros(total_iter);
    temp_regret_sqH = np.zeros(total_iter);
    temp_regret_npmle = np.zeros(total_iter);
    temp_regret_chisq = np.zeros(total_iter);

    for size_iter in range(total_iter):

        train_size = size_min+size_iter*incr;


        ######################### Add new data points ####################

        train_theta, train_data = data_expo(scale0, train_size);
        test_data = np.copy(train_data); test_theta = np.copy(train_theta); # Same sample

        eb_sqH = poisson_eb_sqH(train_data, test_data);
        eb_npmle = poisson_eb_npmle(train_data, test_data);
        eb_chisq = poisson_eb_chisq(train_data, test_data);
        eb_robbins = poisson_eb_robbins_comp(train_data);  # We apply Robbins for compound estimation, so use train data

        ############## Prediction error computation ###############

        est_bayes_test = est_bayes[test_data];

        temp_regret_robbins[size_iter] = np.sum((est_bayes_test-eb_robbins)**2)/len(test_data);

        temp_regret_npmle[size_iter] = np.sum((est_bayes_test-eb_npmle)**2)/len(test_data);

        temp_regret_sqH[size_iter] = np.sum((est_bayes_test-eb_sqH)**2)/len(test_data);

        temp_regret_chisq[size_iter] = np.sum((est_bayes_test - eb_chisq) ** 2) / len(test_data);


    all_regret.loc[len(all_regret.index)] = ["robbins"] + list(temp_regret_robbins);
    all_regret.loc[len(all_regret.index)] = ["npmle"] + list(temp_regret_npmle);
    all_regret.loc[len(all_regret.index)] = ["sqH"] + list(temp_regret_sqH);
    all_regret.loc[len(all_regret.index)] = ["chisq"] + list(temp_regret_chisq);




all_regret.to_csv('all_regret_105_number%g.csv'%(rand_number),index=False);


