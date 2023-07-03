#### This code plots the hockey-data goals from the year 2018 and
#### their corresponding minimum-distance based and Robbins based EB predictions for the year 2019 ####
#### for the position defender ######

import numpy as np;
from eval_hockey_robbins import *;
from main_poisson_eb import *;
from roots_and_preds import *;
from matplotlib.ticker import MaxNLocator;

######################### Data points ####################

file1 = 'season_2018.csv';
file2 = 'season_2019.csv';

plot_title="Season 2018 vs Season 2019: Defender";

(PX, gpast, gfut) = hockey_data_position(file1, file2, "defender");

Xs = np.array(list(PX.keys()));
Phat = np.array(list(PX.values()));
Phat = Phat / np.sum(Phat);



################ Test data ##################

plot_Xs = np.arange(0, np.max(gpast) + 1)

eb_sqH = poisson_eb_sqH(gpast, plot_Xs);
eb_npmle = poisson_eb_npmle(gpast, plot_Xs);
eb_chisq = poisson_eb_chisq(gpast, plot_Xs);
eb_robbins = poisson_eb_robbins_comp(gpast);


############## Prediction error computation ###############

rmse_robbins, l1_robbins = prediction_error(gfut, eb_robbins);

rmse_npmle, l1_npmle = prediction_error(gfut, eb_npmle[gpast]);

rmse_sqH, l1_sqH = prediction_error(gfut, eb_sqH[gpast]);

rmse_chisq, l1_chisq = prediction_error(gfut, eb_chisq[gpast]);




print()
print("Prediction errors : rmse_Robbins=%g, rmse_Hellinger=%g, rmse_NPMLE=%g, mse_Chisquare=%g"
      % (rmse_robbins, rmse_sqH, rmse_npmle, rmse_chisq));
print()
print("Prediction errors absolute: Robbins=%g, Hellinger=%g, NPMLE=%g, Chisquare=%g"
      % (l1_robbins, l1_sqH, l1_npmle, l1_chisq));
print()

fg = plt.figure(num=10,figsize=(7,5));
plt.clf();

fg = fg.gca();
fg.xaxis.set_major_locator(MaxNLocator(integer=True));
asort = np.argsort(-gpast);
plt.plot(gpast, gfut, 'o', fillstyle="none",
         label="Past vs Future",color="dodgerblue");
plt.xlabel('Past');
plt.ylabel('Future');
plt.title(plot_title);


############################################################

plt.plot(gpast[asort], eb_robbins[asort], 'o--',
         color="orange", label="Robbins estimator");

plt.plot(plot_Xs, eb_sqH, '^--',
         color="blue", label="Hellinger estimator");

plt.plot(plot_Xs, eb_npmle, '*--',
         color="green", label="NPMLE estimator");

plt.plot(plot_Xs, eb_chisq, 'x--',
         color="red", label="Chi square estimator");


plt.ylim([-5,80]);
plt.legend(bbox_to_anchor=(0,1), loc='upper left');

plt.savefig("pred_plot_real_w_robbins_defender.png");

plt.pause(1);

plt.show()
