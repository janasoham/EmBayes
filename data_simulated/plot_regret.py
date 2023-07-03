### The code below plots the regret values from all_030_data.csv, all_105_data.csv, all_200_data.csv,
# and saves them as .png files ####


import numpy as np
import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import csv;




size_vec = np.zeros(11);

for i in range(11):
    size_vec[i] = 50+ i*25

size_vec = size_vec.astype("int32")

########### Scale 2.0 ############


scale0=2;

plot_title="Prior: Exponential [scale=%g]"%(scale0);


plt.figure(num=25,figsize=(7,5),dpi=100); plt.clf();

plt.title(plot_title);
plt.xlim([min(size_vec)-50,max(size_vec)+50]);

plt.ylim(0.05,0.45);
plt.xlabel("sample size"); plt.ylabel("Training regret over the Bayes estimator");


leg_str=[];

rows_plot=10000;




df_200 = pd.read_csv("all_200_data.csv");
df_200_sqH_temp = df_200.loc[df_200["method"] == "sqH"].reset_index(drop=True);
df_200_sqH = df_200_sqH_temp.loc[range(rows_plot), df_200_sqH_temp.columns != "method"];
means_200_sqH = df_200_sqH.mean();
std_200_sqH = df_200_sqH.std();
ci_len95_200_sqH = 1.96 * std_200_sqH/np.sqrt(df_200_sqH.shape[0]);



df_200 = pd.read_csv("all_200_data.csv");
df_200_npmle_temp = df_200.loc[df_200["method"] == "npmle"].reset_index(drop=True);
df_200_npmle = df_200_npmle_temp.loc[1:rows_plot, df_200_npmle_temp.columns != "method"];
means_200_npmle = df_200_npmle.mean()
std_200_npmle = df_200_npmle.std();
ci_len95_200_npmle = 1.96 * std_200_npmle/np.sqrt(df_200_npmle.shape[0]);






df_200 = pd.read_csv("all_200_data.csv");
df_200_chisq_temp = df_200.loc[df_200["method"] == "chisq"].reset_index(drop=True);
df_200_chisq = df_200_chisq_temp.loc[1:rows_plot, df_200_chisq_temp.columns != "method"];
means_200_chisq = df_200_chisq.mean();
std_200_chisq = df_200_chisq.std();
ci_len95_200_chisq = 1.96 * std_200_chisq/np.sqrt(df_200_chisq.shape[0])



plt.plot(size_vec, means_200_sqH,'^--',color="blue", label="Hellinger");
plt.plot(size_vec, means_200_npmle,'*--',color="green",label="NPMLE");
plt.plot(size_vec, means_200_chisq,'x--',color="red",label="Chi square");


plt.legend(bbox_to_anchor=(0,1), loc='upper left');


plt.fill_between(size_vec, means_200_npmle-ci_len95_200_npmle, means_200_npmle+ci_len95_200_npmle, color='green', alpha=.2)
plt.fill_between(size_vec, means_200_chisq-ci_len95_200_chisq, means_200_chisq+ci_len95_200_chisq, color='red', alpha=.2)
plt.fill_between(size_vec, means_200_sqH-ci_len95_200_sqH, means_200_sqH+ci_len95_200_sqH, color='blue', alpha=.2)




plt.savefig("regret_expo_200_10k.png");





####### Scale 1.05 ######


plt.figure();

scale0=1.05;

plot_title="Prior: Exponential [scale=%g]"%(scale0);


plt.figure(num=25,figsize=(7,5),dpi=100); plt.clf();

plt.title(plot_title);
plt.xlim([min(size_vec)-50,max(size_vec)+50]);

plt.ylim(0.03,0.22);
plt.xlabel("sample size"); plt.ylabel("Training regret over the Bayes estimator");


leg_str=[];

rows_plot=10000;




df_105 = pd.read_csv("all_105_data.csv");
df_105_sqH_temp = df_105.loc[df_105["method"] == "sqH"].reset_index(drop=True);
df_105_sqH = df_105_sqH_temp.loc[range(rows_plot), df_105_sqH_temp.columns != "method"];
means_105_sqH = df_105_sqH.mean();
std_105_sqH = df_105_sqH.std();
ci_len95_105_sqH = 1.96 * std_105_sqH/np.sqrt(df_105_sqH.shape[0])






df_105 = pd.read_csv("all_105_data.csv");
df_105_npmle_temp = df_105.loc[df_105["method"] == "npmle"].reset_index(drop=True);
df_105_npmle = df_105_npmle_temp.loc[1:rows_plot, df_105_npmle_temp.columns != "method"];
means_105_npmle = df_105_npmle.mean()
std_105_npmle = df_105_npmle.std();
ci_len95_105_npmle = 1.96 * std_105_npmle/np.sqrt(df_105_npmle.shape[0])





df_105 = pd.read_csv("all_105_data.csv");
df_105_chisq_temp = df_105.loc[df_105["method"] == "chisq"].reset_index(drop=True);
df_105_chisq = df_105_chisq_temp.loc[1:rows_plot, df_105_chisq_temp.columns != "method"];
means_105_chisq = df_105_chisq.mean();
std_105_chisq = df_105_chisq.std();
ci_len95_105_chisq = 1.96 * std_105_chisq/np.sqrt(df_105_chisq.shape[0])



plt.plot(size_vec, means_105_sqH,'^--',color="blue", label="Hellinger");
plt.plot(size_vec, means_105_npmle,'*--',color="green",label="NPMLE");
plt.plot(size_vec, means_105_chisq,'x--',color="red",label="Chi square");


plt.legend(bbox_to_anchor=(0,1), loc='upper left');

plt.fill_between(size_vec, means_105_chisq-ci_len95_105_chisq, means_105_chisq+ci_len95_105_chisq, color='red', alpha=.2)
plt.fill_between(size_vec, means_105_sqH-ci_len95_105_sqH, means_105_sqH+ci_len95_105_sqH, color='blue', alpha=.2)
plt.fill_between(size_vec, means_105_npmle-ci_len95_105_npmle, means_105_npmle+ci_len95_105_npmle, color='green', alpha=.2)


plt.savefig("regret_expo_105_10k.png");

####### Scale 0.30 ######


plt.figure();

scale0=0.30;

plot_title="Prior: Exponential [scale=%g]"%(scale0);


plt.figure(num=25,figsize=(7,5),dpi=100); plt.clf();

plt.title(plot_title);
plt.xlim([min(size_vec)-50,max(size_vec)+50]);

plt.ylim(0,0.05);
plt.xlabel("sample size"); plt.ylabel("Training regret over the Bayes estimator");


leg_str=[];

rows_plot=10000;




df_030 = pd.read_csv("all_030_data.csv");
df_030_sqH_temp = df_030.loc[df_030["method"] == "sqH"].reset_index(drop=True);
df_030_sqH = df_030_sqH_temp.loc[range(rows_plot), df_030_sqH_temp.columns != "method"];
means_030_sqH = df_030_sqH.mean();
std_030_sqH = df_030_sqH.std();
ci_len95_030_sqH = 1.96 * std_030_sqH/np.sqrt(df_030_sqH.shape[0])




df_030 = pd.read_csv("all_030_data.csv");
df_030_npmle_temp = df_030.loc[df_030["method"] == "npmle"].reset_index(drop=True);
df_030_npmle = df_030_npmle_temp.loc[1:rows_plot, df_030_npmle_temp.columns != "method"];
means_030_npmle = df_030_npmle.mean()
std_030_npmle = df_030_npmle.std();
ci_len95_030_npmle = 1.96 * std_030_npmle/np.sqrt(df_030_npmle.shape[0])



df_030 = pd.read_csv("all_030_data.csv");
df_030_chisq_temp = df_030.loc[df_030["method"] == "chisq"].reset_index(drop=True);
df_030_chisq = df_030_chisq_temp.loc[1:rows_plot, df_030_chisq_temp.columns != "method"];
means_030_chisq = df_030_chisq.mean();
std_030_chisq = df_030_chisq.std();
ci_len95_030_chisq = 1.96 * std_030_chisq/np.sqrt(df_030_chisq.shape[0])


plt.plot(size_vec, means_030_sqH,'^--',color="blue", label="Hellinger");
plt.plot(size_vec, means_030_npmle,'*--',color="green",label="NPMLE");
plt.plot(size_vec, means_030_chisq,'x--',color="red",label="Chi square");


plt.fill_between(size_vec, means_030_chisq-ci_len95_030_chisq, means_030_chisq+ci_len95_030_chisq, color='red', alpha=.2)
plt.fill_between(size_vec, means_030_npmle-ci_len95_030_npmle, means_030_npmle+ci_len95_030_npmle, color='green', alpha=.2)
plt.fill_between(size_vec, means_030_sqH-ci_len95_030_sqH, means_030_sqH+ci_len95_030_sqH, color='blue', alpha=.2)

plt.legend(bbox_to_anchor=(0,1), loc='upper left');



plt.savefig("regret_expo_030_10k.png");

