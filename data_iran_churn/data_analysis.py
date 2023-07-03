
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from statsmodels.tools import add_constant
import sys
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


# Disable all warnings
warnings.filterwarnings("ignore")

# Add parent directory to the Python module search path
sys.path.append("../eb_codes")

# Import function from parent directory
from main_poisson_eb import *

# Load your dataset into a DataFrame
df = pd.read_csv('Customer_Churn.csv')

df.columns = ["call_fails", "complains", "subsc_len", "charge",
              "sec_use", "tot_calls", "tot_texts", "tot_diff_nums",
              "age_grp", "tariff", "status", "age", "cust_value",
              "churn"]

# Empirical Bayes filtering
df["eb_call_fails"] = poisson_eb_npmle(np.copy(df["call_fails"]), np.copy(df["call_fails"]))
df["eb_subsc_len"] = poisson_eb_npmle(np.copy(df["subsc_len"]), np.copy(df["subsc_len"]))

df["eb_tot_diff_nums"] = np.copy(df["tot_diff_nums"])

# Till above EmBayes filtering improves result

df["eb_sec_use"] = np.copy(df["sec_use"])
df["eb_tot_calls"] = np.copy(df["tot_calls"])
df["eb_tot_texts"] = np.copy(df["tot_texts"])

df_temp = df[["call_fails", "subsc_len", "sec_use", "tot_calls",
             "tot_texts", "tot_diff_nums",
              "eb_call_fails", "eb_subsc_len", "eb_sec_use", "eb_tot_calls",
             "eb_tot_texts", "eb_tot_diff_nums", "status", "churn"]]



# Define the dependent variable (target)
y = df_temp['churn']  # Replace with your actual target column

repeat = 1000

all_props = []

df_errors = pd.DataFrame()

total_count = 5

for count in tqdm(range(total_count-1)):

    prop = round((count+1)/total_count*100, 3)

    all_props.append(prop)

    errors = []

    eb_errors = []

    for rand in range(repeat):
        # Split the data into train and test sets
        df_temp_train, df_temp_test, y_train, y_test = train_test_split(df_temp, y, test_size=prop/100)

        # Define the independent variables with EmBayes (features)
        eb_X_train = df_temp_train[["eb_call_fails", "eb_subsc_len", "eb_sec_use", "eb_tot_calls",
                        "eb_tot_texts", "eb_tot_diff_nums"]]
        eb_X_test = df_temp_test[["eb_call_fails", "eb_subsc_len", "eb_sec_use", "eb_tot_calls",
                        "eb_tot_texts", "eb_tot_diff_nums"]]


        # Define the independent variables (features)
        X_train = df_temp_train[["call_fails", "subsc_len", "sec_use", "tot_calls",
                "tot_texts", "tot_diff_nums"]]  # Replace with your actual feature columns
        X_test = df_temp_test[["call_fails", "subsc_len", "sec_use", "tot_calls",
                "tot_texts", "tot_diff_nums"]]  # Replace with your actual feature columns


        # Fit the logistic regression model
        model = LogisticRegression()

        model.fit(add_constant(X_train), y_train)

        # Make predictions on new data

        predictions = model.predict(add_constant(X_test))

        errors.append(sum((predictions - y_test)**2)/len(y_test))



        # Analysis with empirical Bayes filter

        # Fit the logistic regression model
        model = LogisticRegression()

        model.fit(add_constant(eb_X_train), y_train)

        # Make predictions on new data

        predictions = model.predict(add_constant(eb_X_test))

        eb_errors.append(sum((predictions - y_test)**2)/len(y_test))

    df_errors["test%f_errors" % prop] = errors
    df_errors["test%f_eb_errors" % prop] = eb_errors


error_means = []
error_ci_wid = []

eb_error_means = []
eb_error_ci_wid = []

for prop in all_props:

    error_means.append(np.mean(df_errors["test%f_errors" % prop]))
    error_ci_wid.append(1.96*np.std(df_errors["test%f_errors" % prop])/np.sqrt(df_errors.shape[0]))

    eb_error_means.append(np.mean(df_errors["test%f_eb_errors" % prop]))
    eb_error_ci_wid.append(1.96*np.std(df_errors["test%f_eb_errors" % prop])/np.sqrt(df_errors.shape[0]))

lower_bound = np.array(error_means) - np.array(error_ci_wid)
upper_bound = np.array(error_means) + np.array(error_ci_wid)

eb_lower_bound = np.array(eb_error_means) - np.array(eb_error_ci_wid)
eb_upper_bound = np.array(eb_error_means) + np.array(eb_error_ci_wid)


# Create a figure with custom size
plt.figure(figsize=(8, 6))

plt.plot(all_props, error_means, "--", label="errors without EmBayes")
plt.fill_between(all_props, lower_bound, upper_bound, alpha=0.5)

plt.plot(all_props, eb_error_means, "-o", label="errors with EmBayes")
plt.fill_between(all_props, eb_lower_bound, eb_upper_bound, alpha=0.5)

# Add labels and title
plt.xlabel('Percentage of test samples')
plt.ylabel('Proportion of wrong predictions')
plt.title('Prediction errors with Confidence Bands')

# Add legend
plt.legend()

plt.savefig('plot_errors%d_repeats.png' % repeat)

# Show the plot
plt.show()

# Save the values

plot_values = pd.DataFrame({"props":all_props, "means":error_means,
                            "ci_lb":lower_bound, "ci_ub" : upper_bound,
                            "EB_means": eb_error_means, "EB_ci_lb": eb_lower_bound,
                            "EB_ci_ub": eb_upper_bound})

plot_values.to_csv("plot_values%d_repeats_%d_props.csv" % (repeat, len(all_props)))
