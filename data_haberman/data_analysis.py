
import statsmodels.api as sm
import pandas as pd

import sys

# Add parent directory to the Python module search path
sys.path.append("../eb_codes")

# Import function from parent directory
from main_poisson_eb import *


from main_poisson_eb import *

# Load your dataset into a DataFrame
df = pd.read_csv('haberman.csv')

# Empirical Bayes filtering
df["eb_age"] = np.copy(poisson_eb_npmle(df["age"]-min(df["age"]), df["age"]-min(df["age"])))
df["eb_opr_yr"] = poisson_eb_npmle(df["opr_yr"]-min(df["opr_yr"]), df["opr_yr"]-min(df["opr_yr"]))
df["eb_pos_nodes"] = np.copy(poisson_eb_npmle(df["pos_nodes"], df["pos_nodes"]))

# Define the independent variables (features)
X = df[['eb_age', 'eb_pos_nodes']]  # Replace with your actual feature columns





# Add a constant column for the intercept term
X = sm.add_constant(X)

# Define the dependent variable (target)
y = df['surv']  # Replace with your actual target column

# Fit the logistic regression model
model = sm.MNLogit(y, X)
result = model.fit()

# Print the summary of the model
print(result.summary())

contents = result.summary().as_text()

# Save result summary

file_path_results = "results_w_all_eb.txt"  # Replace with the actual path where you want to save the text file

# Open the file in write mode and write the content
with open(file_path_results, 'w') as file:
    file.write(contents)


