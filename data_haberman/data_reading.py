import pandas as pd
import csv
import matplotlib.pyplot as plt




# reading csv files
df = pd.read_csv('haberman.data', sep=",", header=None)


header = ["age", "opr_yr", "pos_nodes", "surv"]

df.columns = header

df.to_csv('haberman.csv', index=False)

print(df.head())

# # This here demonstrates how to read and save contents description file

file_path = 'haberman.names'

with open(file_path, 'r') as file:
    contents = file.read()
    # Process the contents of the file as needed
    print(contents)

file_path2 = "contents.txt"  # Replace with the actual path where you want to save the text file

# Open the file in write mode and write the content
with open(file_path2, 'w') as file:
    file.write(contents)


exit()







