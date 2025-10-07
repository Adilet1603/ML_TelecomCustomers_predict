import pandas as pd

df = pd.read_csv("TelcoCustomers.csv")

print(df["InternetService"].unique())
print(df["OnlineSecurity"].unique())
print(df["TechSupport"].unique())
print(df["Contract"].unique())