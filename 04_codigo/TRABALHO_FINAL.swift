import pandas as pd

df = pd.read_csv("ansiedade_e_depressao.csv")

print(df.columns.tolist())
print(df.head())