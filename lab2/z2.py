import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("./iris1.csv")
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:, ["variety"]].values

# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)
