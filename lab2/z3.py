import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("./iris1.csv")

print(df)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Sepal")

targets = ["Setosa", "Versicolor", "Virginica"]
colors = ["r", "g", "b"]

for target, color in zip(targets, colors):
    indicesToKeep = df["variety"] == target
    ax.scatter(
        df.loc[indicesToKeep, "sepal.length"],
        df.loc[indicesToKeep, "sepal.width"],
        c=color,
        s=50,
    )
ax.set_title("Original Dataset")
ax.legend(targets)
ax.grid()

plt.show()

features = ["sepal.length", "sepal.width"]
x = df[features].values

y = df["variety"].values

# Scales each feature to have mean of 0 and sd of 1
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("Sepal Length (Normalized)")
ax.set_ylabel("Sepal Width (Normalized)")
ax.set_title("Z-Core Scaled Dataset")

targets = ["Setosa", "Versicolor", "Virginica"]
colors = ["r", "g", "b"]

for target, color in zip(targets, colors):
    indicesToKeep = df["variety"] == target
    ax.scatter(
        x_normalized[indicesToKeep, 0],
        x_normalized[indicesToKeep, 1],
        c=color,
        s=50,
    )
ax.legend(targets)
ax.grid()

plt.show()

# Scaler transforms features by scaling each feature to a given range,
# It computes minimum and maximum values of each feature and
# then scales the values to the specified range
# Default is 0 an 1
scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("Sepal Length (Normalized)")
ax.set_ylabel("Sepal Width (Normalized)")
ax.set_title("Min-Max Normalized Dataset")

targets = ["Setosa", "Versicolor", "Virginica"]
colors = ["r", "g", "b"]

for target, color in zip(targets, colors):
    indicesToKeep = df["variety"] == target
    ax.scatter(
        x_normalized[indicesToKeep, 0],
        x_normalized[indicesToKeep, 1],
        c=color,
        s=50,
    )
ax.legend(targets)
ax.grid()

plt.show()
