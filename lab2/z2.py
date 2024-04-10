import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


df = pd.read_csv("./iris1.csv")

features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:, ["variety"]].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

retained_variance = np.sum(pca.explained_variance_ratio_)
percentage_retained = retained_variance * 100
print(percentage_retained)

principalDf = pd.DataFrame(
    data=principalComponents, columns=["principal component 1", "principal component 2"]
)

finalDf = pd.concat([principalDf, df[["variety"]]], axis=1)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize=15)
ax.set_ylabel("Principal Component 2", fontsize=15)
ax.set_title("2 component PCA", fontsize=20)

targets = ["Setosa", "Versicolor", "Virginica"]
colors = ["r", "g", "b"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf["variety"] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "principal component 1"],
        finalDf.loc[indicesToKeep, "principal component 2"],
        c=color,
        s=50,
    )
ax.legend(targets)
ax.grid()

plt.show()


pca = PCA(n_components=3)  # Setting n_components to 3 for three principal components
principalComponents = pca.fit_transform(x)

retained_variance = np.sum(pca.explained_variance_ratio_)
percentage_retained = retained_variance * 100
print(percentage_retained)

principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2", "PC3"])

finalDf = pd.concat([principalDf, df[["variety"]]], axis=1)

# Creating the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("3D PCA")

for target, color in zip(targets, colors):
    indicesToKeep = finalDf["variety"] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "PC1"],
        finalDf.loc[indicesToKeep, "PC2"],
        finalDf.loc[indicesToKeep, "PC3"],
        c=color,
        s=50,
        label=target,
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()
