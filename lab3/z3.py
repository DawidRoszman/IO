import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("./Iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285808)

train_inputs = train_set[:, 1:5]
train_classes = train_set[:, 5]
test_inputs = test_set[:, 1:5]
test_classes = test_set[:, 5]

good_predictions = 0

neighbours = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(train_inputs)
distances, indices = neighbours.kneighbors(test_inputs)
print(distances)
print(indices)
