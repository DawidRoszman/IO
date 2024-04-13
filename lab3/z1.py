import pandas as pd
from sklearn.model_selection import train_test_split


# versicolor slen_min = 4.9, slen_max=6.9
# virginica slen_min = 4.9
def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    if petal_length <= 2:
        return "Iris-setosa"
    elif petal_length < 4.5:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"


df = pd.read_csv("./Iris.csv")

(train_set, test_set) = train_test_split(
    df.values, train_size=0.7, random_state=285808)

print(*sorted(train_set, key=lambda x: x[5]), sep="\n")

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(*test_set[i, 1:5]) == test_set[i, 5]:
        good_predictions += 1

print(good_predictions)
print(good_predictions / len * 100, "%")
