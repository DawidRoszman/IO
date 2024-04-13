import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv("./Iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285808)

train_inputs = train_set[:, 1:5]
train_classes = train_set[:, 5]
test_inputs = test_set[:, 1:5]
test_classes = test_set[:, 5]

good_predictions = 0

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)

tree.plot_tree(clf)
tree.export_text(clf)


# Test the classifier
for i in range(len(test_inputs)):
    prediction = clf.predict([test_inputs[i]])
    if prediction == test_classes[i]:
        good_predictions += 1

score = clf.score(test_inputs, test_classes)

print("Accuracy: ", good_predictions / len(test_inputs) * 100, "%")
print("Score: ", score)
