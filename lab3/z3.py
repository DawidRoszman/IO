import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def evaluate_classifier(
    classifier, train_inputs, train_classes, test_inputs, test_classes
):
    classifier.fit(train_inputs, train_classes)
    test_pred = classifier.predict(test_inputs)
    accuracy = accuracy_score(test_classes, test_pred)
    confusion = confusion_matrix(test_classes, test_pred)
    return accuracy, confusion


df = pd.read_csv("./Iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285808)

train_inputs = train_set[:, 1:5]
train_classes = train_set[:, 5]
test_inputs = test_set[:, 1:5]
test_classes = test_set[:, 5]

neighbour_cout = [3, 5, 11]
knn_accuracies = []
knn_confusions = []

for n in neighbour_cout:
    classifier = KNeighborsClassifier(n_neighbors=n, algorithm="ball_tree")
    accuracy, confusion = evaluate_classifier(
        classifier, train_inputs, train_classes, test_inputs, test_classes
    )
    knn_accuracies.append(accuracy)
    knn_confusions.append(confusion)
    print(f"Accuracy for {n} neighbours: {accuracy}")
    print(f"Confusion matrix for {n} neighbours: {confusion}")

nb_classifier = GaussianNB()
nb_accuracy, nb_confusion = evaluate_classifier(
    nb_classifier, train_inputs, train_classes, test_inputs, test_classes
)
print(f"Accuracy for Naive Bayes: {nb_accuracy}")
print(f"Confusion matrix for Naive Bayes: {nb_confusion}")

all_accuracies = [nb_accuracy] + knn_accuracies
best_classifier_idx = np.argmax(all_accuracies)
best_classifier_name = (
    "Naive Bayes"
    if best_classifier_idx == 0
    else f"KNN with {neighbour_cout[best_classifier_idx - 1]} neighbours"
)
print(f"The best classifier is: {best_classifier_name}")
