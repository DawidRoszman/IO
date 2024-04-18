from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

datasets = train_test_split(iris.data, iris.target, test_size=0.3)

train_data, test_data, train_classes, test_classes = datasets

scaler = StandardScaler()

scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

accuracies = []

# Dwa neurony wejściowe
mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp.fit(train_data, train_classes)

predictions_train = mlp.predict(train_data)
print("Accuracy on training data:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_data)
print("Accuracy for test classes:", accuracy_score(predictions_test, test_classes))
accuracies.append(accuracy_score(predictions_test, test_classes))

print(confusion_matrix(predictions_test, test_classes))
print(classification_report(predictions_test, test_classes))

# Trzy neurony wejściowe
mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)
mlp.fit(train_data, train_classes)

predictions_train = mlp.predict(train_data)
print("Accuracy on training data:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_data)
print("Accuracy for test classes:", accuracy_score(predictions_test, test_classes))
accuracies.append(accuracy_score(predictions_test, test_classes))

print(confusion_matrix(predictions_test, test_classes))
print(classification_report(predictions_test, test_classes))

# Dwie warstwy ukryte po 3 neurony
mlp = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=1000)
mlp.fit(train_data, train_classes)

predictions_train = mlp.predict(train_data)
print("Accuracy on training data:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_data)
print("Accuracy for test classes:", accuracy_score(predictions_test, test_classes))
accuracies.append(accuracy_score(predictions_test, test_classes))

print(confusion_matrix(predictions_test, test_classes))
print(classification_report(predictions_test, test_classes))


# Best model
if max(accuracies) == accuracies[0]:
    print("Best model: two input neurons")
elif max(accuracies) == accuracies[1]:
    print("Best model: three input neurons")
else:
    print("Best model: two hidden layers with three neurons each")
