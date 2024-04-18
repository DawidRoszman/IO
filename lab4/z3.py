from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

df = pd.read_csv("./diabetes1.csv")

datasets = train_test_split(df.iloc[:, 0:8], df.iloc[:, 8], test_size=0.3)

train_data, test_data, train_classes, test_classes = datasets

scaler = StandardScaler()

scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

mlp = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=500, activation="relu")

mlp.fit(train_data, train_classes)

predictions_train = mlp.predict(train_data)
print("Accuracy on training data:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_data)
print("Accuracy for test classes:", accuracy_score(predictions_test, test_classes))

print(confusion_matrix(predictions_test, test_classes))
print(classification_report(predictions_test, test_classes))

# W tym przypadku FN może okazać się gorszym błędem niż FP, ponieważ lepiej zrobić dodakowe badania, niż nie zauważyć choroby.
