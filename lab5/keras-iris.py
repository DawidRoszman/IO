import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

# Define the model
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(y_encoded.shape[1], activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adamw", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save("iris_model.keras")

# Plot and save the model architecture
plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

# Co robi StandardScaler? Jak transformowane są dane liczbowe?
# StandardScaler przeskalowuje dane tak, aby miały średnią 0 i wariancję 1.
# Transformacja polega na odjęciu średniej i podzieleniu przez odchylenie standardowe.

# Czym jest OneHotEncoder (i kodowanie "one hot" ogólnie)? Jak etykiety klas są transformowane przez ten encoder?
# A: OneHotEncoder przekształca etykiety klas na wektory binarne, gdzie każda klasa jest reprezentowana przez wektor
# złożony z samych zer i jednej jedynki na pozycji odpowiadającej indeksowi klasy.
# Na przykład, dla trzech klas: 0, 1, 2, etykieta 0 jest kodowana jako [1, 0, 0], etykieta 1 jako [0, 1, 0], a etykieta 2 jako [0, 0, 1].
# W ten sposób model może łatwo przewidywać klasę, która ma najwyższe prawdopodobieństwo.

# Model ma 4 warstwy: wejściową, dwie ukryte warstwy z 64 neuronami każda i warstwę wyjściową. Ile
# neuronów ma warstwa wejściowa i co oznacza X_train.shape[1]? Ile neuronów ma warstwa wyjściowa i co
#  oznacza y_encoded.shape[1]?
# Warstwa wejściowa ma tyle neuronów, ile cech ma każdy przykład w zbiorze danych. X_train.shape[1] oznacza liczbę cech w każdym przykładzie.
# Warstwa wyjściowa ma tyle neuronów, ile klas jest w zbiorze danych. y_encoded.shape[1] oznacza liczbę klas w zbiorze danych.
# W tym przypadku, warstwa wejściowa ma 4 neurony (4 cechy w zbiorze danych iris) i warstwa wyjściowa ma 3 neurony (3 klasy w zbiorze danych iris).

# Czy funkcja aktywacji relu jest najlepsza do tego zadania?
# Funkcja aktywacji relu jest często stosowana w warstwach ukrytych sieci neuronowych, ponieważ pomaga w zapobieganiu zanikaniu gradientu.
# W przypadku tego zadania, funkcja aktywacji relu jest odpowiednia, ponieważ działa dobrze w przypadku problemów klasyfikacji.
