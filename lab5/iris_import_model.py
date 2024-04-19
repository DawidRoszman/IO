import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model

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
# Load the pre-trained model
model = load_model("iris_model.keras")
# Continue training the model for 10 more epochs
# model.fit(X_train, y_train, epochs=10)

# Save the updated model
# model.save("updated_iris_model.keras")
# Evaluate the updated model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Accuracy: {test_accuracy*100:.2f}%")
from sklearn.preprocessing import LabelEncoder

# Create a label encoder
label_encoder = LabelEncoder()

# Fit the encoder on the iris names
label_encoder.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# predict the class of a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # replace with your actual sample

# Preprocess the new sample
new_sample_scaled = scaler.transform(new_sample)

# Predict the class of the new sample
prediction = model.predict(new_sample_scaled)

# The prediction will be a one-hot encoded array, we can use argmax to get the class index
predicted_class = np.argmax(prediction, axis=1)

label = label_encoder.inverse_transform(predicted_class)

print(f"The predicted class is: {predicted_class}")
print(label)
