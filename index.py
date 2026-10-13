
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load the Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# 2. Display the first two images and their labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

fig, axes = plt.subplots(1, 2, figsize=(6, 3))

for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]} - {class_names[y_train[i]]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# 3. Select ONLY the first two training instances
# Use one pixel, at row 14 and column 14
X = X_train[:2, 14, 14].astype(np.float64).reshape(2, 1)
X = X / 255.0

# Targets are the first two labels
y = y_train[:2].astype(np.float64).reshape(2, 1)

print("Selected pixel values:", X.flatten())
print("Target labels:", y.flatten())

# 4. Initialize weight (k) and bias (m)
np.random.seed(42)
k = np.random.randn()
m = 0.0

# 5. Set gradient descent parameters
learning_rate = 0.01
epochs = 10000
n = len(X)

loss_history = []

# 6. Train the one-neuron linear model
for epoch in range(epochs):

    # Forward pass: prediction = k*x + m
    y_pred = k * X + m

    # Mean squared error loss
    error = y_pred - y
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # Gradients of the MSE loss
    dk = (2 / n) * np.sum(error * X)
    dm = (2 / n) * np.sum(error)

    # Gradient descent update
    k = k - learning_rate * dk
    m = m - learning_rate * dm

# 7. Print the final results
y_pred = k * X + m

print("\nNumPy model")
print("Final weight k:", k)
print("Final bias m:", m)
print("Final loss:", np.mean((y_pred - y) ** 2))

for i in range(n):
    print(
        f"Image {i}: true label={y[i, 0]:.0f}, "
        f"predicted label={y_pred[i, 0]:.4f}"
    )

# 8. Plot the training loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title("NumPy model: training loss")
plt.grid(True)
plt.show()

# Task 1(ii): Train the same one-neuron model using Keras

# Use the same two images, same pixel and same labels
X_keras = X_train[:2, 14, 14].astype(np.float32).reshape(2, 1)
X_keras = X_keras / 255.0

y_keras = y_train[:2].astype(np.float32).reshape(2, 1)

# Create a neural network with one input and one output neuron
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1, activation="linear")
])

# Configure the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="mse"
)
#D######################################################################################3
##TASK 2######
(X_train_full, y_train_full), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Split the original training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)