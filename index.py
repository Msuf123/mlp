
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

# Normalize pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten images from (28, 28) to 784 features
X_train = X_train.reshape(-1, 784)
X_val = X_val.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)
print("Number of classes:", len(np.unique(y_train)))

# Task 2.2: Build the shallow MLP

tf.keras.utils.set_random_seed(42)

model_mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,), name="input_layer"),
    tf.keras.layers.Dense(100, activation="relu", name="hidden_layer"),
    tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
], name="model_mlp")

# Display the architecture
model_mlp.summary()

# Inspect the weights and biases of each Dense layer
for layer in model_mlp.layers:
    weights, biases = layer.get_weights()

    print(f"\nLayer: {layer.name}")
    print("Weight matrix shape:", weights.shape)
    print("Bias vector shape:", biases.shape)
    print("Number of weights:", weights.size)
    print("Number of biases:", biases.size)
    print("First five weights:", weights.flatten()[:5])
    print("First five biases:", biases[:5])

# Task 2.3: Compile the model

model_mlp.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the network
history_mlp = model_mlp.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Task 2.4: Plot loss and accuracy

epochs_range = range(1, len(history_mlp.history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history_mlp.history["loss"], label="Training")
axes[0].plot(epochs_range, history_mlp.history["val_loss"], label="Validation")
axes[0].set_title("MLP: loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_range, history_mlp.history["accuracy"], label="Training")
axes[1].plot(epochs_range, history_mlp.history["val_accuracy"], label="Validation")
axes[1].set_title("MLP: accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Task 2.5: Evaluate on the independent test set

test_loss, test_accuracy = model_mlp.evaluate(
    X_test, y_test, verbose=0
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict class probabilities
probabilities = model_mlp.predict(X_test, verbose=0)

# Convert probabilities into predicted class indices
y_pred_mlp = np.argmax(probabilities, axis=1)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred_mlp,
    labels=np.arange(10),
    target_names=class_names,
    zero_division=0
))

# Display probabilities for the first five test images
for i in range(5):
    predicted_class = np.argmax(probabilities[i])

    print(f"\nImage {i}")
    print("True class:", class_names[y_test[i]])
    print("Predicted class:", class_names[predicted_class])
    print("Predicted probability:", probabilities[i, predicted_class])

    for j, name in enumerate(class_names):
        print(f"  {name}: {probabilities[i, j]:.4f}")

# Task 2.6: Plot the confusion matrix

cm_mlp = confusion_matrix(
    y_test,
    y_pred_mlp,
    labels=np.arange(10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_mlp,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Fashion-MNIST: MLP confusion matrix")
plt.tight_layout()
plt.show()

# Calculate recall for each class
class_recall = np.diag(cm_mlp) / cm_mlp.sum(axis=1)

print("\nRecall by class:")
for name, recall in zip(class_names, class_recall):
    print(f"{name}: {recall:.4f}")

print("\nClasses with lowest recall:")
for idx in np.argsort(class_recall)[:3]:
    print(f"{class_names[idx]}: {class_recall[idx]:.4f}")

####Taks 3

# TASK 3: Logistic regression as a neural network

tf.keras.utils.set_random_seed(42)

model_log_reg = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,), name="input_layer"),
    tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
], name="model_log_reg")

# Inspect architecture and parameters
model_log_reg.summary()

model_log_reg.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_log_reg = model_log_reg.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate on the same independent test set
log_reg_loss, log_reg_accuracy = model_log_reg.evaluate(
    X_test, y_test, verbose=0
)

print(f"Logistic regression test loss: {log_reg_loss:.4f}")
print(f"Logistic regression test accuracy: {log_reg_accuracy:.4%}")

# Predictions and classification report
probabilities_log_reg = model_log_reg.predict(X_test, verbose=0)
y_pred_log_reg = np.argmax(probabilities_log_reg, axis=1)

print(classification_report(
    y_test,
    y_pred_log_reg,
    labels=np.arange(10),
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
cm_log_reg = confusion_matrix(
    y_test, y_pred_log_reg, labels=np.arange(10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_log_reg,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Fashion-MNIST: Logistic regression confusion matrix")
plt.tight_layout()
plt.show()
