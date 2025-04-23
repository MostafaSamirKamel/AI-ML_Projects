# train_model.py

import tensorflow as tf
from tensorflow import keras
import numpy as np


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images from (28, 28) to (784,)
x_train_flatten = x_train.reshape(len(x_train), 28 * 28)
x_test_flatten = x_test.reshape(len(x_test), 28 * 28)

# Build a more complex model with Batch Normalization and Dropout layers
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),
    keras.layers.BatchNormalization(),  # Batch Normalization
    keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting

    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),  # Batch Normalization
    keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting

    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),  # Batch Normalization
    keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting

    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons for digits 0-9
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)



# Train the model
model.fit(x_train_flatten, y_train, epochs=30, validation_data=(x_test_flatten, y_test))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test_flatten, y_test)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('model.h5')
print("model saved as 'model.h5'")