import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense

model = Sequential()
model.add(Input(shape=(64, 64, 3)))  # Define the input shape here
model.add(Conv2D(32, (3, 3)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import numpy as np

# Example data: replace with your actual data
num_samples = 1000
X_train = np.random.rand(num_samples, 64, 64, 3)  # 1000 samples of 64x64 RGB images
y_train = np.random.randint(2, size=(num_samples, 10))  # Binary labels for 10 classes

X_val = np.random.rand(200, 64, 64, 3)  # Validation data
y_val = np.random.randint(2, size=(200, 10))  # Validation labels

X_test = np.random.rand(200, 64, 64, 3)  # Test data
y_test = np.random.randint(2, size=(200, 10))  # Test labels

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
