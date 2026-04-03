import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Original Model Accuracy:", acc)

for layer in model.layers:
    weights = layer.get_weights()
    new_weights = [w.astype(np.float16) for w in weights]
    layer.set_weights(new_weights)
# Evaluate again
loss_q, acc_q = model.evaluate(x_test, y_test)
print("Quantized Model Accuracy:", acc_q)
# Add noise to input
noise = np.random.normal(0, 0.1, x_test.shape)
x_test_noisy = x_test + noise
# Clip values
x_test_noisy = np.clip(x_test_noisy, 0, 1)
# Evaluate on noisy data
loss_adv, acc_adv = model.evaluate(x_test_noisy, y_test)
print("Accuracy under attack:", acc_adv)