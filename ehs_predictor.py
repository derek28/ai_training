#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

def import_data(filename):
    fd = open(filename, "r")
    nSamps = sum(1 for line in fd)
    fd.close()
    fd = open(filename, "r")
    data = np.zeros([nSamps, 105])
    i = 0;
    for line in fd:
        data[i] = line.split(",")
        i += 1
    fd.close()
    print("Number of samples read: ", nSamps)
    return data, nSamps

# set up training
model = tf.keras.Sequential([
# Adds a densely-connected layer with 104 units to the model:
    layers.Dense(104, activation = 'relu', 
                    input_shape = (104,), 
                    kernel_initializer = 'random_normal',
                    bias_initializer = 'zeros'
                ),
# Add another:
    layers.Dense(104, activation = 'relu',
                    kernel_initializer = 'random_normal',
                    bias_initializer = 'zeros'
                ),
# Add an output layer with 1 output, the prediction of hand strength:
    layers.Dense(1)])

print(model.summary())

# Configure and compile module
model.compile(optimizer = tf.keras.optimizers.Adam(0.01), loss = 'mse')

# import training data
samples, num = import_data("training.dat")
hands = samples[:, :-1]
strength = samples[:, -1]
print(hands[:5, :])
print(strength[:5])

# devide data into training, validation and test
posTrain = int(0.8 * num);           # 80% as training data
posValid = int(0.9 * num);  # 10% as validation, 10% as test

hands_train = hands[:posTrain, :]
hands_valid = hands[posTrain : posValid, :]
hands_test = hands[posValid : , :]
strength_train = strength[ : posTrain]
strength_valid = strength[posTrain : posValid]
strength_test = strength[posValid :]

print("Number of training samples:", len(strength_train))
print("Number of validation samples:", len(strength_valid))
print("Number of test samples:", len(strength_test))

# training model
model.fit(hands_train, strength_train, 
            epochs = 5,
            validation_data = (hands_valid, strength_valid))

# evaluate model
results = model.evaluate(hands_test, strength_test)
print('test loss = ', results)

predictions = model.predict(hands_test[:10, :])
print("Predictions vs Real Strength")
for i in range(10):
    print(predictions[i], strength_test[i])
