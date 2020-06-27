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

def export_weights(filename, weights):
    fout = open(filename, "w")

    # print weights
    for x in weights[0]:
        for y in x:
            fout.write(str(y))
            fout.write(" ")
        fout.write("\n")
        
    # print bias term
    for x in weights[1]:
        fout.write(str(x))
        fout.write(" ")
    fout.close()

# set up training
inputs = keras.Input(shape = (104,))
l1 = layers.Dense(104, activation = 'relu', 
                    kernel_initializer = 'random_normal',
                    bias_initializer = 'zeros'
                )
a1 = l1(inputs)
l2 = layers.Dense(104, activation = 'relu',
                    kernel_initializer = 'random_normal',
                    bias_initializer = 'zeros'
                )
a2 = l2(a1)
outputs = layers.Dense(1)(a2)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

# Configure and compile module
model.compile(optimizer = tf.keras.optimizers.Adam(0.01), loss = 'mse')

# import training data
samples, num = import_data("training.dat")
hands = samples[:, :-1]
strength = samples[:, -1]
#print(hands[:5, :])
#print(strength[:5])

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
            epochs = 30,
            validation_data = (hands_valid, strength_valid))

# evaluate model
results = model.evaluate(hands_test, strength_test)
print('test loss = ', results)

predictions = model.predict(hands_test[-20:, :])
#for j in range(8):
#    for i in range(13):
#        print(hands_test[-1][i + 13 * j], end = " ")
#    print("\n")

a = l1(hands_test[-1:, :])
print(a)
b = l2(a)
print(b)

print("Predictions vs Real Strength")
for i in range(20):
    print(predictions[i], strength_test[-20+i])

weights = model.layers[1].get_weights()
print("Length of weights from layer 1:", len(weights))
export_weights("weight1.txt", weights)

weights = model.layers[2].get_weights()
print("Length of weights from layer 2:", len(weights))
export_weights("weight2.txt", weights)

weights = model.layers[3].get_weights()
print("Length of weights from layer 3:", len(weights))
export_weights("weight3.txt", weights)
