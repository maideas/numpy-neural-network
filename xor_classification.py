#!/usr/bin/env python

import numpy as np
import numpy_neural_network as npnn

################################################################################
# some useful functions ...

def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std
    return x

def class_vector_to_onehot_array(y_classes):
    num_vectors = len(y_classes)
    num_classes = np.max(y_classes) + 1
    y_onehot = np.zeros((num_vectors, num_classes))
    for n in np.arange(num_vectors):
        y_onehot[n][y_classes[n]] = 1
    return y_onehot

################################################################################
# the neural network model ...

model = npnn.network.Model([
    npnn.FullyConn(2, 4),
    npnn.LeakyReLU(4),
    npnn.FullyConn(4, 4),
    npnn.LeakyReLU(4),
    npnn.FullyConn(4, 2),
    npnn.Softmax(2)
])

model.loss_layer = npnn.loss_layer.CrossEntropyLoss(2)
optimizer = npnn.optimizer.Adam(model, alpha=1e-3)

################################################################################
# the data and its preparation ...

x_batch = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_class_batch = np.array([0, 1, 1, 0])

x_batch_norm = normalize(x_batch)
y_batch = class_vector_to_onehot_array(y_class_batch)

################################################################################
# the training loop ...

for episode in np.arange(2000):
    optimizer.step(x_batch_norm, y_batch)
    print("episide = {0:5d}, loss = {1:8.6f}".format(episode, np.mean(optimizer.loss)))

################################################################################
# prediction using the trained network ...

zp = np.zeros(y_batch.shape)
zc = np.zeros(y_class_batch.shape)

for n in np.arange(len(x_batch_norm)):
    zp[n] = model.predict(x_batch_norm[n])
    zc[n] = model.predict_class(x_batch_norm[n])

print("x={0}".format(x_batch))
print("target classes={0}".format(y_class_batch))
print("target onehot={0}".format(y_batch))
print("predicted probabilities={0}".format(zp))
print("predicted classes={0}".format(zc))

