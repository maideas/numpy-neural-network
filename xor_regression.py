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

################################################################################
# the neural network model ...

model = npnn.network.Model([
    npnn.FullyConn(2, 4),
    npnn.LeakyReLU(4),
    npnn.FullyConn(4, 4),
    npnn.LeakyReLU(4),
    npnn.FullyConn(4, 1),
    npnn.Linear(1)
])

model.loss_layer = npnn.loss_layer.RMSLoss(1)
optimizer = npnn.optimizer.Adam(model, alpha=1e-3)

################################################################################
# the data and it's preparation ...

x_batch = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_batch = np.array([0, 1, 1, 0])

x_batch_norm = normalize(x_batch)

################################################################################
# the training loop ...

for episode in np.arange(2000):
    optimizer.step(x_batch_norm, y_batch)
    print("episide = {0:5d}, loss = {1:8.6f}".format(episode, np.mean(optimizer.loss)))

################################################################################
# prediction using the trained network ...

z = np.zeros(y_batch.shape)
for n in np.arange(len(x_batch_norm)):
    z[n] = model.predict(x_batch_norm[n])

print("x={0}".format(x_batch))
print("target={0}".format(y_batch))
print("predicted={0}".format(z))

