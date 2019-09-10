#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
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
    num_classes = int(np.max(y_classes)) + 1
    y_onehot = np.zeros((num_vectors, num_classes))
    for n in np.arange(num_vectors):
        y_onehot[n][int(y_classes[n])] = 1
    return y_onehot

################################################################################
# the neural network model ...

model = npnn.network.Model([
    npnn.Conv2d(shape_in=(10, 10, 1), shape_out=(8, 8, 4), kernel_size=3, stride=1),
    npnn.LeakyReLU(8*8*4),
    npnn.MaxPool(shape_in=(8, 8, 4), shape_out=(4, 4, 4), kernel_size=2),
    npnn.Conv2d(shape_in=(4, 4, 4), shape_out=(2, 2, 4), kernel_size=3, stride=1),
    npnn.LeakyReLU(2*2*4),
    npnn.MaxPool(shape_in=(2, 2, 4), shape_out=(1, 1, 4), kernel_size=2),
    npnn.FullyConn(4, 4),
    npnn.Softmax(4)
])

model.loss_layer = npnn.loss_layer.CrossEntropyLoss(4)
optimizer = npnn.optimizer.Adam(model, alpha=1e-2)

################################################################################
# the data and it's preparation ...

x_batch = np.zeros(((8*8*4), 10, 10, 1))
y_class_batch = np.zeros(8*8*4)

k = 0

# 0 1 1
# 1 0 1
# 1 1 0
for n in np.arange(8):
    for m in np.arange(8):
        x_batch[k, n+0, m+0, 0] = 0
        x_batch[k, n+1, m+0, 0] = 1
        x_batch[k, n+2, m+0, 0] = 1
        x_batch[k, n+0, m+1, 0] = 1
        x_batch[k, n+1, m+1, 0] = 0
        x_batch[k, n+2, m+1, 0] = 1
        x_batch[k, n+0, m+2, 0] = 1
        x_batch[k, n+1, m+2, 0] = 1
        x_batch[k, n+2, m+2, 0] = 0
        y_class_batch[k] = 0
        k += 1

# 0 1 0
# 1 1 1
# 0 1 0
for n in np.arange(8):
    for m in np.arange(8):
        x_batch[k, n+0, m+0, 0] = 0
        x_batch[k, n+1, m+0, 0] = 1
        x_batch[k, n+2, m+0, 0] = 0
        x_batch[k, n+0, m+1, 0] = 1
        x_batch[k, n+1, m+1, 0] = 1
        x_batch[k, n+2, m+1, 0] = 1
        x_batch[k, n+0, m+2, 0] = 0
        x_batch[k, n+1, m+2, 0] = 1
        x_batch[k, n+2, m+2, 0] = 0
        y_class_batch[k] = 1
        k += 1

# 1 0 1
# 0 1 0
# 1 0 1
for n in np.arange(8):
    for m in np.arange(8):
        x_batch[k, n+0, m+0, 0] = 1
        x_batch[k, n+1, m+0, 0] = 0
        x_batch[k, n+2, m+0, 0] = 1
        x_batch[k, n+0, m+1, 0] = 0
        x_batch[k, n+1, m+1, 0] = 1
        x_batch[k, n+2, m+1, 0] = 0
        x_batch[k, n+0, m+2, 0] = 1
        x_batch[k, n+1, m+2, 0] = 0
        x_batch[k, n+2, m+2, 0] = 1
        y_class_batch[k] = 2
        k += 1

# 0 1 0
# 1 0 1
# 0 1 0
for n in np.arange(8):
    for m in np.arange(8):
        x_batch[k, n+0, m+0, 0] = 0
        x_batch[k, n+1, m+0, 0] = 1
        x_batch[k, n+2, m+0, 0] = 0
        x_batch[k, n+0, m+1, 0] = 1
        x_batch[k, n+1, m+1, 0] = 0
        x_batch[k, n+2, m+1, 0] = 1
        x_batch[k, n+0, m+2, 0] = 0
        x_batch[k, n+1, m+2, 0] = 1
        x_batch[k, n+2, m+2, 0] = 0
        y_class_batch[k] = 3
        k += 1

x_batch = normalize(x_batch)
y_batch = class_vector_to_onehot_array(y_class_batch)

################################################################################
# shuffle data items ...

idx = np.random.permutation(y_batch.shape[0])

x_batch = x_batch[idx]
y_batch = y_batch[idx]
y_class_batch = y_class_batch[idx]

################################################################################
# split data items into traning and validation set ...

num_train = int(len(x_batch) * 0.7)

x_batch_train = x_batch[:num_train]
y_batch_train = y_batch[:num_train]
y_class_batch_train = y_class_batch[:num_train]

x_batch_valid = x_batch[num_train:]
y_batch_valid = y_batch[num_train:]
y_class_batch_valid = y_class_batch[num_train:]

################################################################################
# the training loop ...

plt.ion()
plt.show()

train_accuracy = []
valid_accuracy = []

for episode in np.arange(2000):

    optimizer.step(x_batch_train, y_batch_train)
    print("episide = {0:5d}, loss = {1:8.6f}".format(episode, np.mean(optimizer.loss)))

    n = 0
    m = 0
    for k in np.arange(len(x_batch_train)):
        if model.predict_class(x_batch_train[k]) == y_class_batch_train[k]:
            n += 1
        m += 1
    train_accuracy.append(100.0 * n / m)
    #print("training accuracy = {0}".format(train_accuracy[-1]))

    n = 0
    m = 0
    for k in np.arange(len(x_batch_valid)):
        if model.predict_class(x_batch_valid[k]) == y_class_batch_valid[k]:
            n += 1
        m += 1
    valid_accuracy.append(100.0 * n / m)
    #print("validation accuracy = {0}".format(valid_accuracy[-1]))

    plt.cla()
    plt.plot(
        np.arange(0, len(train_accuracy)), train_accuracy,
        np.arange(0, len(valid_accuracy)), valid_accuracy
    )
    plt.draw()
    plt.pause(0.001)

input("Press Enter to close ...")

