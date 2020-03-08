#!/usr/bin/env python

import sys
sys.path.append("../../")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import numpy_neural_network as npnn
import npnn_datasets

matplotlib.rcParams['toolbar'] = 'None'

################################################################################

model = npnn.network.Model([
    npnn.FullyConn(1, 10),
    npnn.Tanh(10),
    npnn.FullyConn(10, 20),
    npnn.Tanh(20),
    npnn.FullyConn(20, 40),
    npnn.Tanh(40),
    npnn.FullyConn(40, 80),
    npnn.Tanh(80),
    npnn.FullyConn(80, 40),
    npnn.Tanh(40),
    npnn.FullyConn(40, 20),
    npnn.Tanh(20),
    npnn.FullyConn(20, 10),
    npnn.Tanh(10),
    npnn.FullyConn(10, 1),
    npnn.Linear(1)
])

model.loss_layer = npnn.loss_layer.RMSLoss(1)

#optimizer = npnn.optimizer.SGD(model, alpha=1e-2)  # ReLU seems to need a little bit larger alpha
#optimizer = npnn.optimizer.SGD(model, alpha=2e-3)  # Tanh
#optimizer = npnn.optimizer.RMSprop(model, alpha=5e-5)  # ReLU
#optimizer = npnn.optimizer.Adam(model, alpha=5e-3)  # Sigmoid
optimizer = npnn.optimizer.Adam(model, alpha=2e-4)  # Tanh
#optimizer = npnn.optimizer.Adam(model, alpha=1e-3)  # LeakyReLU

optimizer.dataset = npnn_datasets.NoisySine()

################################################################################

plt.ion()
plt.show()
plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

loss_x = []
train_loss_y = []

for episode in np.arange(200):

    optimizer.step()
    print("episide = {0:5d}, alpha = {1:8.6f}, loss = {2:8.6f}".format(episode, optimizer.alpha, np.mean(optimizer.loss)))

    ax1.cla()
    ax1.scatter(optimizer.dataset.x_data, optimizer.dataset.y_data, s=2, c='tab:green')

    x = optimizer.dataset.x_data
    z = optimizer.predict(x)
    ax1.plot(x, z)

    loss_x.append(episode)
    train_loss_y.append(np.mean(optimizer.loss))

    ax2.cla()
    ax2.set_ylim(0, 0.1+max(train_loss_y))
    ax2.set_xlabel('episode')
    ax2.set_ylabel('loss')
    ax2.plot(loss_x, train_loss_y)

    plt.draw()
    plt.savefig('video/episode{0:04d}.png'.format(episode))
    plt.pause(0.001)

input("Press Enter to close ...")

