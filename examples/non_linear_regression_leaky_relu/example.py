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
    npnn.LeakyReLU(10),
    npnn.FullyConn(10, 20),
    npnn.LeakyReLU(20),
    npnn.FullyConn(20, 40),
    npnn.LeakyReLU(40),
    npnn.FullyConn(40, 80),
    npnn.LeakyReLU(80),
    npnn.FullyConn(80, 40),
    npnn.LeakyReLU(40),
    npnn.FullyConn(40, 20),
    npnn.LeakyReLU(20),
    npnn.FullyConn(20, 10),
    npnn.LeakyReLU(10),
    npnn.FullyConn(10, 1),
    npnn.Linear(1)
])

model.loss_layer = npnn.loss_layer.RMSLoss(1)

optimizer = npnn.optimizer.Adam(model, alpha=1e-3)  # LeakyReLU

optimizer.dataset = npnn_datasets.NoisySine()

################################################################################

plt.ion()
plt.show()
plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

loss_x = []
train_loss_y = []
validation_loss_y = []

for episode in np.arange(1000):

    # plot the green dataset points ...
    x = optimizer.dataset.x_data
    z = optimizer.predict(x)
    ax1.cla()
    ax1.scatter(optimizer.dataset.x_data, optimizer.dataset.y_data, s=2, c='tab:green')
    ax1.plot(x, z)

    # step the optimizer ...
    optimizer.step()

    # append the optimizer step train loss ...
    loss_x.append(episode)
    tloss = np.mean(optimizer.loss)
    train_loss_y.append(tloss)

    # calculate and append the validation loss ...
    vloss = np.mean(
        optimizer.calculate_loss(
            optimizer.dataset.x_validation_data,
            optimizer.dataset.y_validation_data
        )
    )
    validation_loss_y.append(vloss)

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {2:8.6f}, vloss = {2:8.6f}".format(episode, tloss, vloss))

    # print the train loss (blue) and validation loss (orange) ...
    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('loss')
    ax2.set_yscale('log')
    ax2.set_ylim((min(train_loss_y)/2.0, max(train_loss_y)*2.0))
    ax2.plot(loss_x, train_loss_y, loss_x, validation_loss_y)

    plt.draw()
    plt.savefig('png/episode{0:04d}.png'.format(episode))  # save png to create mp4 later on
    plt.pause(0.001)

input("Press Enter to close ...")

