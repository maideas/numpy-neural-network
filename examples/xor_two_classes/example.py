#!/usr/bin/env python

import sys
sys.path.append("../../")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy_neural_network as npnn
import npnn_datasets

matplotlib.rcParams['toolbar'] = 'None'

################################################################################

model = npnn.network.Model([
    npnn.FullyConn(2, 4),
    npnn.LeakyReLU(4),
    npnn.FullyConn(4, 4),
    npnn.LeakyReLU(4),
    npnn.FullyConn(4, 2),
    npnn.Softmax(2)
])

model.loss_layer = npnn.loss_layer.CrossEntropyLoss(2)

optimizer = npnn.optimizer.Adam(model, alpha=2e-2)

optimizer.dataset = npnn_datasets.XORTwoClasses()

################################################################################

plt.ion()
plt.show()
plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

loss_x = []
train_loss_y0 = []
train_loss_y1 = []
validation_loss_y0 = []
validation_loss_y1 = []

for episode in np.arange(500):

    # step the optimizer ...
    optimizer.step()

    # append the optimizer step train loss ...
    loss_x.append(episode)
    tloss = optimizer.loss
    train_loss_y0.append(tloss[0])
    train_loss_y1.append(tloss[1])

    # calculate and append the validation loss ...
    vloss = optimizer.calculate_loss(
        optimizer.dataset.x_validation_data,
        optimizer.dataset.y_validation_data
    )
    validation_loss_y0.append(vloss[0])
    validation_loss_y1.append(vloss[1])

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {2:8.6f}, vloss = {2:8.6f}".format(episode, tloss[0], vloss[0]))

    # print the train loss (blue) and validation loss (orange) ...
    ax1.cla()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('loss class 0')
    ax1.set_yscale('log')
    ax1.set_ylim((min(train_loss_y0)/2.0, max(train_loss_y0)*2.0))
    ax1.plot(loss_x, train_loss_y0, loss_x, validation_loss_y0)

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('loss class 1')
    ax2.set_yscale('log')
    ax2.set_ylim((min(train_loss_y1)/2.0, max(train_loss_y1)*2.0))
    ax2.plot(loss_x, train_loss_y1, loss_x, validation_loss_y1)

    plt.draw()
    plt.savefig('png/episode{0:04d}.png'.format(episode))  # save png to create mp4 later on
    plt.pause(0.001)

input("Press Enter to close ...")

