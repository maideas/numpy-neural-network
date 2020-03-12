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

optimizer = npnn.optimizer.Adam(model, alpha=1e-2)

optimizer.dataset = npnn_datasets.XORTwoClasses()

################################################################################

plt.ion()
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

loss_x = []
train_loss_y0 = []
train_loss_y1 = []
validation_loss_y0 = []
validation_loss_y1 = []
train_accuracy_y = []
validation_accuracy_y = []

for episode in np.arange(500):

    # step the optimizer ...
    optimizer.step()

    # append the optimizer step train loss ...
    loss_x.append(episode)
    tloss = optimizer.loss
    taccuracy = optimizer.accuracy
    train_loss_y0.append(tloss[0])
    train_loss_y1.append(tloss[1])
    train_accuracy_y.append(taccuracy * 100)

    # calculate and append the validation loss ...
    vloss, vaccuracy = optimizer.calculate_loss(
        optimizer.dataset.x_validation_data,
        optimizer.dataset.y_validation_data
    )
    validation_loss_y0.append(vloss[0])
    validation_loss_y1.append(vloss[1])
    validation_accuracy_y.append(vaccuracy * 100)

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {2:8.6f}, vloss = {2:8.6f}".format(episode, tloss[0], vloss[0]))

    # print the train loss (blue) and validation loss (orange) ...
    ax1.cla()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('class 0 loss')
    ax1.set_yscale('log')
    ax1.set_ylim((min(train_loss_y0)/2.0, max(train_loss_y0)*2.0))
    ax1.plot(loss_x, train_loss_y0, loss_x, validation_loss_y0)

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('class 1 loss')
    ax2.set_yscale('log')
    ax2.set_ylim((min(train_loss_y1)/2.0, max(train_loss_y1)*2.0))
    ax2.plot(loss_x, train_loss_y1, loss_x, validation_loss_y1)

    ax3.cla()
    ax3.set_xlabel('episode')
    ax3.set_ylabel('accuracy')
    ax3.set_ylim(-10, 110)
    ax3.plot(loss_x, train_accuracy_y, loss_x, validation_accuracy_y)

    plt.draw()
    plt.savefig('png/episode{0:04d}.png'.format(episode))  # save png to create mp4 later on
    plt.pause(0.001)

input("Press Enter to close ...")

