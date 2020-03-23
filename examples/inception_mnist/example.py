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
    npnn.Inception((28, 28, 1),
           2,
        2, 4,
        2, 4,
           2
    ),
    npnn.MaxPool(shape_in=(28, 28, 12), shape_out=(14, 14, 12), kernel_size=2),
    npnn.Inception((14, 14, 12),
            2,
        4,  6,
        4,  6,
            2
    ),
    npnn.MaxPool(shape_in=(14, 14, 16), shape_out=(7, 7, 16), kernel_size=2),
    npnn.Inception((7, 7, 16),
           2,
        6, 6,
        6, 6,
           2
    ),
    npnn.Dense(7*7*16, 140),
    npnn.LeakyReLU(140),
    npnn.Dense(140, 10),
    npnn.Softmax(10)
])

model.loss_layer = npnn.loss_layer.CrossEntropyLoss(10)

optimizer = npnn.optimizer.Adam(model, alpha=1e-3)

optimizer.dataset = npnn_datasets.MNIST()

################################################################################

plt.ion()
plt.show()
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

fig2, axs = plt.subplots(4, 3, figsize=(10,8))

episodes = []

mini_train_loss = []
mini_validation_loss = []
mini_train_accuracy = []
mini_validation_accuracy = []

for episode in np.arange(400):

    # step the optimizer ...
    optimizer.step()
    episodes.append(episode)

    #===========================================================================
    # mini batch loss and accuracy
    #===========================================================================

    tloss = np.mean(optimizer.loss)
    taccuracy = optimizer.accuracy

    mini_train_loss.append(tloss)
    mini_train_accuracy.append(taccuracy)

    ax1.cla()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('train mini-batch loss')
    ax1.set_yscale('log')
    ax1.set_ylim((min(mini_train_loss)/2.0, max(mini_train_loss)*2.0))
    ax1.plot(episodes, mini_train_loss)

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('train mini-batch accuracy [%]')
    ax2.set_ylim(-5, 105)
    ax2.plot(episodes, mini_train_accuracy)

    #===========================================================================
    # complete dataset loss and accuracy
    #===========================================================================

    x_validation_batch, t_validation_batch = optimizer.dataset.get_validation_batch()
    y_validation_batch = optimizer.predict(x_validation_batch, t_validation_batch)

    vloss = np.mean(optimizer.loss)
    vaccuracy = optimizer.accuracy

    mini_validation_loss.append(vloss)
    mini_validation_accuracy.append(vaccuracy)

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {1:5.3f}, vloss = {2:5.3f}, taccuracy = {3:5.3f}%, vaccuracy = {4:5.3f}%".format(
        episode, tloss, vloss, taccuracy, vaccuracy
    ))

    ax3.cla()
    ax3.set_xlabel('episode')
    ax3.set_ylabel('validation mini-batch loss')
    ax3.set_yscale('log')
    ax3.set_ylim((min(mini_validation_loss)/2.0, max(mini_validation_loss)*2.0))
    ax3.plot(episodes, mini_validation_loss, c='tab:orange')

    ax4.cla()
    ax4.set_xlabel('episode')
    ax4.set_ylabel('validation mini-batch accuracy [%]')
    ax4.set_ylim(-5, 105)
    ax4.plot(episodes, mini_validation_accuracy, c='tab:orange')

    #===========================================================================
    # network output plots (validation batch based)
    #===========================================================================

    k = np.arange(x_validation_batch.shape[0])

    for n in np.arange(4):
        for m in np.arange(3):
            c = n * 3 + m
            if c < 10:
                axs[n, m].cla()
                axs[n, m].set_ylim(-0.1, 1.1)
                axs[n, m].scatter(k, t_validation_batch[:,c], s=10, c='tab:green')
                axs[n, m].scatter(k, y_validation_batch[:,c], s=10, c='tab:orange')

    #===========================================================================
    # draw and save PNG to generate video files later on
    #===========================================================================

    plt.draw()
    fig1.savefig('png/episode{0:04d}.png'.format(episode))
    fig2.savefig('png_2/episode{0:04d}.png'.format(episode))
    plt.pause(0.001)

input("Press Enter to close ...")

