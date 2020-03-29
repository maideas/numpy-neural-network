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
    npnn.Conv2D(shape_in=(3, 3, 1), shape_out=(2, 2, 6), kernel_size=2, stride=1),
    npnn.Tanh(2 * 2 * 6),
    npnn.Conv2D(shape_in=(2, 2, 6), shape_out=(1, 1, 2), kernel_size=2, stride=1),
    npnn.Tanh(1 * 1 * 2),
    npnn.UpConv2D(shape_in=(1, 1, 2), shape_out=(2, 2, 6), kernel_size=2, stride=1),
    npnn.Tanh(2 * 2 * 6),
    npnn.UpConv2D(shape_in=(2, 2, 6), shape_out=(3, 3, 1), kernel_size=2, stride=1),
    npnn.Tanh(3 * 3 * 1)
])

model.loss_layer = npnn.loss_layer.RMSLoss(shape_in=(3, 3, 1))

optimizer = npnn.optimizer.Adam(model, alpha=1e-2)

optimizer.dataset = npnn_datasets.FourSmallImages()

# because of the small dataset, use all data every time for validation loss calculation ...
optimizer.dataset.validation_batch_size = optimizer.dataset.num_validation_data

################################################################################

plt.ion()
plt.show()
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

episodes = []

mini_train_loss = []
mini_validation_loss = []

train_loss = []
valid_loss = []

for episode in np.arange(200):

    # step the optimizer ...
    optimizer.step()
    episodes.append(episode)

    #===========================================================================
    # mini batch loss
    #===========================================================================

    mini_train_loss.append(np.mean(optimizer.loss))

    ax1.cla()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('train mini-batch loss')
    ax1.set_yscale('log')
    ax1.set_ylim((min(mini_train_loss)/2.0, max(mini_train_loss)*2.0))
    ax1.plot(episodes, mini_train_loss)

    #===========================================================================
    # complete dataset loss
    #===========================================================================

    optimizer.predict(optimizer.dataset.x_train_data, optimizer.dataset.y_train_data)
    tloss = optimizer.loss
    train_loss.append(np.mean(tloss))

    y_predicted_data = optimizer.predict(optimizer.dataset.x_validation_data, optimizer.dataset.y_validation_data)
    vloss = optimizer.loss
    valid_loss.append(np.mean(vloss))

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {1:5.3f}, vloss = {2:5.3f}".format(
        episode, np.mean(tloss), np.mean(vloss)
    ))

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('dataset loss')
    ax2.set_yscale('log')
    ax2.set_ylim((min(min(train_loss), min(valid_loss))/2.0, max(max(train_loss), max(valid_loss))*2.0))
    ax2.plot(episodes, train_loss, episodes, valid_loss)

    #===========================================================================

    latent_data = []
    latent_classes = []
    for n in np.arange(optimizer.dataset.x_data.shape[0]):
        optimizer.predict(np.array([optimizer.dataset.x_data[n]]))
        latent_data.append(model.layers[3].y)
        latent_classes.append(optimizer.dataset.c_data[n])

    latent_data = np.array(latent_data).reshape(-1, 2)
    latent_classes = np.array(latent_classes).reshape(-1, 1)

    ax3.cla()
    for n in np.arange(4):
        data = np.array([d for d, c in zip(latent_data, latent_classes) if c == n])
        ax3.scatter(data[:,0], data[:,1], s=4)

    #===========================================================================
    # draw and save PNG to generate video files later on
    #===========================================================================

    plt.draw()
    fig1.savefig('png/episode{0:04d}.png'.format(episode))
    plt.pause(0.001)

input("Press Enter to close ...")
