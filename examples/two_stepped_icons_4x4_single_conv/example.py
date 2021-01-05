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

model = npnn.Sequential()
model.layers = [
    npnn.Conv2D(shape_in=(4, 4, 1), shape_out=(2, 2, 2), kernel_size=3, stride=1),
    npnn.LeakyReLU(2 * 2 * 2),

    npnn.AvgPool(shape_in=(2, 2, 2), shape_out=(1, 1, 2), kernel_size=2),
    npnn.Shape(shape_in=(1, 1, 2), shape_out=(2)),
    npnn.Softmax(2)
]

loss_layer = npnn.loss_layer.CrossEntropyLoss(2)
optimizer  = npnn.optimizer.Adam(alpha=1e-2)
dataset    = npnn_datasets.TwoSteppedIcons()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer

# because of the small dataset, use all data every time for validation loss calculation ...
dataset.validation_batch_size = dataset.num_validation_data

################################################################################

plt.ion()
plt.show()
fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10,12))

episodes = []

mini_train_loss = []
mini_validation_loss = []
mini_train_accuracy = []
mini_validation_accuracy = []

train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []

for episode in np.arange(250):

    # step the optimizer ...
    optimizer.step(*dataset.get_train_batch())
    episodes.append(episode)

    #===========================================================================
    # mini batch loss and accuracy
    #===========================================================================

    mini_train_loss.append(loss_layer.get_loss())
    mini_train_accuracy.append(loss_layer.get_accuracy())

    ax1.cla()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('train mini-batch loss')
    ax1.set_yscale('log')
    ax1.set_ylim((min(mini_train_loss)/2.0, max(mini_train_loss)*2.0))
    ax1.plot(episodes, mini_train_loss)

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('train mini-batch accuracy')
    ax2.set_ylim(-0.05, 1.05)
    ax2.plot(episodes, mini_train_accuracy)

    #===========================================================================
    # complete dataset loss and accuracy
    #===========================================================================

    y_predicted_data = optimizer.predict(dataset.x_validation_data, dataset.y_validation_data)
    vloss = loss_layer.get_loss()
    vaccuracy = loss_layer.get_accuracy()
    valid_loss.append(vloss)
    valid_accuracy.append(vaccuracy)

    optimizer.predict(dataset.x_train_data, dataset.y_train_data)
    tloss = loss_layer.get_loss()
    taccuracy = loss_layer.get_accuracy()
    train_loss.append(tloss)
    train_accuracy.append(taccuracy)

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {1:5.3f}, vloss = {2:5.3f}, taccuracy = {3:5.3f}, vaccuracy = {4:5.3f}".format(
        episode, np.mean(tloss), np.mean(vloss), taccuracy, vaccuracy
    ))

    ax3.cla()
    ax3.set_xlabel('episode')
    ax3.set_ylabel('dataset loss')
    ax3.set_yscale('log')
    ax3.set_ylim((min(min(train_loss), min(valid_loss))/2.0, max(max(train_loss), max(valid_loss))*2.0))
    ax3.plot(episodes, train_loss, episodes, valid_loss)

    ax4.cla()
    ax4.set_xlabel('episode')
    ax4.set_ylabel('dataset accuracy')
    ax4.set_ylim(-0.05, 1.05)
    ax4.plot(episodes, train_accuracy, episodes, valid_accuracy)

    #===========================================================================
    # batch network output plots
    #===========================================================================

    k = np.arange(dataset.num_validation_data)

    ax5.cla()
    ax5.set_ylabel('mini-batch class 0')
    ax5.set_ylim(-0.1, 1.1)
    ax5.scatter(k, dataset.y_validation_data[:,0], s=10, c='tab:green')
    ax5.scatter(k, y_predicted_data[:,0], s=10, c='tab:orange')

    ax6.cla()
    ax6.set_ylabel('mini-batch class 1')
    ax6.set_ylim(-0.1, 1.1)
    ax6.scatter(k, dataset.y_validation_data[:,1], s=10, c='tab:green')
    ax6.scatter(k, y_predicted_data[:,1], s=10, c='tab:orange')

    #===========================================================================
    # draw and save PNG to generate video files later on
    #===========================================================================

    plt.draw()
    fig1.savefig('png/episode{0:04d}.png'.format(episode))
    plt.pause(0.001)

input("Press Enter to close ...")

