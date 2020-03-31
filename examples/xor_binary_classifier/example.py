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
    npnn.Dense(2, 4),
    npnn.LeakyReLU(4),
    npnn.Dense(4, 4),
    npnn.LeakyReLU(4),
    npnn.Dense(4, 1),
    npnn.Sigmoid(1)
]

loss_layer = npnn.loss_layer.BinaryCrossEntropyLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=5e-3)
dataset    = npnn_datasets.XORBinaryClassifier()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer

# because of the small dataset, use all data every time for validation loss calculation ...
dataset.validation_batch_size = dataset.num_validation_data

################################################################################

plt.ion()
plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1)

episodes = []

train_loss_y = []
validation_loss_y = []
train_accuracy_y = []
validation_accuracy_y = []

for episode in np.arange(200):

    # step the optimizer ...
    optimizer.step(*dataset.get_train_batch())
    episodes.append(episode)

    # append the optimizer step train loss ...
    tloss = np.mean(optimizer.loss)
    taccuracy = optimizer.accuracy
    train_loss_y.append(tloss)
    train_accuracy_y.append(taccuracy)

    # calculate and append the validation loss ...
    x_validation_batch, t_validation_batch = dataset.get_validation_batch()
    y_validation_batch = optimizer.predict(x_validation_batch, t_validation_batch)

    vloss = np.mean(optimizer.loss)
    vaccuracy = optimizer.accuracy

    validation_loss_y.append(vloss)
    validation_accuracy_y.append(vaccuracy)

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {2:5.3f}, vloss = {2:5.3f}".format(episode, tloss, vloss))

    # print the train loss (blue) and validation loss (orange) ...
    ax1.cla()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('loss')
    ax1.set_yscale('log')
    ax1.set_ylim((min(train_loss_y)/2.0, max(train_loss_y)*2.0))
    ax1.plot(episodes, train_loss_y, episodes, validation_loss_y)

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('accuracy')
    ax2.set_ylim(-0.05, 1.05)
    ax2.plot(episodes, train_accuracy_y, episodes, validation_accuracy_y)

    plt.draw()
    plt.savefig('png/episode{0:04d}.png'.format(episode))  # save png to create mp4 later on
    plt.pause(0.001)

input("Press Enter to close ...")

