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
    npnn.Dense(2, 4),
    npnn.LeakyReLU(4),
    npnn.Dense(4, 4),
    npnn.LeakyReLU(4),
    npnn.Dense(4, 1),
    npnn.Sigmoid(1)
])

model.loss_layer = npnn.loss_layer.RMSLoss(1)

optimizer = npnn.optimizer.Adam(model, alpha=2e-2)

optimizer.dataset = npnn_datasets.XORFunction()

# because of the small dataset, use all data every time for validation loss calculation ...
optimizer.dataset.validation_batch_size = optimizer.dataset.num_validation_data

################################################################################

plt.ion()
plt.show()
plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

mesh_x, mesh_y = np.meshgrid(np.arange(-0.3, 1.4, 0.05), np.arange(-0.3, 1.4, 0.1))
linear_x = mesh_x.reshape((-1,1))
linear_y = mesh_y.reshape((-1,1))
linear_xy = np.hstack((linear_x, linear_y))

episodes = []
train_loss_y = []
validation_loss_y = []

for episode in np.arange(300):

    linear_z = optimizer.predict(linear_xy)
    mesh_z = linear_z.reshape(mesh_x.shape)
    ax1.cla()
    ax1.pcolormesh(mesh_x, mesh_y, mesh_z, cmap='coolwarm');
    
    ax1.scatter([0, 1], [0, 1], s=30, c='tab:blue')
    ax1.scatter([0, 1], [1, 0], s=30, c='tab:red')

    # step the optimizer ...
    optimizer.step()
    episodes.append(episode)

    # append the optimizer step train loss ...
    tloss = np.mean(optimizer.loss)
    train_loss_y.append(tloss)

    # calculate and append the validation loss ...
    x_validation_batch, t_validation_batch = optimizer.dataset.get_validation_batch()
    y_validation_batch = optimizer.predict(x_validation_batch, t_validation_batch)

    vloss = np.mean(optimizer.loss)
    validation_loss_y.append(vloss)

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {2:5.3f}, vloss = {2:5.3f}".format(episode, tloss, vloss))

    # print the train loss (blue) and validation loss (orange) ...
    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('loss')
    ax2.set_yscale('log')
    ax2.set_ylim((min(train_loss_y)/2.0, max(train_loss_y)*2.0))
    ax2.plot(episodes, train_loss_y, episodes, validation_loss_y)

    plt.draw()
    plt.savefig('png/episode{0:04d}.png'.format(episode))  # save png to create mp4 later on
    plt.pause(0.001)

input("Press Enter to close ...")

