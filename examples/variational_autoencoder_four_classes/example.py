#!/usr/bin/env python

import sys
sys.path.append("../../")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy_neural_network as npnn
import npnn_datasets
from numpy_neural_network import Sequential

matplotlib.rcParams['toolbar'] = 'None'

################################################################################

encoder_model = npnn.Sequential()
encoder_model.layers = [
    npnn.Conv2D(shape_in=(3, 3, 1), shape_out=(2, 2, 6), kernel_size=2, stride=1),
    npnn.Tanh(2 * 2 * 6),
    npnn.Conv2D(shape_in=(2, 2, 6), shape_out=(1, 1, 2), kernel_size=2, stride=1),
    npnn.Tanh(1 * 1 * 2),
    npnn.Latent(shape_in=(1, 1, 2))
]


decoder_steps_per_encoder_step = 25

class DecoderSequential(Sequential):

    def step(self, x=None, t=None):  # override of Sequential step method
        global decoder_steps_per_encoder_step

        g_dec = np.zeros(x.shape)

        for _ in np.arange(decoder_steps_per_encoder_step):
            x_dec = x.copy()
            t_dec = t.copy()

            x_dec = self.forward(x_dec)
            g, y_dec = self.chain.step(x=x_dec, t=t_dec)
            g_dec += self.backward(g)

        g_dec         = g_dec               / decoder_steps_per_encoder_step
        self.loss     = self.chain.loss     / decoder_steps_per_encoder_step
        self.accuracy = self.chain.accuracy / decoder_steps_per_encoder_step
        return g_dec, y_dec


decoder_model = DecoderSequential()
decoder_model.layers = [
    npnn.Sample(shape_out=(1, 1, 2)),
    npnn.UpConv2D(shape_in=(1, 1, 2), shape_out=(2, 2, 6), kernel_size=2, stride=1),
    npnn.Tanh(2 * 2 * 6),
    npnn.UpConv2D(shape_in=(2, 2, 6), shape_out=(3, 3, 1), kernel_size=2, stride=1),
    npnn.Tanh(3 * 3 * 1)
]

latent_layer = encoder_model.layers[4]
sample_layer = decoder_model.layers[0]
sample_layer.train_z = []

################################################################################

loss_layer = npnn.loss_layer.RMSLoss(shape_in=(3, 3, 1))
optimizer  = npnn.optimizer.Adam(alpha=5e-3)
dataset    = npnn_datasets.FourSmallImages()

encoder_model.chain = decoder_model
decoder_model.chain = loss_layer

optimizer.norm  = dataset.norm
optimizer.model = encoder_model

# because of the small dataset, use all data every time for validation loss calculation ...
dataset.validation_batch_size = dataset.num_validation_data

dataset.train_batch_size = int(dataset.num_train_data / 2)

################################################################################

plt.ion()
plt.show()
fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))

plt.style.use('seaborn-whitegrid')

plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.3)

episodes = []

mini_train_loss = []
mini_validation_loss = []

train_loss = []
valid_loss = []
mini_train_kl_loss = []
mini_validation_kl_loss = []

train_batch_z0_mean = []
train_batch_z1_mean = []

for episode in np.arange(2000):

    # step the optimizer ...
    optimizer.step(*dataset.get_train_batch())
    episodes.append(episode)

    train_z = np.array(sample_layer.train_z).reshape(-1, 2)

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

    optimizer.predict(dataset.x_train_data, dataset.y_train_data)
    tloss = optimizer.loss
    train_loss.append(np.mean(tloss))
    mini_train_kl_loss.append(latent_layer.kl_loss / dataset.num_train_data)

    y_predicted_data = optimizer.predict(dataset.x_validation_data, dataset.y_validation_data)
    vloss = optimizer.loss
    valid_loss.append(np.mean(vloss))

    # print the episode and loss values ...
    print("episode = {0:5d}, tloss = {1:5.3f}, vloss = {2:5.3f}".format(
        episode, np.mean(tloss), np.mean(vloss)
    ))

    ax2.cla()
    ax2.set_xlabel('episode')
    ax2.set_ylabel('dataset reconstruction loss')
    ax2.set_yscale('log')
    ax2.set_ylim((min(min(train_loss), min(valid_loss))/2.0, max(max(train_loss), max(valid_loss))*2.0))
    ax2.plot(episodes, train_loss, episodes, valid_loss)

    #===========================================================================

    latent_mean = []
    latent_variance = []
    latent_data = []
    latent_classes = []

    for n in np.arange(dataset.x_data.shape[0]):
        optimizer.predict(np.array([dataset.x_data[n]]))

        latent_mean.append(latent_layer.x_mean)
        latent_variance.append(latent_layer.x_variance)
        latent_data.append(sample_layer.y)
        latent_classes.append(dataset.c_data[n])

    latent_mean = np.array(latent_mean).reshape(-1, 2)
    latent_variance = np.array(latent_variance).reshape(-1, 2)
    latent_data = np.array(latent_data).reshape(-1, 2)
    latent_classes = np.array(latent_classes).reshape(-1, 1)

    #===========================================================================

    optimizer.predict(dataset.x_validation_data)
    mini_validation_kl_loss.append(latent_layer.kl_loss / dataset.num_validation_data)

    #===========================================================================

    ax3.cla()
    ax3.set_xlabel('episode')
    ax3.set_ylabel('KL divergence loss')
    ax3.set_yscale('log')
    ax3.set_ylim(
        min(min(mini_train_kl_loss), min(mini_validation_kl_loss)) / 2.0,
        max(max(mini_train_kl_loss), max(mini_validation_kl_loss)) * 2.0
    )
    ax3.plot(episodes, mini_train_kl_loss, episodes, mini_validation_kl_loss)

    #===========================================================================

    ax4.cla()
    ax4.set_xlabel('2D latent space (training)')
    ax4.set_xlim(min(-5.0, 1.1*min(train_z[:,0])), max(5.0, 1.1*max(train_z[:,0])))
    ax4.set_ylim(min(-5.0, 1.1*min(train_z[:,1])), max(5.0, 1.1*max(train_z[:,1])))
    ax4.scatter(train_z[:,0], train_z[:,1], s=4)

    #===========================================================================

    ax5.cla()
    ax5.set_xlabel('2D latent space variance')
    ax5.set_xlim(min(0.0, 1.1*min(latent_variance[:,0])), max(2.0, 1.1*max(latent_variance[:,0])))
    ax5.set_ylim(min(0.0, 1.1*min(latent_variance[:,1])), max(2.0, 1.1*max(latent_variance[:,1])))

    for n in np.arange(4):
        x_variance = np.array([d for d, c in zip(latent_variance, latent_classes) if c == n])
        ax5.scatter(x_variance[:,0], x_variance[:,1], s=4)

    #===========================================================================

    ax6.cla()
    ax6.set_xlabel('2D latent space mean')
    ax6.set_xlim(min(-1.0, 1.1*min(latent_mean[:,0])), max(1.0, 1.1*max(latent_mean[:,0])))
    ax6.set_ylim(min(-1.0, 1.1*min(latent_mean[:,1])), max(1.0, 1.1*max(latent_mean[:,1])))

    for n in np.arange(4):
        x_mean = np.array([d for d, c in zip(latent_mean, latent_classes) if c == n])
        ax6.scatter(x_mean[:,0], x_mean[:,1], s=4)

    #===========================================================================
    # draw and save PNG to generate video files later on
    #===========================================================================

    plt.draw()
    fig1.savefig('png/episode{0:04d}.png'.format(episode))
    plt.pause(0.001)

input("Press Enter to close ...")

