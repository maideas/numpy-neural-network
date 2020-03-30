
import numpy as np
import numpy_neural_network as npnn
from numpy_neural_network import Layer

class Latent(Layer):
    '''variational autoencoder (VAE) latent layer'''

    def __init__(self, shape_in):

        shape_out = (shape_in[0], shape_in[1], 2 * shape_in[2])

        super(Latent, self).__init__(shape_in, shape_out, None)

        self.latent_mean_model = npnn.network.Model([
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),
            npnn.Tanh(shape_in=shape_in),
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),

            # Linear output can be negative and positive without limits
            # like a mean value ...
            npnn.Linear(shape_in=shape_in)
        ])

        self.latent_variance_model = npnn.network.Model([
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),
            npnn.Tanh(shape_in=shape_in),
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),

            # Softplus output is always positive like a variance value ...
            npnn.Softplus(shape_in=shape_in)
        ])

        self.x_mean = np.zeros(shape_in)
        self.x_variance = np.zeros(shape_in)

        self.kl_loss = 0.0

    def update_kl_loss(self):

        kl_loss = 0.5 * (np.square(self.x_mean) + self.x_variance - 1.0 - np.log(self.x_variance))
        self.kl_loss += np.mean(kl_loss)

    def forward(self, x):

        self.x_mean = x.copy()
        for layer in self.latent_mean_model.layers:
            self.x_mean = layer.forward(self.x_mean)

        self.x_variance = x.copy()
        for layer in self.latent_variance_model.layers:
            self.x_variance = layer.forward(self.x_variance)

        self.y[:,:,:self.shape_in[2]] = self.x_variance
        self.y[:,:,self.shape_in[2]:] = self.x_mean

        self.update_kl_loss()
        return self.y

    def backward(self, grad_y):

        grad_y_variance = grad_y[:,:,:self.shape_in[2]]
        grad_y_mean     = grad_y[:,:,self.shape_in[2]:]

        self.grad_x = grad_y_mean + grad_y_variance
        return self.grad_x

    def zero_grad(self):
        '''
        set all gradient values to zero
        '''
        for layer in self.latent_mean_model.layers:
            layer.zero_grad()
        for layer in self.latent_variance_model.layers:
            layer.zero_grad()

    def init_w(self):
        '''
        weight initialization
        '''
        for layer in self.latent_mean_model.layers:
            layer.init_w()
        for layer in self.latent_variance_model.layers:
            layer.init_w()

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        self.is_training = is_training

        for layer in self.latent_mean_model.layers:
            layer.step_init(is_training=is_training)
        for layer in self.latent_variance_model.layers:
            layer.step_init(is_training=is_training)

        self.kl_loss = 0.0

    def update_weights(self, callback):
        '''
        weight update
        '''
        for layer in self.latent_mean_model.layers:
            layer.update_weights(callback=callback)
        for layer in self.latent_variance_model.layers:
            layer.update_weights(callback=callback)


