
import numpy as np
import numpy_neural_network as npnn
from numpy_neural_network import Layer

class Latent(Layer):
    '''variational autoencoder (VAE) latent layer'''

    def __init__(self, shape_in):

        shape_out = (shape_in[0], shape_in[1], 2 * shape_in[2])

        super(Latent, self).__init__(shape_in, shape_out, None)

        self.latent_mean_model = npnn.Sequential()
        self.latent_mean_model.layers = [
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),
            npnn.Tanh(shape_in=shape_in),
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),

            # Linear output can be negative and positive without limits
            # like a mean value ...
            npnn.Linear(shape_in=shape_in)
        ]

        self.latent_variance_model = npnn.Sequential()
        self.latent_variance_model.layers = [
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),
            npnn.Tanh(shape_in=shape_in),
            npnn.Dense(shape_in=shape_in, shape_out=shape_in),

            # Softplus output is always positive like a variance value ...
            npnn.Softplus(shape_in=shape_in)
        ]

        self.x_mean = np.zeros(shape_in)
        self.x_variance = np.zeros(shape_in)


    def forward(self, x):

        self.x_variance = self.latent_variance_model.forward(x)
        self.x_mean     = self.latent_mean_model.forward(x)

        self.y[:,:,:self.shape_in[2]] = self.x_variance
        self.y[:,:,self.shape_in[2]:] = self.x_mean

        return self.y

    def backward(self, grad_y):

        grad_y_variance = grad_y[:,:,:self.shape_in[2]]
        grad_y_mean     = grad_y[:,:,self.shape_in[2]:]

        grad_y_variance = self.latent_variance_model.backward(grad_y_variance)
        grad_y_mean     = self.latent_mean_model.backward(grad_y_mean)

        self.grad_x = grad_y_mean + grad_y_variance
        return self.grad_x

    def zero_grad(self):
        '''
        set all gradient values to zero
        '''
        self.latent_mean_model.zero_grad()
        self.latent_variance_model.zero_grad()

    def init_w(self):
        '''
        weight initialization
        '''
        self.latent_mean_model.init_w()
        self.latent_variance_model.init_w()

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        self.is_training = is_training
        self.latent_mean_model.step_init(is_training=is_training)
        self.latent_variance_model.step_init(is_training=is_training)

    def update_weights(self, callback):
        '''
        weight update
        '''
        self.latent_mean_model.update_weights(callback=callback)
        self.latent_variance_model.update_weights(callback=callback)


