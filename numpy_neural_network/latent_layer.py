
import numpy as np
import numpy_neural_network as npnn
from numpy_neural_network import Layer

class Latent(Layer):
    '''variational autoencoder (VAE) latent layer'''

    def __init__(self, shape_in):

        super(Latent, self).__init__(shape_in, shape_in, None)

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
        self.train_z = None  # training mode hidden state z data mini batch

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

        if self.is_training:
            # the mean of gausian sampled variance mean(x_mean + N(0, I)) is 0, which
            # means that x_mean represents the mean vector of the distribution
            self.y = self.x_mean + self.x_variance * np.random.normal(0.0, 1.0, self.x_variance.shape)

            if self.train_z is not None:
                self.train_z.append(self.y)  # collect hidden state z data for evaluation
        else:
            # non-training mode z = hidden state mean (its the expected value) ...
            self.y = self.x_mean

        self.update_kl_loss()
        return self.y

    def backward(self, grad_y):

        # KL mean gradient = derivative of: 0.5*(x_mean^2) ...
        kl_mean_grad = self.x_mean

        # KL variance gradient = derivative of: 0.5*(x_variance - log(variance) - 1) ...
        kl_variance_grad = 0.5 - np.divide(0.5, self.x_variance + 1e-9)

        # combine the gradients from the decoder with the KL gradients ...
        grad_y_mean     = 0.5 * grad_y + kl_mean_grad
        grad_y_variance = 0.5 * grad_y + kl_variance_grad

        for layer in self.latent_mean_model.layers[::-1]:
            grad_y_mean = layer.backward(grad_y_mean)

        for layer in self.latent_variance_model.layers[::-1]:
            grad_y_variance = layer.backward(grad_y_variance)

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
        if is_training and self.train_z is not None:
            self.train_z = []

    def update_weights(self, callback):
        '''
        weight update
        '''
        for layer in self.latent_mean_model.layers:
            layer.update_weights(callback=callback)
        for layer in self.latent_variance_model.layers:
            layer.update_weights(callback=callback)


