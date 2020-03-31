
import numpy as np
from numpy_neural_network import Layer

class Sequential(Layer):

    def __init__(self, shape_in=None, shape_out=None):
        super(Sequential, self).__init__(shape_in, shape_out, None)
        self.layers = []
        self.chain = None
        self.loss = np.array([0.0])
        self.accuracy = 0.0

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, g):
        for layer in self.layers[::-1]:
            g = layer.backward(g)
        return g

    def step(self, x=None, t=None):
        x = self.forward(x)
        g, y = self.chain.step(x=x, t=t)
        g = self.backward(g)

        self.loss     = self.chain.loss
        self.accuracy = self.chain.accuracy
        return g, y

    def predict(self, x, t=None):
        for layer in self.layers:
            x = layer.forward(x)
        x = self.chain.predict(x, t)

        self.loss     = self.chain.loss
        self.accuracy = self.chain.accuracy
        return x

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
        self.chain.zero_grad()

    def step_init(self, is_training):
        for layer in self.layers:
            layer.step_init(is_training=is_training)
        self.chain.step_init(is_training=is_training)

    def update_weights(self, callback):
        for layer in self.layers:
            layer.update_weights(callback)
        self.chain.update_weights(callback)

