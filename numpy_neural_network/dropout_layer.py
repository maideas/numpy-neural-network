
import numpy as np
from numpy_neural_network import Layer

class Dropout(Layer):
    '''dropout layer'''

    def __init__(self, shape_in, prob=0.5):
        super(Dropout, self).__init__(shape_in, shape_in, None)

        self.prob = prob
        self.mask = np.ones(self.shape_in)

    def forward(self, x):
        '''
        data forward path
        '''
        if self.is_training:
            self.mask = np.array(np.random.random(self.shape_in) > self.prob).astype(int)
            return np.multiply(x, self.mask) / (1e-3 + 1.0 - self.prob)

        return x

    def backward(self, grad_y):
        '''
        gradients backward path
        '''
        return grad_y

