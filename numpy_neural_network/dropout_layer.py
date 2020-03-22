
if 'CUDA' in globals() or 'CUDA' in locals():
    import cupy as np
else:
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
            return np.multiply(x, self.mask)
        return x

    def backward(self, grad_y):
        '''
        gradients backward path
        '''
        if self.is_training:
            return np.multiply(grad_y, self.mask)
        return grad_y

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        super(Dropout, self).step_init(is_training)

        if is_training:
            self.mask = np.array(np.random.random(self.shape_in) > self.prob).astype(int)
        else:
            self.mask = np.ones(self.shape_in)

