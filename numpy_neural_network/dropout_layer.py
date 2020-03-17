
import numpy as np

class Dropout:
    '''dropout layer'''

    def __init__(self, shape_in, prob=0.5):
        self.shape_in = shape_in
        self.prob = prob
        self.mask = np.ones(self.shape_in)
        self.is_training = False
        self.w = None

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

    def zero_grad(self):
        pass

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        if is_training:
            self.is_training = True
            self.mask = np.array(np.random.random(self.shape_in) > self.prob).astype(int)

        else:
            self.is_training = False
            self.mask = np.ones(self.shape_in)

