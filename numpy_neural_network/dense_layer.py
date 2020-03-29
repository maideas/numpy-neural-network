
import numpy as np
from numpy_neural_network import Layer

class Dense(Layer):
    '''dense (fully) connected layer'''

    def __init__(self, shape_in, shape_out):
        shape_w = (np.prod(shape_out), np.prod(shape_in))

        super(Dense, self).__init__(shape_in, shape_out, shape_w)

        self.x = np.zeros(np.prod(shape_in))
        self.init_w()

    def forward(self, x):
        self.x = x.ravel()
        self.y = np.matmul(self.w, self.x).reshape(self.shape_out)
        self.y += self.wb

        return self.y

    def backward(self, grad_y):
        self.grad_w += np.outer(grad_y, self.x)
        self.grad_wb += grad_y
        self.grad_x = np.matmul(grad_y, self.w).reshape(self.shape_in)

        return self.grad_x

    def init_w(self):
        '''
        weight initialization (Xavier Glorot et al.) ...
        mean = 0
        variance = sqrt(6) / (num neurons in previous layer + num neurons in this layer)
        bias weights = 0
        '''
        stddev = np.sqrt(2.45 / (np.prod(self.shape_in) + np.prod(self.shape_out)))
        self.w = np.random.normal(0.0, stddev, self.shape_w)
        self.wb = np.zeros(self.shape_out)  # ... set the bias weights to 0

