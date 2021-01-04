
import numpy as np
from numpy_neural_network import Layer

class Pad2D(Layer):
    '''2D padding layer'''

    def __init__(self, shape_in, pad_axis0=0, pad_axis1=0, pad_value=0):
        super(Pad2D, self).__init__(shape_in, None, None)
        
        self.pad_axis0 = pad_axis0
        self.pad_axis1 = pad_axis1

        self.y = np.full((
            self.shape_in[0] + 2 * pad_axis0,
            self.shape_in[1] + 2 * pad_axis1,
            self.shape_in[2]
        ), pad_value, dtype='float64')

    def forward(self, x):
        '''
        data forward path
        '''
        self.y[
            self.pad_axis0:self.pad_axis0 + self.shape_in[0],
            self.pad_axis1:self.pad_axis1 + self.shape_in[1],
            :
        ] = x
        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        '''
        return grad_y[
            self.pad_axis0:self.pad_axis0 + self.shape_in[0],
            self.pad_axis1:self.pad_axis1 + self.shape_in[1],
            :
        ]

