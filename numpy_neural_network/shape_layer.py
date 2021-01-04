
import numpy as np
from numpy_neural_network import Layer

class Shape(Layer):
    '''tensor reshape layer'''

    def __init__(self, shape_in, shape_out):
        super(Shape, self).__init__(shape_in, shape_out, None)

    def forward(self, x):
        self.y = x.ravel().reshape(self.shape_out)
        return self.y

    def backward(self, grad_y):
        self.grad_x = grad_y.ravel().reshape(self.shape_in)
        return self.grad_x

