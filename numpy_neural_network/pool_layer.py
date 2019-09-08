
import numpy as np

class MaxPool:
    '''max value pooling layer'''

    def __init__(self, shape_in, shape_out, kernel_size):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.kernel_size = kernel_size  # = stride

        self.w = None
        self.x = np.zeros(self.shape_in)  # layer input data
        self.y = np.zeros(self.shape_out)  # layer output data
        self.grad_x = np.zeros(self.shape_in)  # layer input gradients

        self.steps1 = self.shape_out[1]
        self.steps2 = self.shape_out[2]

        self.check()

    def forward(self, x):
        '''
        data forward path
        returns : layer output data
        '''
        assert x.shape == self.shape_in, \
            "MaxPool: forward() data shape ({0}) has ".format(x.shape) + \
            "to be equal to layer shape_in ({0}) !".format(self.shape_in)

        self.x = x.copy()
        self.y = np.full(self.shape_out, np.nan)

        for ch in np.arange(self.shape_in[0]):
            for s1 in np.arange(self.steps1):
                for s2 in np.arange(self.steps2):

                    kernel_x = self.x[
                        ch,
                        s1 * self.kernel_size : s1 * self.kernel_size + self.kernel_size,
                        s2 * self.kernel_size : s2 * self.kernel_size + self.kernel_size
                    ]

                    # set single output channel value to maximum input slice data value ...
                    self.y[ch, s1, s2] = np.amax(kernel_x)

        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        returns : layer input gradients
        '''
        assert grad_y.shape == self.shape_out, \
            "MaxPool: backward() gradient shape ({0}) has ".format(grad_y.shape) + \
            "to be equal to layer shape_out ({0}) !".format(self.shape_out)

        for ch in np.arange(self.shape_in[0]):
            for s1 in np.arange(self.steps1):
                for s2 in np.arange(self.steps2):

                    kernel_x = self.x[
                        ch,
                        s1 * self.kernel_size : s1 * self.kernel_size + self.kernel_size,
                        s2 * self.kernel_size : s2 * self.kernel_size + self.kernel_size
                    ]

                    # get 2D index of max value inside current kernel x data ...
                    idx = np.unravel_index(np.argmax(kernel_x), kernel_x.shape)

                    # set max x value kernel area position to related y gradient value ...
                    # (all other gradient values inside kernel area are kept 0)
                    self.grad_x[
                        ch,
                        s1 * self.kernel_size + idx[0],
                        s2 * self.kernel_size + idx[1]
                    ] = grad_y[ch, s1, s2]

        return self.grad_x

    def zero_grad(self):
        '''set all gradient values to zero'''
        self.grad_x = np.zeros(self.grad_x.shape)

    def check(self):
        '''check layer configuration consistency'''

        assert self.shape_in[1] % self.kernel_size == 0, \
            "MaxPool: data shape_in[1] ({0}) ".format(self.shape_in[1]) + \
            "has to be a multiple of kernel_size ({0}) !".format(self.kernel_size)
        assert self.shape_in[2] % self.kernel_size == 0, \
            "MaxPool: data shape_in[2] ({0}) ".format(self.shape_in[2]) + \
            "has to be a multiple of kernel_size ({0}) !".format(self.kernel_size)

        assert self.shape_in[1] >= self.kernel_size, \
            "MaxPool: layer shape_in[1] ({0}) has ".format(self.shape_in[1]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)
        assert self.shape_in[2] >= self.kernel_size, \
            "MaxPool: layer shape_in[2] ({0}) has ".format(self.shape_in[2]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)

        assert self.shape_out[1] == self.steps1, \
            "Conv2d: layer shape_out[1] ({0}) has ".format(self.shape_out[1]) + \
            "to be equal to layer internal steps1 ({0}) !".format(self.steps1)
        assert self.shape_out[2] == self.steps2, \
            "Conv2d: layer shape_out[2] ({0}) has ".format(self.shape_out[2]) + \
            "to be equal to layer internal steps2 ({0}) !".format(self.steps2)
