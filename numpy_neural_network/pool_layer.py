
import numpy as np

class MaxPool:
    '''max value pooling layer'''

    def __init__(self, channels, kernel_size):
        self.channels = channels
        self.kernel_size = kernel_size  # = stride

        self.w = None
        self.x = None  # pool layer input data of depth self.channels
        self.y = None  # pool layer output data of depth self.channels
        self.grad_x = None  # layer input gradients
        self.steps1 = 0
        self.steps2 = 0

    def forward(self, x):
        '''
        data forward path
        returns : layer output data
        '''
        assert x.shape[0] == self.channels, \
            "MaxPool: forward() data shape[0] ({0}) has ".format(x.shape[0]) + \
            "to be equal to layer channels ({0}) !".format(self.channels)

        assert x.shape[1] >= self.kernel_size, \
            "MaxPool: forward() data shape[1] ({0}) has ".format(x.shape[1]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)
        assert x.shape[2] >= self.kernel_size, \
            "MaxPool: forward() data shape[2] ({0}) has ".format(x.shape[2]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)

        assert x.shape[1] % self.kernel_size == 0, \
            "MaxPool: forward() data shape[1] ({0}) ".format(x.shape[1]) + \
            "has to be a multiple of kernel_size ({0}) !".format(self.kernel_size)
        assert x.shape[2] % self.kernel_size == 0, \
            "MaxPool: forward() data shape[2] ({0}) ".format(x.shape[2]) + \
            "has to be a multiple of kernel_size ({0}) !".format(self.kernel_size)

        self.steps1 = int(np.trunc(x.shape[1] / self.kernel_size))
        self.steps2 = int(np.trunc(x.shape[2] / self.kernel_size))

        self.x = x.copy()
        self.y = np.full((self.channels, self.steps1, self.steps2), np.nan)

        for ch in np.arange(self.channels):
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
        assert grad_y.shape[0] == self.channels, \
            "MaxPool: backward() gradient shape[0] ({0}) has ".format(grad_y.shape[0]) + \
            "to be equal to layer channels ({0}) !".format(self.channels)
        assert grad_y.shape[1] == self.steps1, \
            "MaxPool: backward() gradient shape[1] ({0}) has ".format(grad_y.shape[1]) + \
            "to be equal to layer internal steps1 ({0}) value !".format(self.steps1)
        assert grad_y.shape[2] == self.steps2, \
            "MaxPool: backward() gradient shape[2] ({0}) has ".format(grad_y.shape[2]) + \
            "to be equal to layer internal steps2 ({0}) value !".format(self.steps2)

        for ch in np.arange(self.channels):
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
        self.grad_x = np.zeros(self.x.shape)

