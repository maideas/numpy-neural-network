
import numpy as np

class MaxPool:
    '''max value pooling layer'''

    def __init__(self, shape_in, shape_out, kernel_size, stride=None):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.kernel_size = kernel_size
        self.stride = stride
        if self.stride is None:
            self.stride = kernel_size

        self.w = None
        self.x = np.zeros(self.shape_in)  # layer input data
        self.y = np.zeros(self.shape_out)  # layer output data
        self.grad_x = np.zeros(self.shape_in)  # layer input gradients

        self.steps_h = 1 + int(np.trunc((self.shape_in[0] - self.kernel_size) / self.stride))
        self.steps_w = 1 + int(np.trunc((self.shape_in[1] - self.kernel_size) / self.stride))

        self.check()

        self.x_indices = []
        self.y_indices = []
        for ch in np.arange(self.shape_in[2]):
            for sh in np.arange(self.steps_h):
                for sw in np.arange(self.steps_w):
                    self.x_indices.append((
                        slice(sh * self.stride, sh * self.stride + self.kernel_size),
                        slice(sw * self.stride, sw * self.stride + self.kernel_size),
                        ch
                    ))
                    self.y_indices.append((
                        sh,
                        sw,
                        ch
                    ))

    def forward(self, x):
        '''
        data forward path
        returns : layer output data
        '''
        assert x.shape == self.shape_in, \
            "MaxPool: forward() data shape ({0}) has ".format(x.shape) + \
            "to be equal to layer shape_in ({0}) !".format(self.shape_in)

        self.x = x.copy()
        self.y = np.zeros(self.shape_out)

        for x_index, y_index in zip(self.x_indices, self.y_indices):

            kernel_x = self.x[x_index]

            # set single output channel value to maximum input slice data value ...
            self.y[y_index] = np.amax(kernel_x)

        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        returns : layer input gradients
        '''
        assert grad_y.shape == self.shape_out, \
            "MaxPool: backward() gradient shape ({0}) has ".format(grad_y.shape) + \
            "to be equal to layer shape_out ({0}) !".format(self.shape_out)

        for x_index, y_index in zip(self.x_indices, self.y_indices):

            kernel_x = self.x[x_index]

            # get 2D index of max value inside current kernel x data ...
            idx = np.unravel_index(np.argmax(kernel_x), kernel_x.shape)

            # set max x value kernel area position to related y gradient value ...
            # (all other gradient values inside kernel area are kept 0)
            self.grad_x[x_index][idx] += grad_y[y_index]

        return self.grad_x

    def zero_grad(self):
        '''set all gradient values to zero'''
        self.grad_x = np.zeros(self.grad_x.shape)

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        pass

    def check(self):
        '''check layer configuration consistency'''

        assert (self.shape_in[0] - self.kernel_size) % self.stride == 0, \
            "MaxPool: layer shape_in[0] ({0}) ".format(self.shape_in[0]) + \
            "minus kernel_size ({0}) has ".format(self.kernel_size) + \
            "to be a multiple of stride ({0}) !".format(self.stride)
        assert (self.shape_in[1] - self.kernel_size) % self.stride == 0, \
            "MaxPool: layer shape_in[1] ({0}) ".format(self.shape_in[1]) + \
            "minus kernel_size ({0}) has ".format(self.kernel_size) + \
            "to be a multiple of stride ({0}) !".format(self.stride)

        assert self.shape_in[0] >= self.kernel_size, \
            "MaxPool: layer shape_in[0] ({0}) has ".format(self.shape_in[0]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)
        assert self.shape_in[1] >= self.kernel_size, \
            "MaxPool: layer shape_in[1] ({0}) has ".format(self.shape_in[1]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)

        assert self.shape_out[0] == self.steps_h, \
            "MaxPool: layer shape_out[0] ({0}) has ".format(self.shape_out[0]) + \
            "to be equal to layer internal steps_h ({0}) !".format(self.steps_h)
        assert self.shape_out[1] == self.steps_w, \
            "MaxPool: layer shape_out[1] ({0}) has ".format(self.shape_out[1]) + \
            "to be equal to layer internal steps_w ({0}) !".format(self.steps_w)

        assert self.shape_out[2] == self.shape_in[2], \
            "MaxPool: data shape_out[2] ({0}) ".format(self.shape_out[2]) + \
            "has to be the same as shape_in[2] ({0}) !".format(self.shape_in[2])

