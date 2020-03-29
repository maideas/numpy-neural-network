
import numpy as np
from numpy_neural_network import Layer

class Conv2D(Layer):
    '''2D convolutional layer'''

    def __init__(self, shape_in, shape_out, kernel_size, stride=1, groups=1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

        self.channels_in_per_group = int(np.trunc(shape_in[2] / self.groups))
        self.channels_out_per_group = int(np.trunc(shape_out[2] / self.groups))

        shape_w = (
            self.groups, self.channels_out_per_group,
            self.kernel_size, self.kernel_size, self.channels_in_per_group
        )

        super(Conv2D, self).__init__(shape_in, shape_out, shape_w)

        self.steps_h = 1 + int(np.trunc((shape_in[0] - self.kernel_size) / self.stride))
        self.steps_w = 1 + int(np.trunc((shape_in[1] - self.kernel_size) / self.stride))

        self.check()

        self.x = np.zeros(self.shape_in)

        self.init_w()

        self.x_indices = []
        self.y_indices = []
        self.w_indices = []
        for group in np.arange(self.groups):
            for sh in np.arange(self.steps_h):
                for sw in np.arange(self.steps_w):
                    self.x_indices.append((
                        slice(sh * self.stride, sh * self.stride + self.kernel_size),
                        slice(sw * self.stride, sw * self.stride + self.kernel_size),
                        slice(group * self.channels_in_per_group, (group + 1) * self.channels_in_per_group)
                    ))
                    self.y_indices.append((
                        sh,
                        sw,
                        slice(group * self.channels_out_per_group, (group + 1) * self.channels_out_per_group)
                    ))
                    self.w_indices.append((
                        group
                    ))

    def forward(self, x):
        '''
        data forward path
        input data -> weighted sums -> output data
        returns : layer output data
        '''
        assert x.shape == self.shape_in, \
            "Conv2D: forward() data shape ({0}) has ".format(x.shape) + \
            "to be equal to layer shape_in ({0}) !".format(self.shape_in)

        self.x = x
        self.y = np.zeros(self.shape_out)

        for x_index, y_index, w_index in zip(self.x_indices, self.y_indices, self.w_indices):
            for kernel_sel in np.arange(self.channels_out_per_group):
                self.y[y_index][kernel_sel] += np.sum(np.multiply(self.w[w_index][kernel_sel], self.x[x_index]))

        self.y += self.wb
        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        output gradients (grad_y) -> derivative w.r.t weights -> weight gradients (grad_w)
        output gradients (grad_y) -> derivative w.r.t inputs -> input gradients (grad_x)
        returns : layer input gradients
        '''
        assert grad_y.shape == self.shape_out, \
            "Conv2D: backward() gradient shape ({0}) has ".format(grad_y.shape) + \
            "to be equal to layer shape_out ({0}) !".format(self.shape_out)

        self.grad_x = np.zeros(self.shape_in)
        for x_index, y_index, w_index in zip(self.x_indices, self.y_indices, self.w_indices):
            for kernel_sel in np.arange(self.channels_out_per_group):
                self.grad_x[x_index] += np.multiply(grad_y[y_index][kernel_sel], self.w[w_index][kernel_sel])
                self.grad_w[w_index][kernel_sel] += grad_y[y_index][kernel_sel] * self.x[x_index]

        self.grad_wb += grad_y
        return self.grad_x

    def init_w(self):
        '''
        mean = 0
        variance = 2.0 / ((kernel size / stride)^2 * channels in + (kernel size)^2 * channels out)
        bias weights = 0
        '''
        stddev = np.sqrt(2.0 / (
            np.square(self.kernel_size) * self.channels_in_per_group +
            np.square(self.kernel_size / self.stride) * self.channels_out_per_group
        ))
        self.w = np.random.normal(0.0, stddev, self.w.shape)
        self.wb = np.zeros(self.shape_out)  # ... set the bias weights to 0

    def check(self):
        '''check layer configuration consistency'''

        assert self.shape_in[2] % self.groups == 0, \
            "Conv2D: layer shape_in[2] ({0}) has ".format(self.shape_in[2]) + \
            "to be a multiple of groups ({0}) !".format(self.groups)
        assert self.shape_out[2] % self.groups == 0, \
            "Conv2D: layer shape_out[2] ({0}) has ".format(self.shape_out[2]) + \
            "to be a multiple of groups ({0}) !".format(self.groups)

        assert self.shape_in[0] >= self.kernel_size, \
            "Conv2D: layer shape_in[0] ({0}) has ".format(self.shape_in[0]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)
        assert self.shape_in[1] >= self.kernel_size, \
            "Conv2D: layer shape_in[1] ({0}) has ".format(self.shape_in[1]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)

        assert (self.shape_in[0] - self.kernel_size) % self.stride == 0, \
            "Conv2D: layer shape_in[0] ({0}) ".format(self.shape_in[0]) + \
            "minus kernel_size ({0}) has ".format(self.kernel_size) + \
            "to be a multiple of stride ({0}) !".format(self.stride)
        assert (self.shape_in[1] - self.kernel_size) % self.stride == 0, \
            "Conv2D: layer shape_in[1] ({0}) ".format(self.shape_in[1]) + \
            "minus kernel_size ({0}) has ".format(self.kernel_size) + \
            "to be a multiple of stride ({0}) !".format(self.stride)

        assert self.shape_out[0] == self.steps_h, \
            "Conv2D: layer shape_out[0] ({0}) has ".format(self.shape_out[0]) + \
            "to be equal to layer internal steps_h ({0}) !".format(self.steps_h)
        assert self.shape_out[1] == self.steps_w, \
            "Conv2D: layer shape_out[1] ({0}) has ".format(self.shape_out[1]) + \
            "to be equal to layer internal steps_w ({0}) !".format(self.steps_w)

