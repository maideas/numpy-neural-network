
import numpy as np

class Conv2d:
    '''2D convolutional layer'''

    def __init__(self, channels_in, channels_out, kernel_size, stride=1, groups=1):
        self.channels_in = channels_in
        self.channels_out = channels_out  # = number of kernels

        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

        assert self.channels_in % self.groups == 0, \
            "Conv2d: layer channels_in ({0}) has ".format(self.channels_in) + \
            "to be a multiple of groups ({0}) !".format(self.groups)
        assert self.channels_out % self.groups == 0, \
            "Conv2d: layer channels_out ({0}) has ".format(self.channels_out) + \
            "to be a multiple of groups ({0}) !".format(self.groups)

        self.channels_in_per_group = int(np.trunc(self.channels_in / self.groups))
        self.channels_out_per_group = int(np.trunc(self.channels_out / self.groups))

        self.kernel_size_in = (
            self.kernel_size * self.kernel_size * self.channels_in_per_group
        ) + 1  # plus bias node

        self.kernel_x = np.zeros(self.kernel_size_in)
        self.kernel_x[-1] = 1.0  # last vector element will be used as bias node of value 1

        self.w = np.zeros((self.groups, self.channels_out_per_group, self.kernel_size_in))
        self.grad_w = np.zeros(self.w.shape)  # layer weight adjustment gradients

        self.x = None  # conv layer input data of depth self.channels_in
        self.y = None  # conv layer output data of depth self.channels_out
        self.grad_x = None  # layer input gradients
        self.steps1 = 0
        self.steps2 = 0

        # optimizer dependent values (will be initialized by the selected optimizer) ...
        self.prev_dw = None
        self.ma_grad1 = None
        self.ma_grad2 = None

        self.init_w()

    def forward(self, x):
        '''
        data forward path
        input data -> weighted sums -> output data
        returns : layer output data
        '''
        assert x.shape[0] == self.channels_in, \
            "Conv2d: forward() data shape[0] ({0}) has ".format(x.shape[0]) + \
            "to be equal to layer channels_in ({0}) !".format(self.channels_in)

        assert x.shape[1] >= self.kernel_size, \
            "Conv2d: forward() data shape[1] ({0}) has ".format(x.shape[1]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)
        assert x.shape[2] >= self.kernel_size, \
            "Conv2d: forward() data shape[2] ({0}) has ".format(x.shape[2]) + \
            "to be equal or larger than kernel_size ({0}) !".format(self.kernel_size)

        assert (x.shape[1] - self.kernel_size) % self.stride == 0, \
            "Conv2d: forward() data shape[1] ({0}) ".format(x.shape[1]) + \
            "minus kernel_size ({0}) has ".format(self.kernel_size) + \
            "to be a multiple of stride ({0}) !".format(self.stride)
        assert (x.shape[2] - self.kernel_size) % self.stride == 0, \
            "Conv2d: forward() data shape[2] ({0}) ".format(x.shape[2]) + \
            "minus kernel_size ({0}) has ".format(self.kernel_size) + \
            "to be a multiple of stride ({0}) !".format(self.stride)

        self.steps1 = 1 + int(np.trunc((x.shape[1] - self.kernel_size) / self.stride))
        self.steps2 = 1 + int(np.trunc((x.shape[2] - self.kernel_size) / self.stride))

        self.x = x.copy()
        self.y = np.full((self.channels_out, self.steps1, self.steps2), np.nan)
        y = np.full((self.channels_out, self.steps1, self.steps2), np.nan)

        for group in np.arange(self.groups):
            for s1 in np.arange(self.steps1):
                for s2 in np.arange(self.steps2):

                    # get the current 3D slice out of input data x ...
                    self.kernel_x[:-1] = self.x[
                        group * self.channels_in_per_group : (group + 1) * self.channels_in_per_group,
                        s1 * self.stride : s1 * self.stride + self.kernel_size,
                        s2 * self.stride : s2 * self.stride + self.kernel_size
                    ].ravel()

                    # set output channel values to weighted slice data sums ...
                    self.y[
                        group * self.channels_out_per_group :
                        (group + 1) * self.channels_out_per_group, s1, s2
                    ] = np.matmul(self.w[group], self.kernel_x)

        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        output gradients (grad_y) -> derivative w.r.t weights -> weight gradients (grad_w)
        output gradients (grad_y) -> derivative w.r.t inputs -> input gradients (grad_x)
        returns : layer input gradients
        '''
        assert grad_y.shape[0] == self.channels_out, \
            "Conv2d: backward() gradient shape[0] ({0}) has ".format(grad_y.shape[0]) + \
            "to be equal to layer channels_out ({0}) !".format(self.channels_out)
        assert grad_y.shape[1] == self.steps1, \
            "Conv2d: backward() gradient shape[1] ({0}) has ".format(grad_y.shape[1]) + \
            "to be equal to layer internal steps1 ({0}) value !".format(self.steps1)
        assert grad_y.shape[2] == self.steps2, \
            "Conv2d: backward() gradient shape[2] ({0}) has ".format(grad_y.shape[2]) + \
            "to be equal to layer internal steps2 ({0}) value !".format(self.steps2)

        for group in np.arange(self.groups):
            for s1 in np.arange(self.steps1):
                for s2 in np.arange(self.steps2):

                    # get the current 3D slice out of input data x ...
                    self.kernel_x[:-1] = self.x[
                        group * self.channels_in_per_group : (group + 1) * self.channels_in_per_group,
                        s1 * self.stride : s1 * self.stride + self.kernel_size,
                        s2 * self.stride : s2 * self.stride + self.kernel_size
                    ].ravel()

                    for co in np.arange(self.channels_out_per_group):

                        # slice related single (scalar) output (y) gradient value ...
                        single_grad_y = grad_y[group * self.channels_out_per_group + co, s1, s2]

                        # weight (w) gradients calculation ...
                        self.grad_w[group, co] += self.kernel_x * single_grad_y

                        # input (x) gradients calculation ...
                        self.grad_x[
                            group * self.channels_in_per_group : (group + 1) * self.channels_in_per_group,
                            s1 * self.stride : s1 * self.stride + self.kernel_size,
                            s2 * self.stride : s2 * self.stride + self.kernel_size
                        ] += (self.w[group, co] * single_grad_y)[:-1].reshape(
                            self.channels_in_per_group,
                            self.kernel_size,
                            self.kernel_size
                        )

        return self.grad_x

    def zero_grad(self):
        '''set all gradient values to zero (preparation for incremental gradient calculation)'''
        self.grad_x = np.zeros(self.x.shape)
        self.grad_w = np.zeros(self.w.shape)

    def init_w(self):
        '''
        mean = 0
        variance = 2.0 / (kernel input size)
        bias weights = 0
        '''
        stddev = np.sqrt(2.0 / self.kernel_size_in)
        for group in np.arange(self.groups):
            self.w[group][:,:-1] = np.random.normal(
                0.0, stddev, (self.channels_out_per_group, self.kernel_size_in - 1)
            )
            self.w[group][:, -1] = 0.0  # ... set the bias weights to 0

