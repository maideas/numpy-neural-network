
import numpy as np

class Conv2D:
    '''2D convolutional layer'''

    def __init__(self, shape_in, shape_out, kernel_size, stride=1, groups=1):
        self.shape_in = shape_in
        self.shape_out = shape_out  # shape_out[2] = number of kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

        self.channels_in_per_group = int(np.trunc(self.shape_in[2] / self.groups))
        self.channels_out_per_group = int(np.trunc(self.shape_out[2] / self.groups))

        self.steps_h = 1 + int(np.trunc((self.shape_in[0] - self.kernel_size) / self.stride))
        self.steps_w = 1 + int(np.trunc((self.shape_in[1] - self.kernel_size) / self.stride))

        self.check()

        self.kernel_size_in = (
            self.kernel_size * self.kernel_size * self.channels_in_per_group
        ) + 1  # plus bias node

        self.w = np.zeros((self.groups, self.channels_out_per_group, self.kernel_size_in))
        self.grad_w = np.zeros(self.w.shape)  # layer weight adjustment gradients

        self.x = np.zeros(self.shape_in)  # layer input data
        self.y = np.zeros(self.shape_out)  # layer output data
        self.grad_x = np.zeros(self.shape_in)  # layer input gradients

        # optimizer dependent values (will be initialized by the selected optimizer) ...
        self.prev_dw = None
        self.ma_grad1 = None
        self.ma_grad2 = None

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

        self.x = x.copy()
        self.y = np.full(self.shape_out, np.nan)

        for x_index, y_index, w_index in zip(self.x_indices, self.y_indices, self.w_indices):

            # get the current 3D slice out of input data x ...
            # last vector element will be used as bias node of value 1
            kernel_x = np.concatenate((self.x[x_index].ravel(), [1.0]), axis=0)

            # set output channel values to weighted slice data sums ...
            self.y[y_index] = np.matmul(self.w[w_index], kernel_x)

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

        for x_index, y_index, w_index in zip(self.x_indices, self.y_indices, self.w_indices):

            # get the current 3D slice out of input data x ...
            # last vector element will be used as bias node of value 1
            kernel_x = np.concatenate((self.x[x_index].ravel(), [1.0]), axis=0)

            # weight (w) gradients calculation ...
            self.grad_w[w_index] += np.outer(grad_y[y_index], kernel_x)

            # input (x) gradients calculation ...
            self.grad_x[x_index] += np.matmul(grad_y[y_index], self.w[w_index])[:-1].reshape(
                self.kernel_size,
                self.kernel_size,
                self.channels_in_per_group
            )

        return self.grad_x

    def zero_grad(self):
        '''set all gradient values to zero (preparation for incremental gradient calculation)'''
        self.grad_x = np.zeros(self.grad_x.shape)
        self.grad_w = np.zeros(self.grad_w.shape)

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
        for group in np.arange(self.groups):
            self.w[group][:,:-1] = np.random.normal(
                0.0, stddev, (self.channels_out_per_group, self.kernel_size_in - 1)
            )
            self.w[group][:, -1] = 0.0  # ... set the bias weights to 0

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        pass

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


class Pad2D:
    '''2D padding layer'''

    def __init__(self, shape_in, pad_axis0=0, pad_axis1=0, pad_value=0):
        self.shape_in = shape_in
        self.pad_axis0 = pad_axis0
        self.pad_axis1 = pad_axis1

        self.w = None
        self.y = np.full((
            shape_in[0] + 2 * pad_axis0,
            shape_in[1] + 2 * pad_axis1,
            shape_in[2]
        ), pad_value)

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

    def zero_grad(self):
        pass

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        pass

