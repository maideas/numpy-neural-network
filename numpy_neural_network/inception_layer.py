
import numpy as np
import numpy_neural_network as npnn
from numpy_neural_network import Layer

class Inception(Layer):
    '''Inception module layer'''

    def __init__(self, shape_in,
                        b0_c1_d_out,
            b1_c3_d_in, b1_c3_d_out,
            b2_c5_d_in, b2_c5_d_out,
                        b3_c1_d_out
        ):

        self.shape_out = (shape_in[0], shape_in[1],
            b0_c1_d_out + b1_c3_d_out + b2_c5_d_out + b3_c1_d_out
        )

        self.d_in        = shape_in[2]
        self.b0_c1_d_out = b0_c1_d_out
        self.b1_c3_d_in  = b1_c3_d_in
        self.b1_c3_d_out = b1_c3_d_out
        self.b2_c5_d_in  = b2_c5_d_in
        self.b2_c5_d_out = b2_c5_d_out
        self.b3_c1_d_out = b3_c1_d_out

        super(Inception, self).__init__(shape_in, self.shape_out, None)

        self.branches = [None, None, None, None]

        # branch 0 : 1x1 convolution + ReLU
        self.branches[0] = npnn.Sequential()
        self.branches[0].layers = [
            npnn.Conv2D(  # b0_c1
                shape_in = (self.shape_in[0], self.shape_in[1], self.d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b0_c1_d_out),
                kernel_size = 1, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b0_c1_d_out)
            )
        ]

        # branch 1 : 1x1 convolution + ReLU + 3x3 convolution + ReLU
        self.branches[1] = npnn.Sequential()
        self.branches[1].layers = [
            npnn.Conv2D(  # b1_c1
                shape_in = (self.shape_in[0], self.shape_in[1], self.d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b1_c3_d_in),
                kernel_size = 1, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b1_c3_d_in)
            ),
            npnn.Pad2D(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b1_c3_d_in),
                pad_axis0 = 1, pad_axis1 = 1
            ),
            npnn.Conv2D(  # b1_c3
                shape_in = (self.shape_in[0] + 2, self.shape_in[1] + 2, self.b1_c3_d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b1_c3_d_out),
                kernel_size = 3, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b1_c3_d_out)
            )
        ]

        # branch 2 : 1x1 convolution + ReLU + 5x5 convolution + ReLU
        self.branches[2] = npnn.Sequential()
        self.branches[2].layers = [
            npnn.Conv2D(  # b2_c1
                shape_in = (self.shape_in[0], self.shape_in[1], self.d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b2_c5_d_in),
                kernel_size = 1, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b2_c5_d_in)
            ),
            npnn.Pad2D(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b2_c5_d_in),
                pad_axis0 = 2, pad_axis1 = 2
            ),
            npnn.Conv2D(  # b2_c5
                shape_in = (self.shape_in[0] + 4, self.shape_in[1] + 4, self.b2_c5_d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b2_c5_d_out),
                kernel_size = 5, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b2_c5_d_out)
            )
        ]

        # branch 3 : 3x3 max pooling + 1x1 convolution + ReLU
        self.branches[3] = npnn.Sequential()
        self.branches[3].layers = [
            npnn.Pad2D(
                shape_in = (self.shape_in[0], self.shape_in[1], self.d_in),
                pad_axis0 = 1, pad_axis1 = 1
            ),
            npnn.MaxPool(  # b3_p3
                shape_in = (self.shape_in[0] + 2, self.shape_in[1] + 2, self.d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.d_in),
                kernel_size = 3, stride = 1
            ),
            npnn.Conv2D(  # b3_c1
                shape_in = (self.shape_in[0], self.shape_in[1], self.d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b3_c1_d_out),
                kernel_size = 1, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b3_c1_d_out)
            )
        ]

    def forward(self, x):
        b0_x = self.branches[0].forward(x)
        b1_x = self.branches[1].forward(x)
        b2_x = self.branches[2].forward(x)
        b3_x = self.branches[3].forward(x)

        y = np.zeros(self.shape_out)

        start, size =   0, self.b0_c1_d_out
        end = start + size
        y[:,:,start:end] = b0_x

        start, size = end, self.b1_c3_d_out
        end = start + size
        y[:,:,start:end] = b1_x

        start, size = end, self.b2_c5_d_out
        end = start + size
        y[:,:,start:end] = b2_x

        start, size = end, self.b3_c1_d_out
        end = start + size
        y[:,:,start:end] = b3_x

        return y

    def backward(self, grad_y):
        start, size =   0, self.b0_c1_d_out
        end = start + size
        b0_grad_y = grad_y[:,:,start:end]

        start, size = end, self.b1_c3_d_out
        end = start + size
        b1_grad_y = grad_y[:,:,start:end]

        start, size = end, self.b2_c5_d_out
        end = start + size
        b2_grad_y = grad_y[:,:,start:end]

        start, size = end, self.b3_c1_d_out
        end = start + size
        b3_grad_y = grad_y[:,:,start:end]

        self.grad_x = np.zeros(self.shape_in)
        self.grad_x += self.branches[0].backward(b0_grad_y)
        self.grad_x += self.branches[1].backward(b1_grad_y)
        self.grad_x += self.branches[2].backward(b2_grad_y)
        self.grad_x += self.branches[3].backward(b3_grad_y)

        return self.grad_x

    def zero_grad(self):
        for branch in self.branches:
            branch.zero_grad()

    def init_w(self):
        for branch in self.branches:
            branch.init_w()

    def step_init(self, is_training=False):
        for branch in self.branches:
            branch.step_init(is_training=is_training)

    def update_weights(self, callback):
        for branch in self.branches:
            branch.update_weights(callback=callback)


