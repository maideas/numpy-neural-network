
if 'CUDA' in globals() or 'CUDA' in locals():
    import cupy as np
else:
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
        self.branches[0] = npnn.network.Model([
            npnn.Conv2D(  # b0_c1
                shape_in = (self.shape_in[0], self.shape_in[1], self.d_in),
                shape_out = (self.shape_in[0], self.shape_in[1], self.b0_c1_d_out),
                kernel_size = 1, stride = 1
            ),
            npnn.LeakyReLU(
                shape_in = (self.shape_in[0], self.shape_in[1], self.b0_c1_d_out)
            )
        ])

        # branch 1 : 1x1 convolution + ReLU + 3x3 convolution + ReLU
        self.branches[1] = npnn.network.Model([
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
        ])

        # branch 2 : 1x1 convolution + ReLU + 5x5 convolution + ReLU
        self.branches[2] = npnn.network.Model([
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
        ])

        # branch 3 : 3x3 max pooling + 1x1 convolution + ReLU
        self.branches[3] = npnn.network.Model([
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
        ])

    def forward(self, x):
        '''
        data forward path
        returns : layer output data
        '''
        b0_x = x
        b1_x = x
        b2_x = x
        b3_x = x

        for layer in self.branches[0].layers:
            b0_x = layer.forward(b0_x)
        for layer in self.branches[1].layers:
            b1_x = layer.forward(b1_x)
        for layer in self.branches[2].layers:
            b2_x = layer.forward(b2_x)
        for layer in self.branches[3].layers:
            b3_x = layer.forward(b3_x)

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
        '''
        gradients backward path
        returns : layer input gradients
        '''
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

        for layer in self.branches[0].layers[::-1]:
            b0_grad_y = layer.backward(b0_grad_y)
        for layer in self.branches[1].layers[::-1]:
            b1_grad_y = layer.backward(b1_grad_y)
        for layer in self.branches[2].layers[::-1]:
            b2_grad_y = layer.backward(b2_grad_y)
        for layer in self.branches[3].layers[::-1]:
            b3_grad_y = layer.backward(b3_grad_y)

        self.grad_x = np.zeros(self.shape_in)
        self.grad_x += b0_grad_y
        self.grad_x += b1_grad_y
        self.grad_x += b2_grad_y
        self.grad_x += b3_grad_y

        return self.grad_x

    def zero_grad(self):
        '''
        set all gradient values to zero
        '''
        for branch in self.branches:
            for layer in branch.layers:
                layer.zero_grad()

    def init_w(self):
        '''
        weight initialization
        '''
        for branch in self.branches:
            for layer in branch.layers:
                layer.init_w()

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        for branch in self.branches:
            for layer in branch.layers:
                layer.step_init(is_training=is_training)

    def update_weights(self, callback):
        '''
        weight update
        '''
        for branch in self.branches:
            for layer in branch.layers:
                layer.update_weights(callback=callback)

