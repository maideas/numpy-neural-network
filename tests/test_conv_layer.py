
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import Conv2d

def conv2d_forward(x, w, kernel_size, stride, num_groups):

    steps_h = int(1 + (x.shape[0] - kernel_size) / stride)
    steps_w = int(1 + (x.shape[1] - kernel_size) / stride)
    kernel_depth = int(x.shape[2] / num_groups)  # kernel input depth !
    num_kernels = w.shape[1]

    y = np.zeros((steps_h, steps_w, num_groups*num_kernels))

    print("x_shape={}, y_shape={}, stride={}, steps_h={}, steps_w={}, kernel_depth={}, num_kernels={}, num_groups={}".format(
        x.shape, y.shape, stride, steps_h, steps_w, kernel_depth, num_kernels, num_groups
    ))

    for group_sel in np.arange(num_groups):
        for kernel_sel in np.arange(num_kernels):
            for step_h in np.arange(steps_h):
                for step_w in np.arange(steps_w):

                    y_h_index = step_h
                    y_w_index = step_w
                    y_d_index = (group_sel*num_kernels) + kernel_sel

                    for kernel_h in np.arange(kernel_size):
                        for kernel_w in np.arange(kernel_size):
                            for kernel_d in np.arange(kernel_depth):

                                x_h_index = (step_h*stride) + kernel_h
                                x_w_index = (step_w*stride) + kernel_w
                                x_d_index = (group_sel*kernel_depth) + kernel_d

                                w_ravel_index = (kernel_h*kernel_size*kernel_depth) + (kernel_w*kernel_depth) + kernel_d

                                x_val = x[x_h_index, x_w_index, x_d_index]
                                w_val = w[group_sel, kernel_sel, w_ravel_index]
                                y[y_h_index, y_w_index, y_d_index] += x_val * w_val

                    bias_val = w[group_sel, kernel_sel, -1]
                    y[step_h, step_w, y_d_index] += bias_val
    return y


class TestConv2D(unittest.TestCase):

    def test_conv2d_layer(self):

        # loop over different random layer configurations ...
        for _ in np.arange(200):

            #============================================
            kernel_size  = np.random.randint(1,  5)
            kernel_depth = np.random.randint(1,  5)
            num_kernels  = np.random.randint(1, 10)
            stride       = np.random.randint(1,  5)
            groups       = np.random.randint(1,  5)
            steps_h      = np.random.randint(1, 10)
            steps_w      = np.random.randint(1, 10)
            #============================================
 
            h_in = (steps_h - 1)*stride + kernel_size
            w_in = (steps_w - 1)*stride + kernel_size
            d_in = kernel_depth * groups
 
            h_out = steps_h
            w_out = steps_w
            d_out = num_kernels * groups
 
            shape_in = (h_in, w_in, d_in)
            shape_out = (h_out, w_out, d_out)
 
            # layer object ...
            layer = Conv2d(
                shape_in    = shape_in,
                shape_out   = shape_out,
                kernel_size = kernel_size,
                stride      = stride,
                groups      = groups
            )
 
            # loop over different random input and weight values ...
            for _ in np.arange(5):
         
                # some random layer weights ...
                w = np.random.normal(0.0, 1.0, (groups, int(d_out/groups), kernel_size * kernel_size * int(d_in/groups) + 1))
                layer.w = w
         
                # some random input data ...
                x = np.random.normal(0.0, 1.0, shape_in)
         
                # reference calculation ...
                y_ref = conv2d_forward(x, w, kernel_size, stride, groups)
         
                # layer forward pass ...
                y = layer.forward(x)
         
                # test almost equal ... layer output against reference output ...
                np.testing.assert_allclose(y, y_ref)

