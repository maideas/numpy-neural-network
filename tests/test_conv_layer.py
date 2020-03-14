
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import Conv2d

def conv2d_forward(x, w, kernel_size, kernel_depth, stride, num_groups):

    # calculation of some useful layer configuration values ...
    steps_h = int(1 + (x.shape[0] - kernel_size) / stride)
    steps_w = int(1 + (x.shape[1] - kernel_size) / stride)
    num_kernels = w.shape[1]

    # initialization of the 3D output array with zeros ...
    y = np.zeros((steps_h, steps_w, num_groups * num_kernels))

    # over all groups ...
    for group_sel in np.arange(num_groups):

        # traverse (3D) output of a group ...
        for kernel_sel in np.arange(num_kernels):
            for step_h in np.arange(steps_h):
                for step_w in np.arange(steps_w):

                    # single output indices ..
                    y_h_index = step_h
                    y_w_index = step_w
                    y_d_index = (group_sel * num_kernels) + kernel_sel

                    # traverse kernel input space related to the selected output value ...
                    for kernel_h in np.arange(kernel_size):
                        for kernel_w in np.arange(kernel_size):
                            for kernel_d in np.arange(kernel_depth):

                                # update the selected output value using the related x and w values ...
                                x_h_index = (step_h * stride) + kernel_h
                                x_w_index = (step_w * stride) + kernel_w
                                x_d_index = (group_sel * kernel_depth) + kernel_d
                                w_ravel_index = (kernel_h * kernel_size * kernel_depth) + (kernel_w * kernel_depth) + kernel_d

                                x_val = x[x_h_index, x_w_index, x_d_index]
                                w_val = w[group_sel, kernel_sel, w_ravel_index]

                                # incremental output value update ...
                                y[y_h_index, y_w_index, y_d_index] += x_val * w_val

                    # add (per-kernel) weighted bias information to the selected output value ...
                    bias_val = w[group_sel, kernel_sel, -1]
                    y[step_h, step_w, y_d_index] += bias_val
    return y


def conv2d_backward_gx(gy, w, kernel_size, kernel_depth, stride, num_groups):

    # calculation of some useful layer configuration values ...
    steps_h = gy.shape[0]
    steps_w = gy.shape[1]
    num_kernels = w.shape[1]

    h_in = (steps_h - 1) * stride + kernel_size
    w_in = (steps_w - 1) * stride + kernel_size
    d_in = kernel_depth * num_groups

    # initialization of the 3D input side gradient array with zeros ...
    gx = np.zeros((h_in, w_in, d_in))

    # over all groups ...
    for group_sel in np.arange(num_groups):

        # traverse (3D) output of a group ...
        for kernel_sel in np.arange(num_kernels):
            for step_h in np.arange(steps_h):
                for step_w in np.arange(steps_w):

                    # single output indices ..
                    y_h_index = step_h
                    y_w_index = step_w
                    y_d_index = (group_sel * num_kernels) + kernel_sel

                    # traverse kernel input space related to the selected output value ...
                    for kernel_h in np.arange(kernel_size):
                        for kernel_w in np.arange(kernel_size):
                            for kernel_d in np.arange(kernel_depth):

                                # update the selected input gradient value using the related gy and w values ...
                                x_h_index = (step_h * stride) + kernel_h
                                x_w_index = (step_w * stride) + kernel_w
                                x_d_index = (group_sel * kernel_depth) + kernel_d
                                w_ravel_index = (kernel_h * kernel_size * kernel_depth) + (kernel_w * kernel_depth) + kernel_d

                                gy_val = gy[y_h_index, y_w_index, y_d_index]
                                w_val = w[group_sel, kernel_sel, w_ravel_index]

                                # incremental input gradient value update ...
                                gx[x_h_index, x_w_index, x_d_index] += gy_val * w_val

    return gx


class TestConv2D(unittest.TestCase):

    def test_conv2d_layer(self):

        # loop over different random layer configurations ...
        for episode in np.arange(200):

            if episode == 0:
                #============================================
                kernel_size  = 1  # kernel INPUT height and width
                kernel_depth = 1  # kernel INPUT depth (per group)
                num_kernels  = 1  # output channels (per group)
                stride       = 1
                groups       = 1
                steps_h      = 1
                steps_w      = 1
                #============================================

            if episode == 1:
                #============================================
                kernel_size  = 1  # kernel INPUT height and width
                kernel_depth = 1  # kernel INPUT depth (per group)
                num_kernels  = 1  # output channels (per group)
                stride       = 1
                groups       = 1
                steps_h      = 2
                steps_w      = 2
                #============================================

            if episode == 2:
                #============================================
                kernel_size  = 2  # kernel INPUT height and width
                kernel_depth = 1  # kernel INPUT depth (per group)
                num_kernels  = 1  # output channels (per group)
                stride       = 2
                groups       = 1
                steps_h      = 2
                steps_w      = 2
                #============================================

            if episode == 3:
                #============================================
                kernel_size  = 3  # kernel INPUT height and width
                kernel_depth = 1  # kernel INPUT depth (per group)
                num_kernels  = 1  # output channels (per group)
                stride       = 2
                groups       = 1
                steps_h      = 2
                steps_w      = 2
                #============================================

            if episode == 4:
                #============================================
                kernel_size  = 3  # kernel INPUT height and width
                kernel_depth = 1  # kernel INPUT depth (per group)
                num_kernels  = 2  # output channels (per group)
                stride       = 2
                groups       = 1
                steps_h      = 2
                steps_w      = 2
                #============================================

            if episode > 4:
                #============================================
                kernel_size  = np.random.randint(1,  5)  # kernel INPUT height and width
                kernel_depth = np.random.randint(1,  5)  # kernel INPUT depth (per group)
                num_kernels  = np.random.randint(1, 10)  # output channels (per group)
                stride       = np.random.randint(1,  5)
                groups       = np.random.randint(1,  5)
                steps_h      = np.random.randint(1, 10)
                steps_w      = np.random.randint(1, 10)
                #============================================
 
            h_in = (steps_h - 1) * stride + kernel_size
            w_in = (steps_w - 1) * stride + kernel_size
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
 
            print("shape_in={}, shape_out={}, stride={}, steps_h={}, steps_w={}, kernel_size={}, kernel_depth={}, num_kernels={}, groups={}".format(
                shape_in, shape_out, stride, steps_h, steps_w, kernel_size, kernel_depth, num_kernels, groups
            ))

            # loop over different random input and weight values ...
            for _ in np.arange(5):
         
                # some random layer weights ...
                w = np.random.normal(0.0, 1.0, (groups, num_kernels, kernel_size * kernel_size * kernel_depth + 1))
                #w = np.ones((groups, num_kernels, kernel_size * kernel_size * kernel_depth + 1))
                layer.w = w
         
                # some random input data ...
                x = np.random.normal(0.0, 1.0, shape_in)
         
                # reference calculation ...
                y_ref = conv2d_forward(x, w, kernel_size, kernel_depth, stride, groups)
         
                # layer forward pass ...
                y = layer.forward(x)
         
                # test almost equal ... layer output against reference output ...
                np.testing.assert_allclose(y, y_ref)

                # some random output side gradients ...
                gy = np.random.normal(0.0, 1.0, shape_out)
                #gy = np.ones(shape_out)
         
                # reference calculation ...
                gx_ref = conv2d_backward_gx(gy, w, kernel_size, kernel_depth, stride, groups)

                # layer backward pass ...
                layer.zero_grad()
                gx = layer.backward(gy)

                # test almost equal ... layer input side gradients against reference gradients ...
                np.testing.assert_allclose(gx, gx_ref)

