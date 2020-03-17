
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import MaxPool


def ref_forward(x, kernel_size, stride):

    # calculation of some useful layer configuration values ...
    steps_h = int(1 + (x.shape[0] - kernel_size) / stride)
    steps_w = int(1 + (x.shape[1] - kernel_size) / stride)

    h_out = steps_h
    w_out = steps_w
    d_out = x.shape[2]  # = d_in

    # initialization of the 3D output array with zeros ...
    y = np.zeros((h_out, w_out, d_out))

    # traverse (3D) output ...
    for d_index in np.arange(d_out):
        for step_h in np.arange(steps_h):
            for step_w in np.arange(steps_w):

                # single output indices ..
                y_h_index = step_h
                y_w_index = step_w

                # traverse kernel input space related to the selected output value ...
                x_vals = []
                for kernel_h in np.arange(kernel_size):
                    for kernel_w in np.arange(kernel_size):

                        x_h_index = (step_h * stride) + kernel_h
                        x_w_index = (step_w * stride) + kernel_w

                        x_vals.append(x[x_h_index, x_w_index, d_index])

                # output value update ...
                y[y_h_index, y_w_index, d_index] = max(x_vals)

    return y


def ref_backward_gx(x, gy, kernel_size, stride):

    # calculation of some useful layer configuration values ...
    steps_h = gy.shape[0]
    steps_w = gy.shape[1]

    h_in = (steps_h - 1) * stride + kernel_size
    w_in = (steps_w - 1) * stride + kernel_size
    d_in = gy.shape[2]

    h_out = steps_h
    w_out = steps_w
    d_out = gy.shape[2]

    # initialization of the 3D input side gradient array with zeros ...
    gx = np.zeros((h_in, w_in, d_in))

    # traverse (3D) output ...
    for d_index in np.arange(d_out):
        for step_h in np.arange(steps_h):
            for step_w in np.arange(steps_w):

                # single output indices ..
                y_h_index = step_h
                y_w_index = step_w

                # traverse kernel input space related to the selected output value ...
                max_x_h_index = step_h * stride
                max_x_w_index = step_w * stride
                max_x_val = x[max_x_h_index, max_x_w_index, d_index]

                for kernel_h in np.arange(kernel_size):
                    for kernel_w in np.arange(kernel_size):

                        x_h_index = (step_h * stride) + kernel_h
                        x_w_index = (step_w * stride) + kernel_w

                        if max_x_val < x[x_h_index, x_w_index, d_index]:
                            max_x_h_index = x_h_index
                            max_x_w_index = x_w_index
                            max_x_val = x[max_x_h_index, max_x_w_index, d_index]

                # input gradient value update ...
                gy_val = gy[y_h_index, y_w_index, d_index]
                gx[max_x_h_index, max_x_w_index, d_index] += gy_val

    return gx


class TestMaxPool(unittest.TestCase):

    def test_max_pool_layer(self):

        # loop over different random layer configurations ...
        for episode in np.arange(200):

            if episode == 0:
                #============================================
                kernel_size  = 1  # kernel INPUT height and width
                stride       = 1
                num_channels = 1
                steps_h      = 1
                steps_w      = 1
                #============================================

            if episode == 1:
                #============================================
                kernel_size  = 2  # kernel INPUT height and width
                stride       = 2
                num_channels = 1
                steps_h      = 1
                steps_w      = 1
                #============================================

            if episode == 2:
                #============================================
                kernel_size  = 2  # kernel INPUT height and width
                stride       = 1
                num_channels = 1
                steps_h      = 2
                steps_w      = 2
                #============================================

            if episode == 3:
                #============================================
                kernel_size  = 2  # kernel INPUT height and width
                stride       = 1
                num_channels = 2
                steps_h      = 2
                steps_w      = 2
                #============================================

            if episode > 3:
                #============================================
                kernel_size  = np.random.randint(1,  5)  # kernel INPUT height and width
                stride       = np.random.randint(1,  5)
                num_channels = np.random.randint(1, 10)
                steps_h      = np.random.randint(1, 10)
                steps_w      = np.random.randint(1, 10)
                #============================================

            h_in = (steps_h - 1) * stride + kernel_size
            w_in = (steps_w - 1) * stride + kernel_size
            d_in = num_channels

            h_out = steps_h
            w_out = steps_w
            d_out = num_channels

            shape_in  = (h_in, w_in, d_in)
            shape_out = (h_out, w_out, d_out)

            #================================================
            # network layer object to be tested ...

            print("shape_in={}, shape_out={}, steps_h={}, steps_w={}, kernel_size={}, stride={}, num_channels={}".format(
                shape_in, shape_out, steps_h, steps_w, kernel_size, stride, num_channels
            ))

            layer = MaxPool(
                shape_in    = shape_in,
                shape_out   = shape_out,
                kernel_size = kernel_size,
                stride      = stride
            )

            #================================================

            # loop over different random input and weight values ...
            for pattern in np.arange(5):

                if pattern == 0:
                    # simple (all zeros) values ...
                    x  = np.zeros(shape_in)
                    gy = np.zeros(shape_out)

                if pattern == 1:
                    # simple (all ones) values ...
                    x  = np.ones(shape_in)
                    gy = np.ones(shape_out)

                if pattern > 1:
                    # random normal weights, input data and output side gradients ...
                    x  = np.random.normal(0.0, 1.0, shape_in)
                    gy = np.random.normal(0.0, 1.0, shape_out)

                # reference calculation ...
                y_ref  = ref_forward     (x,     kernel_size, stride)
                gx_ref = ref_backward_gx (x, gy, kernel_size, stride)

                # layer forward pass ...
                y = layer.forward(x)

                # test almost equal ... layer output against reference output ...
                np.testing.assert_allclose(y_ref, y)

                # layer backward pass ...
                layer.zero_grad()
                gx = layer.backward(gy)

                # test almost equal ... layer input side gradients against reference gradients ...
                np.testing.assert_allclose(gx_ref, gx)


