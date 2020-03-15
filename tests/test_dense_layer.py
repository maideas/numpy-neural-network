
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import FullyConn


def ref_forward(x, w, size_in, size_out):

    # initialization of the output vector with zeros ...
    y = np.zeros(size_out)

    # traverse output ...
    for y_index in np.arange(size_out):

        # traverse input space related to the output index ...
        for x_index in np.arange(size_in):

            # update the selected output value using the related x and w values ...
            x_val = x[x_index]
            w_val = w[y_index, x_index]

            # incremental output value update ...
            y[y_index] += x_val * w_val

        # add weighted bias information to the selected output value ...
        bias_val = w[y_index, -1]
        y[y_index] += bias_val

    return y


def ref_backward_gx(gy, w, size_in, size_out):

    # initialization of the output vector with zeros ...
    gx = np.zeros(size_in)

    # traverse output ...
    for y_index in np.arange(size_out):

        # traverse input space related to the output index ...
        for x_index in np.arange(size_in):

            gy_val = gy[y_index]
            w_val = w[y_index, x_index]

            # incremental input gradient value update ...
            gx[x_index] += gy_val * w_val

    return gx


def ref_backward_gw(gy, x, size_in, size_out):

    # initialization of the weight gradient array with zeros ...
    gw = np.zeros((size_out, size_in + 1))

    # traverse output ...
    for y_index in np.arange(size_out):

        # traverse input space related to the output index ...
        for x_index in np.arange(size_in):

             gy_val = gy[y_index]
             x_val = x[x_index]

             # incremental weight gradient value update ...
             gw[y_index, x_index] += gy_val * x_val

        # bias weight update ...
        gy_val = gy[y_index]
        gw[y_index, -1] += gy_val

    return gw


class TestDense(unittest.TestCase):

    def test_dense_layer(self):

        # loop over different random layer configurations ...
        for episode in np.arange(200):

            if episode == 0:
                #============================================
                size_in  = 1
                size_out = 1
                #============================================

            if episode == 1:
                #============================================
                size_in  = 2
                size_out = 1
                #============================================

            if episode == 2:
                #============================================
                size_in  = 1
                size_out = 2
                #============================================

            if episode > 2:
                #============================================
                size_in  = np.random.randint(1, 20)
                size_out = np.random.randint(1, 20)
                #============================================

            shape_in  = (size_in)
            shape_out = (size_out)
            shape_w   = (size_out, size_in + 1)

            #================================================
            # network layer object to be tested ...

            print("size_in={}, size_out={}".format(size_in, size_out))

            layer = FullyConn(
                size_in  = size_in,
                size_out = size_out
            )

            #================================================

            # loop over different random input and weight values ...
            for pattern in np.arange(10):

                if pattern == 0:
                    # simple (all zeros) values ...
                    w  = np.zeros(shape_w)
                    x  = np.zeros(shape_in)
                    gy = np.zeros(shape_out)

                if pattern == 1:
                    # simple (all ones) values ...
                    w  = np.ones(shape_w)
                    x  = np.ones(shape_in)
                    gy = np.ones(shape_out)

                if pattern > 1:
                    # random normal weights, input data and output side gradients ...
                    w  = np.random.normal(0.0, 1.0, shape_w)
                    x  = np.random.normal(0.0, 1.0, shape_in)
                    gy = np.random.normal(0.0, 1.0, shape_out)

                # reference calculation ...
                y_ref  = ref_forward     (x,  w, size_in, size_out)
                gx_ref = ref_backward_gx (gy, w, size_in, size_out)
                gw_ref = ref_backward_gw (gy, x, size_in, size_out)

                # set the layer weights according the reference values ...
                layer.w = w

                # layer forward pass ...
                y = layer.forward(x)

                # test almost equal ... layer output against reference output ...
                np.testing.assert_allclose(y_ref, y)

                # layer backward pass ...
                layer.zero_grad()
                gx = layer.backward(gy)
                gw = layer.grad_w

                # test almost equal ... layer input side gradients against reference gradients ...
                np.testing.assert_allclose(gx_ref, gx)

                # test almost equal ... layer weight gradients against reference gradients ...
                np.testing.assert_allclose(gw_ref, gw)


