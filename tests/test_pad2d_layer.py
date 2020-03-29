
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import Pad2D


def ref_forward(x, pad_axis0, pad_axis1, pad_value):

    h_in = x.shape[0]
    w_in = x.shape[1]
    d_in = x.shape[2]

    h_out = x.shape[0] + 2 * pad_axis0
    w_out = x.shape[1] + 2 * pad_axis1
    d_out = x.shape[2]  # = d_in

    # initialization of the 3D output array with "pad_value" ...
    y = np.full((h_out, w_out, d_out), pad_value)

    for d_index in np.arange(d_in):
        for h_index in np.arange(h_in):
            for w_index in np.arange(w_in):
                x_val = x[h_index, w_index, d_index]
                y[pad_axis0 + h_index, pad_axis1 + w_index, d_index] = x_val

    return y


def ref_backward_gx(x, gy, pad_axis0, pad_axis1):

    h_in = x.shape[0]
    w_in = x.shape[1]
    d_in = x.shape[2]

    gx = np.zeros((h_in, w_in, d_in))

    for d_index in np.arange(d_in):
        for h_index in np.arange(h_in):
            for w_index in np.arange(w_in):
                gy_val = gy[pad_axis0 + h_index, pad_axis1 + w_index, d_index]
                gx[h_index, w_index, d_index] = gy_val

    return gx


class TestPad2D(unittest.TestCase):

    def test_pad2D_layer(self):

        print("test_pad2D_layer")

        # loop over different random layer configurations ...
        for episode in np.arange(200):

            if episode == 0:
                #============================================
                size_axis0 = 1
                size_axis1 = 1
                size_axis2 = 1
                pad_axis0  = 0
                pad_axis1  = 0
                pad_value  = 0
                #============================================

            if episode == 1:
                #============================================
                size_axis0 = 2
                size_axis1 = 2
                size_axis2 = 1
                pad_axis0  = 1
                pad_axis1  = 0
                pad_value  = 0
                #============================================

            if episode == 2:
                #============================================
                size_axis0 = 2
                size_axis1 = 2
                size_axis2 = 1
                pad_axis0  = 0
                pad_axis1  = 1
                pad_value  = 0
                #============================================

            if episode == 3:
                #============================================
                size_axis0 = 3
                size_axis1 = 3
                size_axis2 = 2
                pad_axis0  = 1
                pad_axis1  = 1
                pad_value  = 0.5
                #============================================

            if episode > 3:
                #============================================
                size_axis0 = np.random.randint(1, 10)
                size_axis1 = np.random.randint(1, 10)
                size_axis2 = np.random.randint(1, 10)
                pad_axis0  = np.random.randint(1, 5)
                pad_axis1  = np.random.randint(1, 5)
                pad_value  = np.random.normal(0.0, 1.0)
                #============================================

            shape_in = (size_axis0, size_axis1, size_axis2)
            shape_out = (size_axis0 + 2 * pad_axis0, size_axis1 + 2 * pad_axis1, size_axis2)

            #================================================
            # network layer object to be tested ...

            print("shape_in={}, pad_axis0={}, pad_axis1={}, pad_value={}".format(
                shape_in, pad_axis0, pad_axis1, pad_value
            ))

            layer = Pad2D(
                shape_in  = shape_in,
                pad_axis0 = pad_axis0,
                pad_axis1 = pad_axis1,
                pad_value = pad_value
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
                y_ref  = ref_forward     (x,     pad_axis0, pad_axis1, pad_value)
                gx_ref = ref_backward_gx (x, gy, pad_axis0, pad_axis1)

                # layer forward pass ...
                y = layer.forward(x)

                # test almost equal ... layer output against reference output ...
                np.testing.assert_allclose(y_ref, y)

                # layer backward pass ...
                layer.zero_grad()
                gx = layer.backward(gy)

                # test almost equal ... layer input side gradients against reference gradients ...
                np.testing.assert_allclose(gx_ref, gx)


