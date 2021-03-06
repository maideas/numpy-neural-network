
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import Dense


def ref_forward(x, w, wb, shape_in, shape_out):

    # initialization of the output vector with zeros ...
    y = np.zeros(shape_out)

    # traverse output ...
    for y_h_index in np.arange(shape_out[0]):
        for y_w_index in np.arange(shape_out[1]):
            for y_d_index in np.arange(shape_out[2]):
                
                w_out_index = y_h_index * shape_out[1] * shape_out[2] + y_w_index * shape_out[2] + y_d_index

                # traverse input space related to the output index ...
                for x_h_index in np.arange(shape_in[0]):
                    for x_w_index in np.arange(shape_in[1]):
                        for x_d_index in np.arange(shape_in[2]):
             
                            x_val = x[x_h_index, x_w_index, x_d_index]
             
                            w_in_index = x_h_index * shape_in[1] * shape_in[2] + x_w_index * shape_in[2] + x_d_index
             
                            w_val = w[w_out_index, w_in_index]
             
                            # incremental output value update ...
                            y[y_h_index, y_w_index, y_d_index] += x_val * w_val
    y += wb
    return y


def ref_backward_gx(gy, w, shape_in, shape_out):

    # initialization of the output vector with zeros ...
    gx = np.zeros(shape_in)

    # traverse output ...
    for y_h_index in np.arange(shape_out[0]):
        for y_w_index in np.arange(shape_out[1]):
            for y_d_index in np.arange(shape_out[2]):

                w_out_index = y_h_index * shape_out[1] * shape_out[2] + y_w_index * shape_out[2] + y_d_index

                # traverse input space related to the output index ...
                for x_h_index in np.arange(shape_in[0]):
                    for x_w_index in np.arange(shape_in[1]):
                        for x_d_index in np.arange(shape_in[2]):
             
                            gy_val = gy[y_h_index, y_w_index, y_d_index]
             
                            w_in_index = x_h_index * shape_in[1] * shape_in[2] + x_w_index * shape_in[2] + x_d_index
             
                            w_val = w[w_out_index, w_in_index]
             
                            gx[x_h_index, x_w_index, x_d_index] += gy_val * w_val

    return gx


def ref_backward_gw(gy, x, shape_in, shape_out):

    # initialization of the weight gradient array with zeros ...
    gw = np.zeros((np.prod(shape_out), np.prod(shape_in)))

    # traverse output ...
    for y_h_index in np.arange(shape_out[0]):
        for y_w_index in np.arange(shape_out[1]):
            for y_d_index in np.arange(shape_out[2]):

                w_out_index = y_h_index * shape_out[1] * shape_out[2] + y_w_index * shape_out[2] + y_d_index

                # traverse input space related to the output index ...
                for x_h_index in np.arange(shape_in[0]):
                    for x_w_index in np.arange(shape_in[1]):
                        for x_d_index in np.arange(shape_in[2]):
             
                            gy_val = gy[y_h_index, y_w_index, y_d_index]
             
                            x_val = x[x_h_index, x_w_index, x_d_index]
             
                            w_in_index = x_h_index * shape_in[1] * shape_in[2] + x_w_index * shape_in[2] + x_d_index
             
                            # incremental weight gradient value update ...
                            gw[w_out_index, w_in_index] += gy_val * x_val
    return gw


class TestDense(unittest.TestCase):

    def test_dense_layer(self):

        print("test_dense_layer")

        # loop over different random layer configurations ...
        for episode in np.arange(100):

            if episode == 0:
                #============================================
                h_in = 1
                w_in = 1
                d_in = 1
                h_out = 1
                w_out = 1
                d_out = 1
                #============================================

            if episode == 1:
                #============================================
                h_in = 2
                w_in = 1
                d_in = 1
                h_out = 1
                w_out = 2
                d_out = 1
                #============================================

            if episode == 2:
                #============================================
                h_in = 1
                w_in = 2
                d_in = 1
                h_out = 1
                w_out = 1
                d_out = 2
                #============================================

            if episode == 3:
                #============================================
                h_in = 1
                w_in = 1
                d_in = 2
                h_out = 2
                w_out = 1
                d_out = 1
                #============================================

            if episode == 4:
                #============================================
                h_in = 1
                w_in = 1
                d_in = 1
                h_out = 2
                w_out = 2
                d_out = 2
                #============================================

            if episode > 4:
                #============================================
                h_in  = np.random.randint(1, 10)
                w_in  = np.random.randint(1, 10)
                d_in  = np.random.randint(1, 5)
                h_out = np.random.randint(1, 10)
                w_out = np.random.randint(1, 10)
                d_out = np.random.randint(1, 5)
                #============================================

            shape_in  = (h_in, w_in, d_in)
            shape_out = (h_out, w_out, d_out)
            shape_w   = (np.prod(shape_out), np.prod(shape_in))

            #================================================
            # network layer object to be tested ...

            print("shape_in={}, shape_out={}".format(shape_in, shape_out))

            layer = Dense(
                shape_in  = shape_in,
                shape_out = shape_out
            )

            #================================================

            # loop over different random input and weight values ...
            for pattern in np.arange(10):

                if pattern == 0:
                    # simple (all zeros) values ...
                    w  = np.zeros(shape_w)
                    wb = np.zeros(shape_out)
                    x  = np.zeros(shape_in)
                    gy = np.zeros(shape_out)

                if pattern == 1:
                    # simple (all ones) values ...
                    w  = np.ones(shape_w)
                    wb = np.ones(shape_out)
                    x  = np.ones(shape_in)
                    gy = np.ones(shape_out)

                if pattern > 1:
                    # random normal weights, input data and output side gradients ...
                    w  = np.random.normal(0.0, 1.0, shape_w)
                    wb = np.random.normal(0.0, 1.0, shape_out)
                    x  = np.random.normal(0.0, 1.0, shape_in)
                    gy = np.random.normal(0.0, 1.0, shape_out)

                # reference calculation ...
                y_ref  = ref_forward     (x,  w, wb, shape_in, shape_out)
                gx_ref = ref_backward_gx (gy, w,     shape_in, shape_out)
                gw_ref = ref_backward_gw (gy, x,     shape_in, shape_out)

                # set the layer weights according the reference values ...
                layer.w = w
                layer.wb = wb

                # layer forward pass ...
                y = layer.forward(x)

                # test almost equal ... layer output against reference output ...
                np.testing.assert_allclose(y_ref, y)

                # layer backward pass ...
                layer.zero_grad()
                gx = layer.backward(gy)
                gw = layer.grad_w
                gwb = layer.grad_wb

                # test almost equal ... layer input side gradients against reference gradients ...
                np.testing.assert_allclose(gx_ref, gx)

                # test almost equal ... layer weight gradients against reference gradients ...
                np.testing.assert_allclose(gw_ref, gw)

                # test almost equal ... layer bias weight gradients (which are equal to gy)
                np.testing.assert_allclose(gy, gwb)

