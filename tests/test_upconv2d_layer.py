
import sys
sys.path.append("../")

import unittest
import numpy as np
from numpy_neural_network import UpConv2D


def ref_forward(x, w, wb, kernel_size, kernel_depth, num_kernels, stride, num_groups):

    steps_h = x.shape[0]
    steps_w = x.shape[1]

    h_out = (steps_h - 1) * stride + kernel_size
    w_out = (steps_w - 1) * stride + kernel_size
    d_out = kernel_depth * num_groups

    # initialize output tensor with zeros ...
    y = np.zeros((h_out, w_out, d_out))

    for group_sel in np.arange(num_groups):        # over groups
        for kernel_sel in np.arange(num_kernels):  # over input d = over all kernels

            for step_h in np.arange(steps_h):      # over input h
                for step_w in np.arange(steps_w):  # over input w

                    # single input element indices ...
                    x_h_index = step_h
                    x_w_index = step_w
                    x_d_index = (group_sel * num_kernels) + kernel_sel

                    # get the input element ...
                    x_val = x[x_h_index, x_w_index, x_d_index]

                    # output tensor base indices ...
                    y_h_index = step_h * stride
                    y_w_index = step_w * stride
                    y_d_index = kernel_depth * group_sel

                    # element-wise product of w_tensor and selected x value
                    for kernel_h in np.arange(kernel_size):
                        for kernel_w in np.arange(kernel_size):
                            for kernel_d in np.arange(kernel_depth):
                                
                                w_val = w[group_sel, kernel_sel, kernel_h, kernel_w, kernel_d]
                                y[y_h_index + kernel_h, y_w_index + kernel_w, y_d_index + kernel_d] += w_val * x_val
    y += wb
    return y


def ref_backward_gx(gy, w, x, kernel_size, kernel_depth, num_kernels, stride, num_groups):

    steps_h = x.shape[0]
    steps_w = x.shape[1]

    gx = np.zeros(x.shape)

    for group_sel in np.arange(num_groups):        # over groups
        for kernel_sel in np.arange(num_kernels):  # over input d = over all kernels

            for step_h in np.arange(steps_h):      # over input h
                for step_w in np.arange(steps_w):  # over input w

                    # single input element indices ...
                    x_h_index = step_h
                    x_w_index = step_w
                    x_d_index = (group_sel * num_kernels) + kernel_sel

                    # output tensor base indices ...
                    y_h_index = step_h * stride
                    y_w_index = step_w * stride
                    y_d_index = kernel_depth * group_sel

                    for kernel_h in np.arange(kernel_size):
                        for kernel_w in np.arange(kernel_size):
                            for kernel_d in np.arange(kernel_depth):

                                gy_val = gy[y_h_index + kernel_h, y_w_index + kernel_w, y_d_index + kernel_d]
                                w_val = w[group_sel, kernel_sel, kernel_h, kernel_w, kernel_d]

                                gx[x_h_index, x_w_index, x_d_index] += gy_val * w_val
    return gx


def ref_backward_gw(gy, x, kernel_size, kernel_depth, num_kernels, stride, num_groups):

    steps_h = x.shape[0]
    steps_w = x.shape[1]

    # initialization of the 3D weight gradient tensor with zeros ...
    gw = np.zeros((num_groups, num_kernels, kernel_size, kernel_size, kernel_depth))

    for group_sel in np.arange(num_groups):        # over groups
        for kernel_sel in np.arange(num_kernels):  # over input d = over all kernels

            for step_h in np.arange(steps_h):      # over input h
                for step_w in np.arange(steps_w):  # over input w

                    # single input element indices ...
                    x_h_index = step_h
                    x_w_index = step_w
                    x_d_index = (group_sel * num_kernels) + kernel_sel

                    # output tensor base indices ...
                    y_h_index = step_h * stride
                    y_w_index = step_w * stride
                    y_d_index = kernel_depth * group_sel

                    for kernel_h in np.arange(kernel_size):
                        for kernel_w in np.arange(kernel_size):
                            for kernel_d in np.arange(kernel_depth):

                                x_val = x[x_h_index, x_w_index, x_d_index]
                                gy_val = gy[y_h_index + kernel_h, y_w_index + kernel_w, y_d_index + kernel_d]

                                gw[group_sel, kernel_sel, kernel_h, kernel_w, kernel_d] += gy_val * x_val
    return gw


class TestUpConv2D(unittest.TestCase):

    def test_upconv2D_layer(self):

        print("test_upconv2D_layer")

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

            h_out = (steps_h - 1) * stride + kernel_size
            w_out = (steps_w - 1) * stride + kernel_size
            d_out = kernel_depth * groups

            h_in = steps_h
            w_in = steps_w
            d_in = num_kernels * groups

            shape_in  = (h_in, w_in, d_in)
            shape_out = (h_out, w_out, d_out)
            shape_w   = (groups, num_kernels, kernel_size, kernel_size, kernel_depth)
            shape_b   = shape_out

            #================================================
            # network layer object to be tested ...

            print("shape_in={}, shape_out={}, stride={}, steps_h={}, steps_w={}, kernel_size={}, kernel_depth={}, num_kernels={}, groups={}".format(
                shape_in, shape_out, stride, steps_h, steps_w, kernel_size, kernel_depth, num_kernels, groups
            ))

            layer = UpConv2D(
                shape_in    = shape_in,
                shape_out   = shape_out,
                kernel_size = kernel_size,
                stride      = stride,
                groups      = groups
            )

            #================================================

            # loop over different random input and weight values ...
            for pattern in np.arange(5):

                if pattern == 0:
                    # simple (all zeros) values ...
                    w  = np.zeros(shape_w)
                    wb = np.zeros(shape_b)
                    x  = np.zeros(shape_in)
                    gy = np.zeros(shape_out)

                if pattern == 1:
                    # simple (all ones) values ...
                    w  = np.ones(shape_w)
                    wb = np.ones(shape_b)
                    x  = np.ones(shape_in)
                    gy = np.ones(shape_out)

                if pattern > 1:
                    # random normal weights, input data and output side gradients ...
                    w  = np.random.normal(0.0, 1.0, shape_w)
                    wb = np.random.normal(0.0, 1.0, shape_b)
                    x  = np.random.normal(0.0, 1.0, shape_in)
                    gy = np.random.normal(0.0, 1.0, shape_out)

                # reference calculation ...
                y_ref  = ref_forward     (x,  w, wb, kernel_size, kernel_depth, num_kernels, stride, groups)
                gx_ref = ref_backward_gx (gy, w, x,  kernel_size, kernel_depth, num_kernels, stride, groups)
                gw_ref = ref_backward_gw (gy, x,     kernel_size, kernel_depth, num_kernels, stride, groups)

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


