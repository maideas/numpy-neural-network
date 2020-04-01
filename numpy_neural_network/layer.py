
import numpy as np

class Layer:
    '''layer base class'''

    def __init__(self, shape_in, shape_out, shape_w):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.shape_w = shape_w

        self.w = np.zeros(self.shape_w)
        self.wb = np.zeros(self.shape_out)
        self.y = np.zeros(self.shape_out)

        self.grad_x = np.zeros(self.shape_in)
        self.grad_w = np.zeros(self.shape_w)
        self.grad_wb = np.zeros(self.shape_out)

        self.batch_size_count = 1
        self.is_training = False

        # optimizer dependent values, which will get
        # initialized by the selected optimizer ...
        self.prev_d_w   = None
        self.ma_grad1_w = None
        self.ma_grad2_w = None

        self.prev_d_wb   = None
        self.ma_grad1_wb = None
        self.ma_grad2_wb = None

    def forward(self, x):
        '''
        data forward path
        input data -> weighted sums -> output data
        returns : layer output data
        '''
        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        output gradients (grad_y) -> derivative w.r.t weights -> weight gradients (grad_w)
        output gradients (grad_y) -> derivative w.r.t inputs -> input gradients (grad_x)
        returns : layer input gradients
        '''
        return self.grad_x

    def zero_grad(self):
        '''
        set all gradient values to zero
        (preparation for incremental gradient calculation)
        '''
        self.grad_x = np.zeros(self.shape_in)
        self.grad_w = np.zeros(self.shape_w)
        self.grad_wb = np.zeros(self.shape_out)
        self.batch_size_count = 1

    def init_w(self):
        '''
        weight initialization
        '''
        pass

    def step_init(self, is_training=False):
        '''
        this method may initialize some layer internals before each optimizer mini-batch step
        '''
        self.is_training = is_training

    def update_weights(self, callback):
        self.grad_w /= self.batch_size_count
        self.grad_wb /= self.batch_size_count
        self.batch_size_count = 1

        callback(self)

