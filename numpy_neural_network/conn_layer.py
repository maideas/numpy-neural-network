
import numpy as np

class FullyConn:
    '''fully connected layer'''

    def __init__(self, size_in, size_out):
        self.size_in = size_in + 1  # plus bias node
        self.size_out = size_out

        self.x = np.zeros(self.size_in)
        self.x[-1] = 1.0  # last vector element will be used as bias node of value 1
        self.y = np.zeros(self.size_out)
        self.grad_x = np.zeros(self.size_in)  # layer input gradients
        self.w = np.zeros((self.size_out, self.size_in))
        self.grad_w = np.zeros((self.size_out, self.size_in))  # layer weight adjustment gradients

        # optimizer dependent values (will be initialized by the selected optimizer) ...
        self.prev_dw = None
        self.ma_grad1 = None
        self.ma_grad2 = None

        self.init_w()

    def forward(self, x):
        '''
        data forward path
        input data -> weighted sums -> output data
        returns : layer output data
        '''
        self.x[:-1] = x.copy()
        self.y = np.matmul(self.w, self.x)
        return self.y

    def backward(self, grad_y):
        '''
        gradients backward path
        output gradients (grad_y) -> derivative w.r.t inputs -> weight gradients (grad_w)
        output gradients (grad_y) -> derivative w.r.t weights -> input gradients (grad_x)
        returns : layer input gradients
        '''
        self.grad_x = np.zeros(self.size_in)
        for n in np.arange(self.size_out):
            self.grad_w[n] = self.x * grad_y[n]
            self.grad_x += self.w[n] * grad_y[n]
        return self.grad_x[:-1]  # ... removal of bias gradient from return value

    def zero_grad(self):
        '''set all gradient values to zero (preparation for incremental gradient calculation)'''
        self.grad_x = np.zeros(self.size_in)
        self.grad_w = np.zeros((self.size_out, self.size_in))

    def init_w(self):
        '''
        weight initialization (Xavier Glorot et al.) ...
        mean = 0
        variance = sqrt(6) / (num neurons in previous layer + num neurons in this layer)
        bias weights = 0
        '''
        stddev = np.sqrt(2.45 / (self.size_in + self.size_out))
        self.w[:,:-1] = np.random.normal(0.0, stddev, (self.size_out, self.size_in - 1))
        self.w[:, -1] = 0.0  # ... set the bias weights to 0

