
if 'CUDA' in globals() or 'CUDA' in locals():
    import cupy as np
else:
    import numpy as np

from numpy_neural_network import Layer

class FuncLayer(Layer):
    '''function layer base class'''

    def __init__(self, shape_in):
        '''
        size of shape_in = number of activation function connections (number of neurons)
        '''
        super(FuncLayer, self).__init__(shape_in, shape_in, None)


class Linear(FuncLayer):
    '''linear activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x) = x
        --------------------------------------------
        '''
        self.y = x.copy()
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = 1.0
        gradient(x) = 1.0 * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = grad_y.copy()
        return self.grad_x


class ReLU(FuncLayer):
    '''rectified linear unit activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        x >= 0 : f(x) = x
        --------------------------------------------
        x < 0 : f(x) = 0
        --------------------------------------------
        '''
        self.y = x.copy()
        self.y[x < 0.0] = 0.0
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        x >= 0 : f'(x) = 1.0
        gradient(x) = 1.0 * gradient(y)
        --------------------------------------------
        x < 0 : f'(x) = 0
        gradient(x) = 0
        --------------------------------------------
        '''
        self.grad_x = grad_y.copy()
        # because of the positive offset-free correlation of x and y, we can
        # use self.y instead of x to decide to set self.grad_x elements to 0 ...
        self.grad_x[self.y < 0.0] = 0.0
        return self.grad_x


class LeakyReLU(FuncLayer):
    '''leaky rectified linear unit activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        x >= 0 : f(x) = x
        --------------------------------------------
        x < 0 : f(x) = 0.1 * x
        --------------------------------------------
        '''
        self.y = x.copy()
        self.y[x < 0.0] *= 0.1
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        x >= 0 : f'(x) = 1.0
        gradient(x) = 1.0 * gradient(y)
        --------------------------------------------
        x <  0 : f'(x) = 0.1
        gradient(x) = 0.1 * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = grad_y.copy()
        # because of the positive offset-free correlation of x and y, we can use
        # self.y instead of x to decide to multiply self.grad_x elements by 0.1 ...
        self.grad_x[self.y < 0.0] *= 0.1
        return self.grad_x


class Tanh(FuncLayer):
    '''tanh activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x) = tanh(x)
        --------------------------------------------
        '''
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = (1.0 - (tanh(x))^2)
              = (1.0 - (f(x))^2)
        gradient(x) = (1.0 - (tanh(x))^2) * gradient(y)
                    = (1.0 - (f(x)^2) * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = (1.0 - np.square(self.y)) * grad_y
        return self.grad_x


class Sigmoid(FuncLayer):
    '''sigmoid (logistic) activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x) = 1.0 / (1.0 + e^-x)
             = sigmoid(x)
        --------------------------------------------
        '''
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = sigmoid(x) * (1.0 - sigmoid(x))
              = f(x) * (1.0 - f(x))
        gradient(x) = sigmoid(x) * (1.0 - sigmoid(x)) * gradient(y)
                    = f(x) * (1.0 - f(x)) * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = (self.y * (1.0 - self.y)) * grad_y
        return self.grad_x


class Softplus(FuncLayer):
    '''softplus activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x) = ln(1.0 + e^x)
        --------------------------------------------
        '''
        self.exp_x = np.exp(x)
        self.y = np.log(1.0 + self.exp_x)
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = e^x / (1.0 + e^x)
        gradient(x) = (e^x / (1.0 + e^x)) * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = (self.exp_x / (1.0 + self.exp_x)) * grad_y
        return self.grad_x


class Softmax(FuncLayer):
    '''softmax activation function'''

    def softmax(self, x):
        '''
        softmax function implementation
 
        for better numeric stability (http://cs231n.github.io/linear-classify/) ...
        -> max a value will be adjusted to 0 -> max e^a will be 1
        This adaption does not change the result of the softmax calculation.
        '''
        a = x - np.max(x)
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a)

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x)[i] = e^x[i] / sum_over_n(e^x[n])
        --------------------------------------------
        '''
        self.y = self.softmax(x)
        return self.y

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        n : softmax layer output index
        k : softmax layer input index
        --------------------------------------------
        n == k : gradient of y[n] w.r.t x[k] = softmax(x[n]) * (1.0 - softmax(x[k]))
                                             = y[n] * (1.0 - y[k])
        --------------------------------------------
        n != k : gradient of y[n] w.r.t x[k] = softmax(x[n]) * (0.0 - softmax(x[k]))
                                             = y[n] * (0.0 - y[k])
        --------------------------------------------
        '''
        self.grad_x = np.zeros(self.y.shape)

        for n in np.arange(self.y.shape[0]):
            for k in np.arange(self.y.shape[0]):
                if n == k:
                    self.grad_x[k] += self.y[n] * (1.0 - self.y[k]) * grad_y[n]
                else:
                    self.grad_x[k] += self.y[n] * (0.0 - self.y[k]) * grad_y[n]

        return self.grad_x

