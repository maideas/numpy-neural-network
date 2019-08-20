
import numpy as np

class FuncLayer:
    '''function layer base class'''

    def __init__(self, size):
        '''
        size : number of activation function connections (number of neurons)
        '''
        self.size = size
        self.x = np.zeros(self.size)
        self.y = np.zeros(self.size)
        self.w = None
        self.grad_x = np.zeros(self.size)

    def zero_grad(self):
        '''set all gradient values to zero (preparation for incremental gradient calculation)'''
        self.grad_x = np.zeros(self.size)


class Linear(FuncLayer):
    '''linear activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x) = x
        --------------------------------------------
        '''
        self.x = x
        self.y = self.x.copy()
        return self.y.copy()

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = 1.0
        gradient(x) = 1.0 * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = grad_y.copy()
        return self.grad_x.copy()


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
        self.x = x
        self.y = self.x.copy()
        self.y[self.x < 0.0] = 0.0
        return self.y.copy()

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
        self.grad_x[self.x < 0.0] = 0.0
        return self.grad_x.copy()


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
        self.x = x
        for n in np.arange(self.size):
            if self.x[n] > 0.0:
                self.y[n] = self.x[n]
            else:
                self.y[n] = 0.1 * self.x[n]
        return self.y.copy()

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
        for n in np.arange(self.size):
            if self.x[n] > 0.0:
                self.grad_x[n] = grad_y[n]
            else:
                self.grad_x[n] = 0.1 * grad_y[n]
        return self.grad_x.copy()


class Tanh(FuncLayer):
    '''tanh activation function'''

    def forward(self, x):
        '''
        activation function, used to pass data forward
        --------------------------------------------
        f(x) = tanh(x)
        --------------------------------------------
        '''
        self.x = x
        self.y = np.tanh(self.x)
        return self.y.copy()

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
        return self.grad_x.copy()


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
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-self.x))
        return self.y.copy()

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
        return self.grad_x.copy()


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
        self.x = x
        self.y = self.softmax(self.x)
        return self.y.copy()

    def backward(self, grad_y):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = softmax(x) * (1.0 - softmax(x))
              = f(x) * (1.0 - f(x))
        gradient(x) = softmax(x) * (1.0 - softmax(x)) * gradient(y)
                    = f(x) * (1.0 - f(x)) * gradient(y)
        --------------------------------------------
        '''
        self.grad_x = self.y * (1.0 - self.y) * grad_y
        return self.grad_x.copy()

