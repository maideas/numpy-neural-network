
import numpy as np

class LossLayer:
    '''loss layer base class'''

    def __init__(self, size):
        self.size = size
        self.x = np.zeros(self.size)
        self.target = np.zeros(self.size)


class RMSLoss(LossLayer):
    '''root mean square loss (L2 loss)'''

    def forward(self, x, target):
        '''
        function, used to calculate target loss
        --------------------------------------------
        f(x) = 0.5 * (x - target)^2
        --------------------------------------------
        '''
        self.x = x
        self.target = target
        return 0.5 * np.square(self.x - self.target)

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = (x - target)
        --------------------------------------------
        '''
        return self.x - self.target


class L1Loss(LossLayer):
    '''absolute (Manhatten) distance loss'''

    def forward(self, x, target):
        '''
        function, used to calculate target loss
        --------------------------------------------
        x >= 0 : f(x) = x
        --------------------------------------------
        x < 0 : f(x) = -x
        --------------------------------------------
        '''
        self.x = x
        self.target = target
        return np.absolute(self.x - self.target)

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        x >= 0 : f'(x) = 1.0
        --------------------------------------------
        x < 0 : f'(x) = -1.0
        --------------------------------------------
        '''
        grad_x = np.ones(self.size)
        grad_x[(self.x - self.target) < 0.0] = -1.0
        return grad_x


class CrossEntropyLoss(LossLayer):
    '''
    distance between probabilities (negative Log-Likelihood).
    loss function for multi-class classification tasks, where
    only one class is the right one per network input sample
    -> one-hot (categorical) target class encoding
    '''

    def forward(self, x, target):
        '''
        function, used to calculate target loss
        --------------------------------------------
        f(x) = -target * ln(x)
        --------------------------------------------
        '''
        self.x = x
        self.target = target
        return -self.target * np.log(self.x)

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = -target / x
        --------------------------------------------
        '''
        min_x = 1e-6  # to prevent divide by 0

        grad_x = np.zeros(self.size)
        for n in np.arange(self.size):
            if self.x[n] > min_x:
                grad_x[n] = np.divide(-self.target[n], self.x[n])
            else:
                grad_x[n] = np.divide(-self.target[n], min_x)
        return grad_x


class BinaryCrossEntropyLoss(LossLayer):
    '''
    loss function for (single-class) binary classification tasks.
    this loss function can also be used on vectors, where multiple
    features can be the right ones for an network input sample
    '''

    def forward(self, x, target):
        '''
        function, used to calculate target loss
        --------------------------------------------
        f(x) = -target * ln(x) - (1.0 - target) * ln(1.0 - x)
        --------------------------------------------
        '''
        self.x = x
        self.target = target
        return -self.target * np.log(self.x) - (1.0 - self.target) * np.log(1.0 - self.x)

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = (-target / x) + ((1.0 - target) / (1.0 - x))
        --------------------------------------------
        '''
        # to prevent divide by 0
        min_x = 1e-6
        max_x = 1.0 - min_x

        grad_x = np.zeros(self.size)
        for n in np.arange(self.size):
            if self.x[n] < min_x:
                grad_x_part1 = np.divide(-self.target[n], min_x)
                grad_x_part2 = np.divide((1.0 - self.target[n]), (1.0 - self.x[n]))

            elif self.x[n] > max_x:
                grad_x_part1 = np.divide(-self.target[n], self.x[n])
                grad_x_part2 = np.divide((1.0 - self.target[n]), max_x)

            else:
                grad_x_part1 = np.divide(-self.target[n], self.x[n])
                grad_x_part2 = np.divide((1.0 - self.target[n]), (1.0 - self.x[n]))

            grad_x[n] = grad_x_part1 + grad_x_part2
        return grad_x


