
import numpy as np

#===============================================================================

class LossLayer:
    '''loss layer base class'''

    def __init__(self, size):
        self.size = size
        self.x = np.zeros(self.size)
        self.target = np.zeros(self.size)

#===============================================================================

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

#===============================================================================

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

#===============================================================================

class CrossEntropyLoss(LossLayer):
    '''distance between probabilities (negative Log-Likelihood)'''

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
        min_x = 1e-3  # prevents divide by 0
        grad_x = np.zeros(self.size)
        for n in np.arange(self.size):
            if self.x[n] > min_x:
                grad_x[n] = np.divide(-self.target[n], self.x[n])
            else:
                grad_x[n] = np.divide(-self.target[n], min_x)
        return grad_x

#===============================================================================

