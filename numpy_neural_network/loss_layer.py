
import numpy as np

class LossLayer:
    '''loss layer base class'''

    def __init__(self, shape_in):
        self.shape_in = shape_in
        self.x = np.zeros(self.shape_in)
        self.t = np.zeros(self.shape_in)
        self.loss = np.zeros(self.shape_in)
        self.accuracy = 0.0

    def forward(self, x, t):
        return x

    def backward(self):
        return np.zeros(self.shape_in)

    def step(self, x, t):
        self.accuracy += self.accuracy_increment(x, t)
        y = self.forward(x, t)
        g = self.backward()
        return g, y

    def predict(self, x, t=None):
        if t is not None:
            self.accuracy += self.accuracy_increment(x, t)
            y = self.forward(x, t)
        else:
            y = x
        return y

    def step_init(self, is_training):
        pass

    def zero_grad(self):
        self.loss = np.zeros(self.shape_in)
        self.accuracy = 0.0

    def accuracy_increment(self, x, t):
        if self.__class__.__name__ == "CrossEntropyLoss":
            # softmax + cross entropy loss accuracy ...
            if np.argmax(x) == np.argmax(t):
                return 1.0
        if self.__class__.__name__ == "BinaryCrossEntropyLoss":
            # sigmoid + binary cross entropy accuracy ...
            return np.array((x > 0.5) == (t > 0.5)).astype(int) / t.shape[0]
        return 0.0

    def update_weights(self, callback):
        pass


class RMSLoss(LossLayer):
    '''root mean square loss (L2 loss)'''

    def forward(self, x, t):
        '''
        function, used to calculate target loss
        --------------------------------------------
        f(x) = 0.5 * (x - t)^2
        --------------------------------------------
        '''
        self.x = x
        self.t = t
        self.loss += 0.5 * np.square(self.x - self.t)
        return self.x

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = (x - t)
        --------------------------------------------
        '''
        return self.x - self.t


class L1Loss(LossLayer):
    '''absolute (Manhatten) distance loss'''

    def forward(self, x, t):
        '''
        function, used to calculate target loss
        --------------------------------------------
        x >= 0 : f(x) = x
        --------------------------------------------
        x < 0 : f(x) = -x
        --------------------------------------------
        '''
        self.x = x
        self.t = t
        self.loss += np.absolute(self.x - self.t)
        return self.x

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        x >= 0 : f'(x) = 1.0
        --------------------------------------------
        x < 0 : f'(x) = -1.0
        --------------------------------------------
        '''
        grad_x = np.ones(self.shape_in)
        grad_x[(self.x - self.t) < 0.0] = -1.0
        return grad_x


class CrossEntropyLoss(LossLayer):
    '''
    distance between probabilities (negative Log-Likelihood).
    loss function for multi-class classification tasks, where
    only one class is the right one per network input sample
    -> one-hot (categorical) t class encoding
    '''

    def forward(self, x, t):
        '''
        function, used to calculate target loss
        --------------------------------------------
        f(x) = -t * ln(x)
        --------------------------------------------
        '''
        self.x = x
        self.t = t
        self.loss += -self.t * np.log(self.x)
        return self.x

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = -t / x
        --------------------------------------------
        '''
        return np.divide(-self.t, self.x)


class BinaryCrossEntropyLoss(LossLayer):
    '''
    loss function for (single-class) binary classification tasks.
    this loss function can also be used on vectors, where multiple
    features can be the right ones for an network input sample
    '''

    def forward(self, x, t):
        '''
        function, used to calculate target loss
        --------------------------------------------
        f(x) = -t * ln(x) - (1.0 - t) * ln(1.0 - x)
        --------------------------------------------
        '''
        self.x = x
        self.t = t
        self.loss += -self.t * np.log(self.x) - (1.0 - self.t) * np.log(1.0 - self.x)
        return self.x

    def backward(self):
        '''
        activation function derivative, used to pass gradients backward
        --------------------------------------------
        f'(x) = (-t / x) + ((1.0 - t) / (1.0 - x))
        --------------------------------------------
        '''
        return np.divide(-self.t, self.x) + np.divide((1.0 - self.t), (1.0 - self.x))

