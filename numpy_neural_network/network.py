
import numpy as np

class Model:
    '''encapsulates the network model structure (e.g. it's layer definition, etc.)'''

    def __init__(self, layers):
        self.layers = layers
        self.loss_layer = None

    def predict(self, x):
        '''
        network model forward path calculation (prediction) of a given x vector
        x : network model input data vector
        returns : network model output data vector
        '''
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_class(self, x):
        '''
        predict class related to input vector x
        x : network model input data vector
        returns : class integer value
        '''
        y = self.predict(x)
        return np.argmax(y)

