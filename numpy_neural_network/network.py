
if 'CUDA' in globals() or 'CUDA' in locals():
    import cupy as np
else:
    import numpy as np

class Model:
    '''encapsulates the network model structure (e.g. its layer definition, etc.)'''

    def __init__(self, layers):
        self.layers = layers
        self.loss_layer = None

