
import numpy as np
from npnn_datasets import DataSet

class XORFunction(DataSet):

    def __init__(self, points=400, train_fraction=0.7):
        super(XORFunction, self).__init__()

        self.x_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.y_data = np.array([0.0, 1.0, 1.0, 0.0])

        self.x_data = np.repeat(self.x_data, points/4, axis=0)
        self.y_data = np.repeat(self.y_data, points/4, axis=0)

        self.x_data += np.random.normal(0.0, 0.05, self.x_data.shape)
        self.y_data += np.random.normal(0.0, 0.05, self.y_data.shape)

        self.gaussian_norm_x()
        self.prepare(train_fraction)

