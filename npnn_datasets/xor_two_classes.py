
import numpy as np
from npnn_datasets import DataSet

class XORTwoClasses(DataSet):

    def __init__(self, points=400, train_fraction=0.7):
        self.x_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.y_data = np.array([0.0, 1.0, 1.0, 0.0])

        self.x_data = np.repeat(self.x_data, points/4, axis=0)
        self.y_data = np.repeat(self.y_data, points/4, axis=0)

        self.x_data += np.random.normal(0.0, 0.05, self.x_data.shape)

        # conversion of 1-dimensional y_data into two-class representation ...
        a = []
        for y in self.y_data:
            a.append([0, 1] if y < 0.5 else [1, 0])
        self.y_data = np.array(a)

        self.prepare(train_fraction, True, True)

