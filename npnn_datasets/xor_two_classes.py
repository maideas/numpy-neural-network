
import numpy as np
from npnn_datasets import DataSet

class XORTwoClasses(DataSet):

    def __init__(self, points=400, train_fraction=0.7):
        super(XORTwoClasses, self).__init__()

        self.x_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.c_data = np.array([0, 1, 1, 0])

        self.x_data = np.repeat(self.x_data, points/4, axis=0)
        self.c_data = np.repeat(self.c_data, points/4, axis=0)

        self.x_data += np.random.normal(0.0, 0.05, self.x_data.shape)

        # conversion of class labels into one-hot y vectors ...
        a = []
        for c in self.c_data:
            a.append([1, 0] if c < 0.5 else [0, 1])

        self.y_data = np.array(a)

        self.gaussian_norm_x()
        self.prepare(train_fraction)

