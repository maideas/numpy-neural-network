
import numpy as np
from npnn_datasets import DataSet

class NoisySine(DataSet):

    def __init__(self, points=1000, train_fraction=0.7):
        super(NoisySine, self).__init__()

        x_min = 0.2
        x_max = 2.0 * np.pi - 0.2
        x_step = (x_max - x_min) / points

        self.x_data = np.arange(x_min, x_max, x_step)
        self.y_data = np.sin(self.x_data) + np.random.normal(0.0, 0.1, self.x_data.shape)

        self.x_data = self.x_data.reshape(-1, 1)
        self.y_data = self.y_data.reshape(-1, 1)

        self.prepare(train_fraction, normalize_x=True, normalize_y=True)

