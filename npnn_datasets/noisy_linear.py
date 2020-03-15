
import numpy as np
from npnn_datasets import DataSet

class NoisyLinear(DataSet):

    def __init__(self, points=1000, train_fraction=0.7):
        super(NoisyLinear, self).__init__()

        x_min = 1.0
        x_max = 3.0
        x_step = (x_max - x_min) / points

        self.x_data = np.arange(x_min, x_max, x_step)
        self.y_data = 4.0 * self.x_data - 2.0 + np.random.normal(0.0, 0.1, self.x_data.shape)

        self.x_data = self.x_data.reshape(-1, 1)
        self.y_data = self.y_data.reshape(-1, 1)

        self.prepare(train_fraction, normalize_x=True, normalize_y=True)

