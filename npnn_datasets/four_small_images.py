
import numpy as np
from npnn_datasets import DataSet

class FourSmallImages(DataSet):

    def __init__(self, points=16*4, train_fraction=0.7):
        super(FourSmallImages, self).__init__()

        self.x_data = np.zeros((points, 3, 3, 1))
        self.y_data = np.zeros((points, 3, 3, 1))
        self.c_data = np.zeros(points)

        k = 0
        for _ in np.arange(16):

            # 1 1 1
            # 1 0 1
            # 1 1 1
            self.x_data[k, 0, 0, 0] = 1
            self.x_data[k, 1, 0, 0] = 1
            self.x_data[k, 2, 0, 0] = 1
            self.x_data[k, 0, 1, 0] = 1
            self.x_data[k, 1, 1, 0] = 0
            self.x_data[k, 2, 1, 0] = 1
            self.x_data[k, 0, 2, 0] = 1
            self.x_data[k, 1, 2, 0] = 1
            self.x_data[k, 2, 2, 0] = 1
            self.c_data[k] = 0
            k += 1

            # 0 1 0
            # 1 1 1
            # 0 1 0
            self.x_data[k, 0, 0, 0] = 0
            self.x_data[k, 1, 0, 0] = 1
            self.x_data[k, 2, 0, 0] = 0
            self.x_data[k, 0, 1, 0] = 1
            self.x_data[k, 1, 1, 0] = 1
            self.x_data[k, 2, 1, 0] = 1
            self.x_data[k, 0, 2, 0] = 0
            self.x_data[k, 1, 2, 0] = 1
            self.x_data[k, 2, 2, 0] = 0
            self.c_data[k] = 1
            k += 1

            # 1 0 1
            # 0 1 0
            # 1 0 1
            self.x_data[k, 0, 0, 0] = 1
            self.x_data[k, 1, 0, 0] = 0
            self.x_data[k, 2, 0, 0] = 1
            self.x_data[k, 0, 1, 0] = 0
            self.x_data[k, 1, 1, 0] = 1
            self.x_data[k, 2, 1, 0] = 0
            self.x_data[k, 0, 2, 0] = 1
            self.x_data[k, 1, 2, 0] = 0
            self.x_data[k, 2, 2, 0] = 1
            self.c_data[k] = 2
            k += 1

            # 0 1 0
            # 1 0 1
            # 0 1 0
            self.x_data[k, 0, 0, 0] = 0
            self.x_data[k, 1, 0, 0] = 1
            self.x_data[k, 2, 0, 0] = 0
            self.x_data[k, 0, 1, 0] = 1
            self.x_data[k, 1, 1, 0] = 0
            self.x_data[k, 2, 1, 0] = 1
            self.x_data[k, 0, 2, 0] = 0
            self.x_data[k, 1, 2, 0] = 1
            self.x_data[k, 2, 2, 0] = 0
            self.c_data[k] = 3
            k += 1

        self.x_data += np.random.normal(0.0, 0.1, self.x_data.shape)
        self.y_data = self.x_data.copy()

        self.prepare(train_fraction, normalize_x=True, normalize_y=True)

