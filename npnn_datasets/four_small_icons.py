
import numpy as np
from npnn_datasets import DataSet

class FourSmallIcons(DataSet):

    def __init__(self, points=32*4, train_fraction=0.7):
        super(FourSmallIcons, self).__init__()

        self.x_data = np.zeros((points, 3, 3, 1))
        self.y_data = np.zeros((points, 4))
        self.c_data = np.zeros(points)

        k = 0
        for _ in np.arange(32):

            # 1 1 1
            # 0 1 0
            # 0 1 0
            self.x_data[k, 0, 0, 0] = 255
            self.x_data[k, 1, 0, 0] = 255
            self.x_data[k, 2, 0, 0] = 255
            self.x_data[k, 0, 1, 0] = 0
            self.x_data[k, 1, 1, 0] = 255
            self.x_data[k, 2, 1, 0] = 0
            self.x_data[k, 0, 2, 0] = 0
            self.x_data[k, 1, 2, 0] = 255
            self.x_data[k, 2, 2, 0] = 0
            self.y_data[k, 0] = 1
            self.c_data[k] = 0
            k += 1

            # 0 0 1
            # 1 1 1
            # 0 0 1
            self.x_data[k, 0, 0, 0] = 0
            self.x_data[k, 1, 0, 0] = 0
            self.x_data[k, 2, 0, 0] = 255
            self.x_data[k, 0, 1, 0] = 255
            self.x_data[k, 1, 1, 0] = 255
            self.x_data[k, 2, 1, 0] = 255
            self.x_data[k, 0, 2, 0] = 0
            self.x_data[k, 1, 2, 0] = 0
            self.x_data[k, 2, 2, 0] = 255
            self.y_data[k, 1] = 1
            self.c_data[k] = 1
            k += 1

            # 0 1 0
            # 0 1 0
            # 1 1 1
            self.x_data[k, 0, 0, 0] = 0
            self.x_data[k, 1, 0, 0] = 255
            self.x_data[k, 2, 0, 0] = 0
            self.x_data[k, 0, 1, 0] = 0
            self.x_data[k, 1, 1, 0] = 255
            self.x_data[k, 2, 1, 0] = 0
            self.x_data[k, 0, 2, 0] = 255
            self.x_data[k, 1, 2, 0] = 255
            self.x_data[k, 2, 2, 0] = 255
            self.y_data[k, 2] = 1
            self.c_data[k] = 2
            k += 1

            # 1 0 0
            # 1 1 1
            # 1 0 0
            self.x_data[k, 0, 0, 0] = 255
            self.x_data[k, 1, 0, 0] = 0
            self.x_data[k, 2, 0, 0] = 0
            self.x_data[k, 0, 1, 0] = 255
            self.x_data[k, 1, 1, 0] = 255
            self.x_data[k, 2, 1, 0] = 255
            self.x_data[k, 0, 2, 0] = 255
            self.x_data[k, 1, 2, 0] = 0
            self.x_data[k, 2, 2, 0] = 0
            self.y_data[k, 3] = 1
            self.c_data[k] = 3
            k += 1

        self.image_norm_x()
        self.prepare(train_fraction)

