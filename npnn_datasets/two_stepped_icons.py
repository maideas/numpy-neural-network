
import numpy as np
from npnn_datasets import DataSet

class TwoSteppedIcons(DataSet):

    def __init__(self, points=2*2*2, train_fraction=0.7):
        super(TwoSteppedIcons, self).__init__()

        self.x_data = np.zeros((points, 4, 4, 1))
        self.y_data = np.zeros((points, 2))
        self.c_data = np.zeros(points)

        k = 0

        for n in np.arange(2):
            for m in np.arange(2):
                # 0 1 1
                # 0 0 1
                # 0 0 0
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 255
                self.x_data[k, n+2, m+0, 0] = 255
                self.x_data[k, n+0, m+1, 0] = 0
                self.x_data[k, n+1, m+1, 0] = 0
                self.x_data[k, n+2, m+1, 0] = 255
                self.x_data[k, n+0, m+2, 0] = 0
                self.x_data[k, n+1, m+2, 0] = 0
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 0] = 1
                self.c_data[k] = 0
                k += 1

        for n in np.arange(2):
            for m in np.arange(2):
                # 0 0 0
                # 1 0 0
                # 1 1 0
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 0
                self.x_data[k, n+2, m+0, 0] = 0
                self.x_data[k, n+0, m+1, 0] = 255
                self.x_data[k, n+1, m+1, 0] = 0
                self.x_data[k, n+2, m+1, 0] = 0
                self.x_data[k, n+0, m+2, 0] = 255
                self.x_data[k, n+1, m+2, 0] = 255
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 1] = 1
                self.c_data[k] = 1
                k += 1

        self.image_norm_x()

        self.num_train_data      = 4
        self.num_validation_data = 4

        idx_train      = [0, 3, 4, 7]
        idx_validation = [1, 2, 5, 6]

        self.x_train_data = self.x_data[idx_train]
        self.y_train_data = self.y_data[idx_train]
        self.c_train_data = self.c_data[idx_train]

        self.x_validation_data = self.x_data[idx_validation]
        self.y_validation_data = self.y_data[idx_validation]
        self.c_validation_data = self.c_data[idx_validation]

        self.train_batch_size      = self.num_train_data
        self.validation_batch_size = self.num_validation_data

