
import numpy as np
from npnn_datasets import DataSet

class FourSteppedIcons(DataSet):

    def __init__(self, points=6*6*4, train_fraction=0.7):
        super(FourSteppedIcons, self).__init__()

        self.x_data = np.zeros((points, 8, 8, 1))
        self.y_data = np.zeros((points, 4))
        self.c_data = np.zeros(points)

        k = 0

        for n in np.arange(6):
            for m in np.arange(6):
                # 1 1 1
                # 0 1 0
                # 0 1 0
                self.x_data[k, n+0, m+0, 0] = 255
                self.x_data[k, n+1, m+0, 0] = 255
                self.x_data[k, n+2, m+0, 0] = 255
                self.x_data[k, n+0, m+1, 0] = 0
                self.x_data[k, n+1, m+1, 0] = 255
                self.x_data[k, n+2, m+1, 0] = 0
                self.x_data[k, n+0, m+2, 0] = 0
                self.x_data[k, n+1, m+2, 0] = 255
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 0] = 1
                self.c_data[k] = 0
                k += 1

        for n in np.arange(6):
            for m in np.arange(6):
                # 0 0 1
                # 1 1 1
                # 0 0 1
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 0
                self.x_data[k, n+2, m+0, 0] = 255
                self.x_data[k, n+0, m+1, 0] = 255
                self.x_data[k, n+1, m+1, 0] = 255
                self.x_data[k, n+2, m+1, 0] = 255
                self.x_data[k, n+0, m+2, 0] = 0
                self.x_data[k, n+1, m+2, 0] = 0
                self.x_data[k, n+2, m+2, 0] = 255
                self.y_data[k, 1] = 1
                self.c_data[k] = 1
                k += 1

        for n in np.arange(6):
            for m in np.arange(6):
                # 0 1 0
                # 0 1 0
                # 1 1 1
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 255
                self.x_data[k, n+2, m+0, 0] = 0
                self.x_data[k, n+0, m+1, 0] = 0
                self.x_data[k, n+1, m+1, 0] = 255
                self.x_data[k, n+2, m+1, 0] = 0
                self.x_data[k, n+0, m+2, 0] = 255
                self.x_data[k, n+1, m+2, 0] = 255
                self.x_data[k, n+2, m+2, 0] = 255
                self.y_data[k, 2] = 1
                self.c_data[k] = 2
                k += 1

        for n in np.arange(6):
            for m in np.arange(6):
                # 1 0 0
                # 1 1 1
                # 1 0 0
                self.x_data[k, n+0, m+0, 0] = 255
                self.x_data[k, n+1, m+0, 0] = 0
                self.x_data[k, n+2, m+0, 0] = 0
                self.x_data[k, n+0, m+1, 0] = 255
                self.x_data[k, n+1, m+1, 0] = 255
                self.x_data[k, n+2, m+1, 0] = 255
                self.x_data[k, n+0, m+2, 0] = 255
                self.x_data[k, n+1, m+2, 0] = 0
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 3] = 1
                self.c_data[k] = 3
                k += 1

        self.image_norm_x()
        self.prepare(train_fraction)

