
import numpy as np
from npnn_datasets import DataSet

class FourImgClasses(DataSet):

    def __init__(self, points=8*8*4, train_fraction=0.7):
        super(FourImgClasses, self).__init__()

        self.x_data = np.zeros((points, 10, 10, 1))
        self.y_data = np.zeros((points, 4))
        
        k = 0
        
        # 0 1 1
        # 1 0 1
        # 1 1 0
        for n in np.arange(8):
            for m in np.arange(8):
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 1
                self.x_data[k, n+2, m+0, 0] = 1
                self.x_data[k, n+0, m+1, 0] = 1
                self.x_data[k, n+1, m+1, 0] = 0
                self.x_data[k, n+2, m+1, 0] = 1
                self.x_data[k, n+0, m+2, 0] = 1
                self.x_data[k, n+1, m+2, 0] = 1
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 0] = 1
                k += 1
        
        # 0 1 0
        # 1 1 1
        # 0 1 0
        for n in np.arange(8):
            for m in np.arange(8):
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 1
                self.x_data[k, n+2, m+0, 0] = 0
                self.x_data[k, n+0, m+1, 0] = 1
                self.x_data[k, n+1, m+1, 0] = 1
                self.x_data[k, n+2, m+1, 0] = 1
                self.x_data[k, n+0, m+2, 0] = 0
                self.x_data[k, n+1, m+2, 0] = 1
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 1] = 1
                k += 1
        
        # 1 0 1
        # 0 1 0
        # 1 0 1
        for n in np.arange(8):
            for m in np.arange(8):
                self.x_data[k, n+0, m+0, 0] = 1
                self.x_data[k, n+1, m+0, 0] = 0
                self.x_data[k, n+2, m+0, 0] = 1
                self.x_data[k, n+0, m+1, 0] = 0
                self.x_data[k, n+1, m+1, 0] = 1
                self.x_data[k, n+2, m+1, 0] = 0
                self.x_data[k, n+0, m+2, 0] = 1
                self.x_data[k, n+1, m+2, 0] = 0
                self.x_data[k, n+2, m+2, 0] = 1
                self.y_data[k, 2] = 1
                k += 1
        
        # 0 1 0
        # 1 0 1
        # 0 1 0
        for n in np.arange(8):
            for m in np.arange(8):
                self.x_data[k, n+0, m+0, 0] = 0
                self.x_data[k, n+1, m+0, 0] = 1
                self.x_data[k, n+2, m+0, 0] = 0
                self.x_data[k, n+0, m+1, 0] = 1
                self.x_data[k, n+1, m+1, 0] = 0
                self.x_data[k, n+2, m+1, 0] = 1
                self.x_data[k, n+0, m+2, 0] = 0
                self.x_data[k, n+1, m+2, 0] = 1
                self.x_data[k, n+2, m+2, 0] = 0
                self.y_data[k, 3] = 1
                k += 1

        self.prepare(train_fraction, normalize_x=True, normalize_y=False)

