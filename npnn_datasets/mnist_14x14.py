
import os
import re
import numpy as np
from npnn_datasets import DataSet
from zipfile import ZipFile
from PIL import Image

class MNIST_14x14(DataSet):

    def __init__(self, points=None, train_fraction=0.7):
        super(MNIST_14x14, self).__init__()

        data_dir = os.path.join(os.path.dirname(__file__), 'mnist')

        zf = ZipFile("{0}/mnist_14x14.zip".format(data_dir))

        self.x_data = []
        self.y_data = []
        self.c_data = []

        for zfi in zf.infolist():
            fi = zf.open(zfi)
            img = Image.open(fi)

            m = re.search('mnist_(\d)_', fi.name)
            c = int(m.group(1))
            onehot_c = np.zeros(10)
            onehot_c[c] = 1.0

            self.x_data.append(np.array(img))
            self.y_data.append(onehot_c)
            self.c_data.append(c)

        self.x_data = np.array(self.x_data).reshape((-1, 14, 14, 1))
        self.y_data = np.array(self.y_data)
        self.c_data = np.array(self.c_data)

        num_classes = 10
        samples_per_class = 200
        self.x_data, self.y_data, self.c_data = self.get_partial_dataset(
            self.x_data, self.y_data, self.c_data, num_classes, samples_per_class
        )

        self.prepare(train_fraction, normalize_x=True, normalize_y=False)

