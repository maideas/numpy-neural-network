
import numpy as np

class DataSet:

    def __init__(self):
        # zero mean, unit variance
        self.x_mean = 0.0
        self.x_variance = 1.0
        self.y_mean = 0.0
        self.y_variance = 1.0

        # x and y data have to be initilized by derived class
        self.x_data = np.array([])
        self.y_data = np.array([])

    def prepare(self, train_fraction=0.7, normalize_x=True, normalize_y=True):

        # calculate data mean and variance (if requested) ...
        if normalize_x:
            self.x_mean, self.x_variance = self.get_mean_and_variance(self.x_data)
        if normalize_y:
            self.y_mean, self.y_variance = self.get_mean_and_variance(self.y_data)

        # split data into train and validation data ...
        self.num_train_data = int(self.x_data.shape[0] * train_fraction)
        self.num_validation_data = self.x_data.shape[0] - self.num_train_data

        # shuffle indices before train/validation split ...
        idx = np.random.permutation(self.x_data.shape[0])

        # split indices into train and validation ...
        idx_train = idx[:self.num_train_data]
        idx_validation = idx[self.num_train_data:]

        self.x_train_data = self.x_data[idx_train]
        self.y_train_data = self.y_data[idx_train]
        self.x_validation_data = self.x_data[idx_validation]
        self.y_validation_data = self.y_data[idx_validation]

        # to get a good stochastic behavior, the mini batch size
        # shall be smaller than the complete data batch size: for this
        # reason we initialize its size to the square root of train data size ...
        self.train_batch_size = int(np.trunc(np.sqrt(self.num_train_data)))

        # initialize validation mini-batch size to complete validation data size ...
        self.validation_batch_size = self.num_validation_data

    def get_train_batch(self, batch_size=None):
        if not batch_size:
            batch_size = self.train_batch_size

        idx = np.random.randint(self.num_train_data, size=batch_size)
        return (self.x_train_data[idx], self.y_train_data[idx])

    def get_validation_batch(self, batch_size=None):
        if not batch_size:
            batch_size = self.validation_batch_size

        idx = np.random.randint(self.num_validation_data, size=batch_size)
        return self.x_validation_data[idx], self.y_validation_data[idx]

    def get_mean_and_variance(self, data):
        return np.mean(data), np.std(data)

    def normalize(self, data, mean, variance):
        return (data - mean) / variance

    def denormalize(self, data, mean, variance):
        return (data * variance) + mean

