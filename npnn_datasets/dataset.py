
import numpy as np

class DataSet:

    def __init__(self):
        # zero mean, unit variance
        self.x_mean     = np.array([0.0])
        self.x_variance = np.array([1.0])
        self.y_mean     = np.array([0.0])
        self.y_variance = np.array([1.0])

        self.norm = {}
        self.norm['x_mean']      = self.x_mean
        self.norm['x_variance']  = self.x_variance
        self.norm['y_mean']      = self.y_mean
        self.norm['y_variance']  = self.y_variance
        self.norm['normalize']   = self.normalize
        self.norm['denormalize'] = self.denormalize

        # x and y data have to be initialized by derived class
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.c_data = None


    def gaussian_norm_x(self):
        '''map data to centered gaussian distribution'''
        self.x_mean, self.x_variance = self.get_mean_and_variance(self.x_data)
        self.norm['x_mean']     = self.x_mean
        self.norm['x_variance'] = self.x_variance

    def gaussian_norm_y(self):
        '''map data to centered gaussian distribution'''
        self.y_mean, self.y_variance = self.get_mean_and_variance(self.y_data)
        self.norm['y_mean']     = self.y_mean
        self.norm['y_variance'] = self.y_variance

    def image_norm_x(self):
        '''map image data [0 ... 255] to centered range [-1.0 ... 1.0]'''
        self.norm['x_mean']     = np.full(self.x_data[0].shape, 128.0)
        self.norm['x_variance'] = np.full(self.x_data[0].shape, 128.0)

    def image_norm_y(self):
        '''map image data [0 ... 255] to centered range [-1.0 ... 1.0]'''
        self.norm['y_mean']     = np.full(self.y_data[0].shape, 128.0)
        self.norm['y_variance'] = np.full(self.y_data[0].shape, 128.0)

    def image_norm_pos_x(self):
        '''map image data [0 ... 255] to positive range [0.0 ... 1.0]'''
        self.norm['x_mean']     = np.full(self.x_data[0].shape, 0.0)
        self.norm['x_variance'] = np.full(self.x_data[0].shape, 256.0)

    def image_norm_pos_y(self):
        '''map image data [0 ... 255] to positive range [0.0 ... 1.0]'''
        self.norm['y_mean']     = np.full(self.y_data[0].shape, 0.0)
        self.norm['y_variance'] = np.full(self.y_data[0].shape, 256.0)


    def prepare(self, train_fraction=0.7):

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

        if self.c_data is None:
            self.c_train_data = np.zeros(self.num_train_data)
            self.c_validation_data = np.zeros(self.num_validation_data)
        else:
            self.c_train_data = self.c_data[idx_train]
            self.c_validation_data = self.c_data[idx_validation]

        # to get a good stochastic behavior, the mini batch size
        # shall be smaller than the complete data batch size: for this
        # reason we initialize its size to the square root of train data size ...
        self.train_batch_size = int(np.trunc(np.sqrt(self.num_train_data)))

        # initialize validation mini-batch size minimum of complete validation
        # data and training batch size ...
        self.validation_batch_size = min(self.num_validation_data, self.train_batch_size)


    def get_train_batch(self, batch_size=None):
        if not batch_size:
            batch_size = self.train_batch_size

        idx = np.random.randint(self.num_train_data, size=batch_size)
        return self.x_train_data[idx], self.y_train_data[idx], self.c_train_data[idx]

    def get_validation_batch(self, batch_size=None):
        if not batch_size:
            batch_size = self.validation_batch_size

        idx = np.random.randint(self.num_validation_data, size=batch_size)
        return self.x_validation_data[idx], self.y_validation_data[idx], self.c_validation_data[idx]


    def get_mean_and_variance(self, data):
        # per-feature mean and variance over all data vectors ...
        # ("feature" means data vector element)
        mean = np.mean(data, axis=0)
        variance = np.std(data, axis=0)
        variance[np.square(variance) < 1e-6] = 1.0
        return mean, variance

    def normalize(self, data, mean, variance):
        return (data - mean) / variance

    def denormalize(self, data, mean, variance):
        return (data * variance) + mean


    def print_data_element(self, element):

        if len(element.shape) == 1:
            print("{} ".format(element))

        if len(element.shape) == 2:
            for s1 in np.arange(element.shape[0]):
                print("{} ".format(element[s1]), end='')
            print("")

        if len(element.shape) > 2:
            for s1 in np.arange(element.shape[0]):
                for s2 in np.arange(element.shape[1]):
                    print("{} ".format(element[s1, s2]), end='')
                print("")

    def print_data(self, x, y=None):

        for n in np.arange(x.shape[0]):
            print("== #{} x =======================================".format(n))
            self.print_data_element(x[n])

            if y is not None:
                print("-- #{} y --".format(n))
                self.print_data_element(y[n])

