
if 'CUDA' in globals() or 'CUDA' in locals():
    import cupy as np
else:
    import numpy as np

#from profilehooks import profile

class Optimizer:
    '''network model weights optimizer base class'''

    def __init__(self, model,
                 alpha=1e-3,
                 gamma=0.0,
                 beta1=0.9,
                 beta2=0.999):
        '''
        model : network model object
        alpha : learning rate
        gamma : weight regularization
        beta1 : Adam beta1 / weight adaption momentum
        beta2 : Adam beta2 / RMSprop beta2
        '''
        self.model = model
        self.dataset = None

        self.alpha = alpha
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2

        self.steps = 1
        self.loss = np.zeros(self.model.loss_layer.shape_in)
        self.accuracy = 0.0

        self.train_x_batch = np.array([])
        self.train_t_batch = np.array([])
        self.train_y_batch = np.array([])

        self.init()

    def init(self):
        '''initializes optimizer specific model layer members'''
        pass

    def zero_grad(self):
        '''set all gradient values to zero (preparation for incremental gradient calculation)'''
        for layer in self.model.layers:
            layer.zero_grad()

    def weight_decay(self, w):
        '''L2 norm of w (scalar value)'''
        return self.gamma * np.sum(np.square(w)) / w.size()

    def update_weights(self, layer):
        '''
        the update_weights method may be overwritten by some more advanced algorithm
        default algorithm : vanilla gradient descent
        '''
        layer.w += -self.alpha * layer.grad_w
        layer.w += self.weight_decay(layer.w)

    #@profile
    def step(self, batch_size=None):
        '''
        stochastic mini batch optimization step
        '''

        # x_batch : network model input data batch
        # t_batch : related network model target data batch
        train_x_batch, train_t_batch = self.dataset.get_train_batch(batch_size)

        # normalize network (input, output) training data ...
        self.train_x_batch = self.dataset.normalize(train_x_batch, self.dataset.x_mean, self.dataset.x_variance)
        self.train_t_batch = self.dataset.normalize(train_t_batch, self.dataset.y_mean, self.dataset.y_variance)
        self.train_y_batch = np.zeros(self.train_t_batch.shape)

        # initialize gradients to zero ...
        self.zero_grad()

        # switch layers to training state ...
        for layer in self.model.layers:
            layer.step_init(is_training=True)

        self.loss = np.zeros(self.model.loss_layer.shape_in)
        self.accuracy = 0.0

        # pass mini batch data through the net ...
        for n in np.arange(self.train_x_batch.shape[0]):
            x = self.train_x_batch[n]
            t = self.train_t_batch[n]

            # forward pass through all layers ...
            for layer in self.model.layers:
                x = layer.forward(x)
            self.train_y_batch[n] = x

            # loss calculation which gives a gradient ...
            self.loss += self.model.loss_layer.forward(x, t)
            grad = self.model.loss_layer.backward()

            self.accuracy += self.accuracy_increment(x, t)

            # backward pass through all layers ...
            for layer in self.model.layers[::-1]:
                grad = layer.backward(grad)

        # calculate mini batch loss ...
        self.loss /= self.train_x_batch.shape[0]
        self.accuracy = 100.0 * self.accuracy / self.train_x_batch.shape[0]  # percentage value

        # adjust the weights ...
        for layer in self.model.layers:
            layer.update_weights(self.update_weights)

        # switch layers back to non-training state ...
        for layer in self.model.layers:
            layer.step_init(is_training=False)

        self.steps += 1

    #@profile
    def predict(self, x_batch_in, t_batch_in=None):
        '''
        network model forward path calculation (prediction) of a given x batch
        x_batch : network model input data
        t_batch : optional target data, which can be used for loss and accuracy calculation
        returns : network model output data
        '''
        # normalize network input data ...
        x_batch = self.dataset.normalize(x_batch_in, self.dataset.x_mean, self.dataset.x_variance)
        y_batch = []

        if t_batch_in is None:
            for x in x_batch:
                for layer in self.model.layers:
                    x = layer.forward(x)
                y_batch.append(x)

        else:
            # normalize target data ...
            t_batch = self.dataset.normalize(t_batch_in, self.dataset.y_mean, self.dataset.y_variance)

            self.loss = np.zeros(self.model.loss_layer.shape_in)
            self.accuracy = 0.0

            for x, t in zip(x_batch, t_batch):
                for layer in self.model.layers:
                    x = layer.forward(x)
                y_batch.append(x)
                
                self.loss += self.model.loss_layer.forward(x, t)
                self.accuracy += self.accuracy_increment(x, t)
         
            self.loss /= x_batch.shape[0]
            self.accuracy = 100.0 * self.accuracy / x_batch.shape[0]  # percentage value

        # denormalize network output data ...
        return self.dataset.denormalize(np.array(y_batch), self.dataset.y_mean, self.dataset.y_variance)

    def predict_class(self, x_batch):
        '''
        predict class batch related to input batch
        x_batch : network model input data
        returns : class integer batch
        '''
        y_batch = self.predict(x_batch)
        c_batch = []
        for y in y_batch:
            c_batch.append(np.argmax(y))
        return np.array(c_batch)

    def accuracy_increment(self, x, t):
        if self.model.loss_layer.__class__.__name__ == "CrossEntropyLoss":
            # softmax + cross entropy loss accuracy ...
            if np.argmax(x) == np.argmax(t):
                return 1.0
        if self.model.loss_layer.__class__.__name__ == "BinaryCrossEntropyLoss":
            # sigmoid + binary cross entropy accuracy ...
            return np.array((x > 0.5) == (t > 0.5)).astype(int) / t.shape[0]
        return 0.0


class SGD(Optimizer):
    '''stochastic gradient descent optimizer with weight momentum'''

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        if layer.prev_dw is None:
            layer.prev_dw = np.zeros(layer.w.shape)

        dw = -self.alpha * layer.grad_w
        layer.w += (1.0 - self.beta1) * dw + self.beta1 * layer.prev_dw
        layer.prev_dw = dw


class RMSprop(Optimizer):
    '''Geoffrey Hinton's unpublished RMSprop optimizer'''

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        if layer.ma_grad2 is None:
            layer.ma_grad2 = np.zeros(layer.w.shape)

        # squared gradient moving average ...
        layer.ma_grad2 = self.beta2 * layer.ma_grad2 + (1.0 - self.beta2) * np.square(layer.grad_w)

        # weight adjustment ...
        layer.w += -self.alpha * np.divide(layer.grad_w, (np.sqrt(layer.ma_grad2) + 1e-9))


class Adam(Optimizer):

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        if layer.ma_grad1 is None:
            layer.ma_grad1 = np.zeros(layer.w.shape)
            layer.ma_grad2 = np.zeros(layer.w.shape)

        # normal and squared gradient moving average ...
        layer.ma_grad1 = self.beta1 * layer.ma_grad1 + (1.0 - self.beta1) *           layer.grad_w
        layer.ma_grad2 = self.beta2 * layer.ma_grad2 + (1.0 - self.beta2) * np.square(layer.grad_w)

        # bias correction (boot strapping) ...
        ma_grad1 = layer.ma_grad1 / (1.0 - np.power(self.beta1, self.steps))
        ma_grad2 = layer.ma_grad2 / (1.0 - np.power(self.beta2, self.steps))

        # weight adjustment ...
        layer.w += -self.alpha * np.divide(ma_grad1, (np.sqrt(ma_grad2) + 1e-9))

