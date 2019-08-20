
import numpy as np

class Optimizer:
    '''network model weights optimizer base class'''
    def __init__(self, model, mini_batch_size=50,
                 alpha=1e-3,
                 gamma=0.0,
                 beta1=0.9,
                 beta2=0.999):
        '''
        model : network model object
        mini_batch_size : number of data vectors used to do a network weight update
        alpha : learning rate
        gamma : weight regularization
        beta1 : Adam beta1 / weight adaption momentum
        beta2 : Adam beta2 / RMSprop beta2
        '''
        self.model = model
        self.mini_batch_size = mini_batch_size

        self.alpha = alpha
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2

        self.steps = 1
        self.loss = 0.0
        self.init()

    def init(self):
        '''initializes optimizer specific model layer members'''
        pass

    def zero_grad(self):
        '''set all gradient values to zero (preparation for incremental gradient calculation)'''
        for layer in self.model.layers:
            layer.zero_grad()

    def weight_penalty(self, w):
        '''L2 norm of w (scalar value)'''
        return self.gamma * np.sum(np.square(w)) / w.size

    def update_weights(self, layer):
        '''
        the update_weights method may be overwritten by some more advanced algorithm
        default algorithm : vanilla gradient descent
        '''
        layer.w += -self.alpha * layer.grad_w
        #print("weights = {0}".format(layer.w))

    def step(self, x_batch, target_batch):
        '''
        stochastic mini batch optimization step
        x_batch      : network model input data batch
        target_batch : related network model target data batch
        '''
        batch_size = len(x_batch)

        # fix mini_batch_size if needed (for small batch sizes) ...
        if self.mini_batch_size > batch_size:
            self.mini_batch_size = batch_size

        # sample mini_batch from given batch data ...
        idx = np.random.randint(batch_size, size=self.mini_batch_size)
        x_mini_batch = x_batch[idx]
        target_mini_batch = target_batch[idx]

        # initialize gradients to zero ...
        self.zero_grad()

        # pass mini batch data through the net ...
        self.loss = 0.0
        for n in np.arange(self.mini_batch_size):
            x = x_mini_batch[n]
            target = target_mini_batch[n]

            # forward pass through all layers ...
            for layer in self.model.layers:
                x = layer.forward(x)
                #print("activation = {0}".format(x))

            # loss calculation which gives a gradient ...
            self.loss += self.model.loss_layer.forward(x, target)
            grad = self.model.loss_layer.backward()

            # backward pass through all layers ...
            for layer in self.model.layers[::-1]:
                #print("gradient = {0}".format(grad))
                grad = layer.backward(grad)

        # calculate mini batch loss ...
        self.loss /= self.mini_batch_size

        # adjust the weights ...
        for layer in self.model.layers:
            if layer.w is not None:
                self.update_weights(layer)
                layer.w += self.weight_penalty(layer.w)
                #print("weights = {0}".format(layer.w))

        self.steps += 1


class SGD(Optimizer):
    '''stochastic gradient descent optimizer with weight momentum'''

    def init(self):
        '''implements optimizer dependent initialization things'''
        for layer in self.model.layers:
            if layer.w is not None:
                layer.prev_dw = np.zeros((layer.size_out, layer.size_in))

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        dw = -self.alpha * layer.grad_w
        layer.w += (1.0 - self.beta1) * dw + self.beta1 * layer.prev_dw
        layer.prev_dw = dw


class RMSprop(Optimizer):
    '''Geoffrey Hinton's unpublished RMSprop optimizer'''

    def init(self):
        '''implements optimizer dependent initialization things'''
        for layer in self.model.layers:
            if layer.w is not None:
                layer.ma_grad2 = np.zeros((layer.size_out, layer.size_in))

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        # squared gradient moving average ...
        layer.ma_grad2 = self.beta2 * layer.ma_grad2 + (1.0 - self.beta2) * np.square(layer.grad_w)

        # weight adjustment ...
        layer.w += -self.alpha * np.divide(layer.grad_w, (np.sqrt(layer.ma_grad2) + 1e-9))


class Adam(Optimizer):

    def init(self):
        '''implements optimizer dependent initialization things'''
        for layer in self.model.layers:
            if layer.w is not None:
                layer.ma_grad1 = np.zeros((layer.size_out, layer.size_in))
                layer.ma_grad2 = np.zeros((layer.size_out, layer.size_in))

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        # normal and squared gradient moving average ...
        layer.ma_grad1 = self.beta1 * layer.ma_grad1 + (1.0 - self.beta1) *           layer.grad_w
        layer.ma_grad2 = self.beta2 * layer.ma_grad2 + (1.0 - self.beta2) * np.square(layer.grad_w)

        # bias correction (boot strapping) ...
        ma_grad1 = layer.ma_grad1 / (1.0 - np.power(self.beta1, self.steps))
        ma_grad2 = layer.ma_grad2 / (1.0 - np.power(self.beta2, self.steps))

        # weight adjustment ...
        layer.w += -self.alpha * np.divide(ma_grad1, (np.sqrt(ma_grad2) + 1e-9))

