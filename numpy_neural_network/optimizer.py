
import numpy as np

#from profilehooks import profile

class Optimizer:
    '''network model weights optimizer base class'''

    def __init__(self,
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
        self.model = None
        self.chain = None
        self.norm = None

        self.alpha = alpha
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2

        self.steps = 1

        self.train_x_batch = np.array([])
        self.train_t_batch = np.array([])
        self.train_y_batch = np.array([])


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

        layer.wb += -self.alpha * layer.grad_wb
        layer.wb += self.weight_decay(layer.wb)


    #@profile
    def step(self, x_batch=None, t_batch=None, c_batch=None):
        '''
        stochastic mini batch optimization step
        '''
        self.train_x_batch = x_batch
        self.train_t_batch = t_batch
        self.train_c_batch = c_batch

        if self.norm is not None:
            # normalize network (input, output) training data ...
            self.train_x_batch = self.norm['normalize'](self.train_x_batch, self.norm['x_mean'], self.norm['x_variance'])
            self.train_t_batch = self.norm['normalize'](self.train_t_batch, self.norm['y_mean'], self.norm['y_variance'])

        # zero gradients and clear loss/accuracy ...
        self.model.zero_grad()

        # switch layers to training state ...
        self.model.step_init(is_training=True)

        # pass mini batch data through the net ...
        g_batch = []
        y_batch = []
        for x, t in zip(self.train_x_batch, self.train_t_batch):
            g, x = self.model.step(x, t)
            g_batch.append(g)
            y_batch.append(x)

        g_batch = np.array(g_batch)
        y_batch = np.array(y_batch)

        # adjust the weights ...
        self.model.update_weights(self.update_weights)

        # switch layers back to non-training state ...
        self.model.step_init(is_training=False)

        if self.norm is not None:
            self.train_y_batch = self.norm['denormalize'](self.train_y_batch, self.norm['y_mean'], self.norm['y_variance'])

        self.steps += 1
        return g_batch, self.train_y_batch


    #@profile
    def predict(self, x_batch, t_batch=None, c_batch=None):
        '''
        network model forward path calculation (prediction) of a given x batch
        x_batch : network model input data
        t_batch : optional target data, which can be used for loss and accuracy calculation
        returns : network model output data
        '''
        # zero gradients and clear loss/accuracy ...
        self.model.zero_grad()

        # switch layers to non-training state and call layer step_init
        self.model.step_init(is_training=False)

        if self.norm is not None:
            # normalize network input data ...
            x_batch = self.norm['normalize'](x_batch, self.norm['x_mean'], self.norm['x_variance'])

        if t_batch is None:
            y_batch = []
            for x in x_batch:
                x = self.model.predict(x)
                y_batch.append(x)

        else:
            # normalize target data ...
            t_batch = self.norm['normalize'](t_batch, self.norm['y_mean'], self.norm['y_variance'])

            y_batch = []
            for x, t in zip(x_batch, t_batch):
                x = self.model.predict(x, t)
                y_batch.append(x)

        y_batch = np.array(y_batch)

        # denormalize network output data ...
        if self.norm is not None:
            return self.norm['denormalize'](y_batch, self.norm['y_mean'], self.norm['y_variance'])
        else:
            return y_batch


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


class SGD(Optimizer):
    '''stochastic gradient descent optimizer with weight momentum'''

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        if layer.prev_d_w is None:
            layer.prev_d_w = np.zeros(layer.w.shape)

        d_w = -self.alpha * layer.grad_w
        layer.w += (1.0 - self.beta1) * d_w + self.beta1 * layer.prev_d_w
        layer.prev_d_w = d_w

        if layer.prev_d_wb is None:
            layer.prev_d_wb = np.zeros(layer.wb.shape)

        d_wb = -self.alpha * layer.grad_wb
        layer.wb += (1.0 - self.beta1) * d_wb + self.beta1 * layer.prev_d_wb
        layer.prev_d_wb = d_wb


class RMSprop(Optimizer):
    '''Geoffrey Hinton's unpublished RMSprop optimizer'''

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        if layer.ma_grad2_w is None:
            layer.ma_grad2_w = np.zeros(layer.w.shape)

        # squared gradient moving average ...
        layer.ma_grad2_w = self.beta2 * layer.ma_grad2_w + (1.0 - self.beta2) * np.square(layer.grad_w)

        # weight adjustment ...
        layer.w += -self.alpha * np.divide(layer.grad_w, (np.sqrt(layer.ma_grad2_w) + 1e-9))

        if layer.ma_grad2_wb is None:
            layer.ma_grad2_wb = np.zeros(layer.wb.shape)

        # squared gradient moving average ...
        layer.ma_grad2_wb = self.beta2 * layer.ma_grad2_wb + (1.0 - self.beta2) * np.square(layer.grad_wb)

        # weight adjustment ...
        layer.wb += -self.alpha * np.divide(layer.grad_wb, (np.sqrt(layer.ma_grad2_wb) + 1e-9))


class Adam(Optimizer):

    def update_weights(self, layer):
        '''
        implements the optimizer dependent layer weight update algorithm
        layer : network model layer to be adapted
        '''
        if layer.ma_grad1_w is None:
            layer.ma_grad1_w = np.zeros(layer.w.shape)
            layer.ma_grad2_w = np.zeros(layer.w.shape)

        # normal and squared gradient moving average ...
        layer.ma_grad1_w = self.beta1 * layer.ma_grad1_w + (1.0 - self.beta1) *           layer.grad_w
        layer.ma_grad2_w = self.beta2 * layer.ma_grad2_w + (1.0 - self.beta2) * np.square(layer.grad_w)

        # bias correction (boot strapping) ...
        ma_grad1_w = layer.ma_grad1_w / (1.0 - np.power(self.beta1, self.steps))
        ma_grad2_w = layer.ma_grad2_w / (1.0 - np.power(self.beta2, self.steps))

        # weight adjustment ...
        layer.w += -self.alpha * np.divide(ma_grad1_w, (np.sqrt(ma_grad2_w) + 1e-9))

        if layer.ma_grad1_wb is None:
            layer.ma_grad1_wb = np.zeros(layer.wb.shape)
            layer.ma_grad2_wb = np.zeros(layer.wb.shape)

        # normal and squared gradient moving average ...
        layer.ma_grad1_wb = self.beta1 * layer.ma_grad1_wb + (1.0 - self.beta1) *           layer.grad_wb
        layer.ma_grad2_wb = self.beta2 * layer.ma_grad2_wb + (1.0 - self.beta2) * np.square(layer.grad_wb)

        # bias correction (boot strapping) ...
        ma_grad1_wb = layer.ma_grad1_wb / (1.0 - np.power(self.beta1, self.steps))
        ma_grad2_wb = layer.ma_grad2_wb / (1.0 - np.power(self.beta2, self.steps))

        # weight adjustment ...
        layer.wb += -self.alpha * np.divide(ma_grad1_wb, (np.sqrt(ma_grad2_wb) + 1e-9))

