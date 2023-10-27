import numpy as np


# to initialize parameters
np.random.seed(1)


class Module:
    """
        base class of all modules in MLP
    """
    def __init__(self):
        """
            param: all trainable parameter
            grad:  all gradient for trainable parameters, updated when back propagation
            m, v:  used by some optimizer
            cache: restore some data for backward when forward
            is_train: in training mode or not, helpful for dropout and BN
        """
        self.param = []
        self.grad = []
        self.m = []
        self.v = []
        self.cache = None
        self.is_train = True

    def forward(self, x: np.array, grad=True) -> np.array:
        """
            calculate forward-output and restore cache if gradient is needed
        :param x:         input
        :param grad:      whether gradient is needed
        :return:          output
        """
        pass

    def backward(self, dy: np.array) -> np.array:
        """
            update self-grad and return dL / dx for next layer
        :param dy:   dL / dy
        :return:     dL / dx
        """
        pass

    def __call__(self, x: np.array, grad=True) -> np.array:
        """
            call forward()
        """
        return self.forward(x, grad=grad)

    def train(self):
        """
            turn to train-mode
        """
        self.is_train = True

    def eval(self):
        """
            turn to eval mode
        """
        self.is_train = False

    def __str__(self) -> str:
        """
            visualize this module
        """
        return "Basic Module with 0 parameters"

    def get_param_number(self) -> (int, int):
        """
            total and trainable parameter number
        """
        num = 0
        for para in self.param:
            num += np.size(para)
        return num, num


class Dropout(Module):
    def __init__(self, dropout):
        """
        :param dropout: p_dropout
        """
        super(Dropout, self).__init__()
        self.dropout = dropout

    def forward(self, x, grad=True):
        if self.is_train:
            # units to dropout
            to_drop = (np.random.random(x.shape) < self.dropout)

            # dropout some units
            y = (1 - to_drop) * x

            if grad:
                self.cache = to_drop
        else:
            # don't drop
            y = x

        return y

    def backward(self, dy):
        # dropped units' gradient is 0
        dx = (1 - self.cache) * dy
        return dx

    def __str__(self):
        return "Dropout(p={})".format(self.dropout)


class Swish(Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x, grad=True):

        x = np.clip(x, -1e2, 1e2)

        # e^-x
        e_x = np.exp(-x)

        # x * sigmoid(x)
        y = x / (1 + e_x)

        if grad:
            self.cache = [x, e_x]

        return y

    def backward(self, dy):

        # x, e^-x
        x, e_x = self.cache
        dx = dy * (1 + e_x + x * e_x) / (1 + e_x) ** 2
        return dx

    def __str__(self):
        return "Swish()"


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x, grad=True):
        # e^x
        ex = np.exp(x)

        # e^-x
        e_x = np.exp(-x)
        y = (ex - e_x) / (ex + e_x)

        if grad:
            self.cache = [(ex - e_x), (ex + e_x)]

        return y

    def backward(self, dy):

        # (e^x - e^-x), (e^x + e^-x)
        a, b = self.cache
        dydx = 1 - (a / b) ** 2
        dx = dy * dydx
        return dx

    def __str__(self):
        return "Tanh()"


class Linear(Module):
    def __init__(self, in_size, out_size, bias=True):
        """
            y = Xw + b
        :param in_size:   n_features
        :param out_size:  output-dim
        :param bias:      whether bias exists
        """
        super(Linear, self).__init__()

        # w0 ~ U(0, 0.01), b0 = 0
        self.param = [
            (np.random.random([in_size, out_size]) - 0.5) / in_size,
            np.zeros([1, out_size]) if bias else None
        ]
        self.grad = [None, None]
        self.cache = None

    def forward(self, x, grad=True):
        w, b = self.param
        y = x @ w
        if b is not None:
            y = y + b

        if grad:
            self.cache = x
        return y

    def backward(self, dy):

        w, b = self.param

        # cache is x
        dw = self.cache.T @ dy
        db = np.sum(dy, axis=0, keepdims=True)
        dx = dy @ w.T

        self.grad = [dw, db]
        return dx

    def __str__(self):
        return "Linear(in_size={}, out_size={}, bias={})".format(
            self.param[0].shape[0], self.param[0].shape[1], not (self.param[0] is None)
        )


class LeakyRelu(Module):
    def __init__(self, beta=1e-3):
        """
        :param beta: gradient for negative units
        """
        super(LeakyRelu, self).__init__()
        self.beta = beta

    def forward(self, x, grad=True):
        # max{bx, x}
        y = (x >= 0) * x + (x < 0) * self.beta * x
        if grad:
            self.cache = (x < 0)
        return y

    def backward(self, dy):
        dx = dy * self.cache * self.beta + dy * (1 - self.cache)
        return dx

    def __str__(self):
        return "LearyRelu(beta={})".format(self.beta)


class Relu(LeakyRelu):
    def __init__(self):
        # equals to LeakyRelu with beta=0
        super(Relu, self).__init__(0.)

    def __str__(self):
        return "Relu()"


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, grad=True):

        # to avoid overflow
        x = np.clip(x, -1e2, 1e2)

        # e^-x
        e_z = np.exp(-x)
        y = 1 / (1 + e_z)
        if grad:
            self.cache = e_z
        return y

    def backward(self, dy):
        dx = dy * self.cache / (1 + self.cache) ** 2
        return dx

    def __str__(self):
        return "Sigmoid()"


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x, grad=True) -> np.array:
        # e^x
        ex = np.exp(x)
        norm = np.sum(ex, axis=1, keepdims=True)
        y = ex / (norm + 1e-30)
        if grad:
            self.cache = y
        return y

    def backward(self, dy):
        dx = np.zeros_like(self.cache)   # n, h
        for i in range(dx.shape[0]):
            y = self.cache[i]
            dydx = np.diag(y) - y.reshape([-1, 1]) @ y.reshape([1, -1])
            dx[i] = dy[i].reshape([1, -1]) @ dydx
        return dx

    def __call__(self, x, grad=True) -> np.array:
        return self.forward(x, grad)

    def __str__(self):
        return "Softmax()"


class LossFunc:
    """
        base class for all loss function
    """
    def __init__(self):
        self.cache = None

    def forward(self, y_hat: np.array, y: np.array, grad=True) -> np.array:
        """
            calculate loss and restore cache
        :param y_hat:   output from model
        :param y:       label
        :param grad:    whether gradient needed
        :return:        loss
        """
        pass

    def backward(self) -> np.array:
        """
            calculate dL / d_y_hat
        :return: dL / d_y_hat
        """
        pass

    def __call__(self, y_hat: np.array, y: np.array, grad=True) -> np.array:
        """
            call forward()
        """
        return self.forward(y_hat, y, grad)


class CrossEntropy(LossFunc):
    """
        involve softmax
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        # softmax
        self.softmax = Softmax()

    def forward(self, y_hat: np.array, y: np.array, grad=True):
        # probs
        y_hat = self.softmax(y_hat)

        # y is class, turn to one-hot
        if len(list(y.shape)) == 1:
            y_ = np.zeros_like(y_hat)
            for i, j in enumerate(y):
                y_[i, j] = 1
            y = y_

        # -E[log(q)]
        loss = - np.mean(y * np.log(y_hat + 1e-30)) * y.shape[1]

        if grad:
            self.cache = [y, y_hat]

        return loss

    def backward(self):
        # label, pred-prob
        y, y_hat = self.cache

        return (y_hat - y) / y_hat.shape[0]


class MSE(LossFunc):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_hat, y, grad=True):
        loss = np.mean((y_hat - y) ** 2) * y.shape[-1] / 2
        if grad:
            self.cache = [y, y_hat]
        return loss

    def backward(self):
        y, y_hat = self.cache
        return (y_hat - y) / y.shape[0]


class L1Loss(LossFunc):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y_hat, y, grad=True):
        loss = np.mean(np.abs(y_hat - y)) * y.shape[-1]
        if grad:
            self.cache = [y, y_hat]
        return loss

    def backward(self):
        y, y_hat = self.cache
        return np.sign(y_hat - y) / y.shape[0]


class BatchNorm(Module):
    def __init__(self, in_size):
        super(BatchNorm, self).__init__()
        # gamma, beta
        self.param = [np.ones([1, in_size]), np.zeros([1, in_size])]
        self.grad = [None, None]
        self.mean = None
        self.std = None
        self.eps = 1e-20

    def forward(self, x, grad=True):
        if self.is_train or self.mean is None:
            self.mean = np.mean(x, axis=0, keepdims=True)

        if self.is_train or self.std is None:
            v = np.mean((x - self.mean) ** 2, axis=0, keepdims=True)
            self.std = (v + self.eps) ** 0.5
        y = (x - self.mean) / self.std
        gamma, beta = self.param
        z = gamma * y + beta
        if grad:
            self.cache = y
        return z

    def backward(self, dy):
        y = self.cache

        gamma, beta = self.param

        dz = dy
        dy = dy * gamma

        dx = (dy - np.mean(dy, axis=0, keepdims=True) -
              (1 - 1 / y.shape[0]) * y * np.mean(dy * y, axis=0, keepdims=True)) / self.std

        d_beta = np.sum(dz, axis=0, keepdims=True)
        d_gamma = np.sum(dz * y, axis=0, keepdims=True)

        self.grad = [d_gamma, d_beta]

        return dx

    def __str__(self):
        return "BatchNorm(in_size={})".format(self.param[0].shape[1])


class Identify(Module):
    """
        y = x
    """
    def __init__(self):
        super(Identify, self).__init__()

    def forward(self, x: np.array, grad=True) -> np.array:
        return x

    def backward(self, dy: np.array) -> np.array:
        return dy

    def __str__(self):
        return "Identify()"


class Sequence(Module):
    """
        a sequence of modules
    """
    def __init__(self, *args):
        """
        :param args: modules
        """
        super(Sequence, self).__init__()
        self.layers = list(args)
        self.grad = True

    def forward(self, x, grad=True):
        """
        :param x:     x
        :param grad:
        :return:      y
        """
        for layer in self.layers:
            x = layer.forward(x, grad and self.is_train)

        return x

    def backward(self, dy):
        """
        :param dy:   dL / dy
        :return:     dL / dx
        """
        # BP
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def eval(self):
        self.is_train = False
        for layer in reversed(self.layers):
            layer.eval()

    def train(self):
        self.is_train = True
        for layer in reversed(self.layers):
            layer.train()

    def save(self, path: str):
        import pickle
        para = [layer.param for layer in self.layers]
        with open(path, 'wb') as f:
            pickle.dump(para, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            para = pickle.load(f)
        for idx, layer in enumerate(self.layers):
            layer.param = para[idx]

    def __str__(self):
        string = "Sequence(\n"
        for layer in self.layers:
            string += "\t" + str(layer) + ",\n"
        return string + ")"


class MLP(Sequence):
    def __init__(
            self, in_size: int, out_size: int, h_sizes: list, dropout: float, act=Swish, norm=True, end_norm=True,
            tail_act: Module = None
    ):
        """
            MLP
        :param in_size:   n_features
        :param out_size:  class number
        :param h_sizes:   hidden sizes
        :param dropout:   dropout
        :param act:       activation function
        :param norm:      whether add BN before each layer
        """
        now_size = in_size
        lst = []
        for h_size in h_sizes:
            if norm:
                lst.append(BatchNorm(now_size))
            lst.append(Dropout(dropout))
            lst.append(Linear(now_size, h_size))
            lst.append(act())
            now_size = h_size
        if end_norm:
            lst.append(BatchNorm(now_size))
        lst.append(Linear(now_size, out_size))
        if tail_act is not None:
            lst.append(tail_act)

        super(MLP, self).__init__(*lst)

    def __str__(self):

        num = [0, 0]
        for layer in self.layers:
            a, b = layer.get_param_number()
            num[0] += a
            num[1] += b

        string = "MLP(\n"
        for layer in self.layers:
            string += "\t" + str(layer) + ",\n"
        string += "\n\ttotal param: {}\n\ttrainable param: {}\n".format(*num)
        return string + ")"


class ResidualBlock(Module):
    """
        shortcut is Identy if in_size = out_size
        else, Linear
    """
    def __init__(self, in_size: int, out_size: int, h_sizes: list, dropout: float, act=Swish, norm=True):
        super(ResidualBlock, self).__init__()
        now_size = in_size
        lst = []
        for h_size in h_sizes:
            if norm:
                lst.append(BatchNorm(now_size))
            lst.append(Dropout(dropout))
            lst.append(Linear(now_size, h_size))
            lst.append(act())
            now_size = h_size
        if norm:
            lst.append(BatchNorm(now_size))
        lst.append(Linear(now_size, out_size))

        shortcut = Identify() if out_size == in_size else Linear(in_size, out_size)
        self.layers = lst + [shortcut]

    def forward(self, x: np.array, grad=True) -> np.array:
        """
            y = f(x) + s(x)
        """
        left = x
        for layer in self.layers[:-1]:
            left = layer.forward(left, grad and self.is_train)
        short = self.layers[-1].forward(x, grad and self.is_train)
        return left + short

    def backward(self, dy: np.array) -> np.array:
        """
            dL/dx = dL/df * df/dx + dL/ds * ds/dx
        """
        d_short = self.layers[-1].backward(dy)
        for layer in reversed(self.layers[:-1]):
            dy = layer.backward(dy)
        return dy + d_short


class GradientDescent:
    """
        base modules for all optimizer
    """
    def __init__(self, layers: list, lr: float, l2=0.):
        """
            w -= lr * dw
        :param layers:    list of modules
        :param lr:        learning rate
        :param l2:        L2 regulation
        """
        self.lr = lr
        self.layers = []
        for layer in layers:
            self.add_layer(layer)
        self.l2 = l2

    def add_layer(self, module):
        """
            GradientDescent.layers should be a list of basic units, which have no member named layers
        """
        if hasattr(module, 'layers'):
            for layer in module.layers:
                self.add_layer(layer)
        else:
            self.layers.append(module)

    def zero_grad(self):
        """
            make all gradient zero
        """
        for layer in self.layers:
            for i in range(len(layer.grad)):
                layer.grad[i] = None

    def step(self):
        """
            update parameters
        """
        for layer in self.layers:
            for i in range(len(layer.grad)):
                layer.param[i] -= (self.lr * layer.grad[i] + self.l2 * layer.param[i])


class Adam(GradientDescent):
    def __init__(self, layers, lr, l2=0., beta1=0.9, beta2=0.999, eps=1e-30):
        super(Adam, self).__init__(layers, lr, l2)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1t = 1.
        self.beta2t = 1.
        self.eps = eps
        for layer in self.layers:
            for i in range(len(layer.grad)):
                layer.m.append(0)
                layer.v.append(0)

    def step(self):
        for layer in self.layers:
            for i in range(len(layer.grad)):
                g = (layer.grad[i] + self.l2 * layer.param[i])
                layer.m[i] = self.beta1 * layer.m[i] + (1-self.beta1) * g
                layer.v[i] = self.beta2 * layer.v[i] + (1-self.beta2) * g ** 2
                self.beta1t *= (1 - self.beta1)
                self.beta2t *= (1 - self.beta2)
                layer.m[i] /= (1-self.beta1t)
                layer.v[i] /= (1-self.beta2t)
                layer.param[i] -= self.lr * (layer.m[i] / ((layer.v[i] + self.eps) ** 0.5 + self.eps))


class Lion(GradientDescent):
    def __init__(self, layers, lr, l2=0., beta1=0.9, beta2=0.99, eps=1e-30):
        super(Lion, self).__init__(layers, lr, l2)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        for layer in self.layers:
            for i in range(len(layer.grad)):
                layer.m.append(0)
                layer.v.append(0)

    def step(self):
        for layer in self.layers:
            for i in range(len(layer.grad)):
                g = (self.lr * layer.grad[i])
                layer.v[i] = self.beta1 * layer.m[i] + (1-self.beta1) * g
                layer.m[i] = self.beta2 * layer.m[i] + (1-self.beta2) * g
                layer.param[i] -= self.lr * (np.sign(layer.v[i]) + self.l2 * layer.param[i])


class AdaGrad(GradientDescent):
    def __init__(self, layers, lr, l2=0., eps=1e-10):
        super(AdaGrad, self).__init__(layers, lr, l2)
        self.eps = eps
        for layer in self.layers:
            for i in range(len(layer.grad)):
                layer.m.append(0)

    def step(self):
        for layer in self.layers:
            for i in range(len(layer.grad)):
                g = (self.lr * layer.grad[i] + self.l2 * layer.param[i])
                layer.m[i] += g ** 2
                layer.param[i] -= self.lr / (self.eps + np.sqrt(layer.m[i])) * g


class RMSProp(GradientDescent):
    def __init__(self, layers, lr, l2=0., rho=0.9, eps=1e-10):
        super(RMSProp, self).__init__(layers, lr, l2)
        self.eps = eps
        self.rho = rho
        for layer in self.layers:
            for i in range(len(layer.grad)):
                layer.m.append(0)

    def step(self):
        for layer in self.layers:
            for i in range(len(layer.grad)):
                g = (self.lr * layer.grad[i] + self.l2 * layer.param[i])
                layer.m[i] = self.rho * layer.m[i] + (1 - self.rho) * g ** 2
                layer.param[i] -= self.lr / (self.eps + np.sqrt(layer.m[i])) * g
