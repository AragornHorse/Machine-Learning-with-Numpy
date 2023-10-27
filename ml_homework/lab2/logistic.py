import numpy as np


class PCA:
    def __init__(self):
        self.w = None
        self.lbd = None

    def get_w(self, x, k=4):
        x = np.copy(x)  # b, h
        x = x - np.mean(x, axis=0).reshape(1, -1)
        cov = np.cov(x, rowvar=False)   # h, h
        lbd, vec = np.linalg.eig(cov)
        idx = np.argsort(lbd)[-k:]
        w = vec[:, idx]  # h, k
        self.w = w
        self.lbd = np.sort(lbd)
        return w

    def __call__(self, x, k=5, update_w=True):
        if self.w is None or self.w.shape[1] != k:
            update_w = True
        if update_w:
            self.get_w(x, k)
        return np.dot(x, self.w)


def cross_entropy(y_hat, y):
    loss = - np.mean(y * np.log(y_hat + 1e-30)) * y.shape[1]
    return loss


class Logistic:
    def __init__(self, in_size, out_size, bias=True):
        self.bias = bias
        self.in_size = in_size
        self.out_size = out_size
        if bias:
            self.in_size += 1
        self.w = np.random.random([self.in_size, out_size]) / 1e3  # h, c
        self.g = None
        self.d = None

    def get_x(self, x, y=None):
        x_ = np.ones([x.shape[0], self.in_size])
        s = x.shape[1]
        x_[:, :s] = x
        x = x_
        return x, y

    def grad_train(self, x, y, lr=1e-5, weight_decay=0.):

        x, y = self.get_x(x, y)

        n = x.shape[0]

        z = x @ self.w   # n, c

        ex = np.exp(z)

        norm = np.sum(ex, axis=1, keepdims=True)
        y_hat = ex / (norm + 1e-30)

        y_onehot = np.zeros_like(y_hat)
        for i, j in enumerate(y):
            y_onehot[i, j] = 1

        loss = cross_entropy(y_hat, y_onehot)

        acc = np.mean((np.argmax(y_hat, axis=-1) == y))

        dz = (y_hat - y_onehot) / n   # n, c

        dw = x.T @ dz  # h, c

        self.w -= (dw + self.w * weight_decay) * lr

        return loss, acc

    def __call__(self, x, manage=True):
        if manage:
            x, _ = self.get_x(x, None)
        z = x @ self.w
        ex = np.exp(z)
        norm = np.sum(ex, axis=1, keepdims=True)
        y_hat = ex / (norm + 1.)
        return y_hat

    def eval(self, x, y):
        x, y = self.get_x(x, y)
        y_hat = self.__call__(x, manage=False)
        return cross_entropy(y_hat, y), np.mean((np.argmax(y_hat, axis=-1) == y))

    def train_newton(self, x, y, weight_decay=0., lr=0.1):

        x, y = self.get_x(x, y)
        n = x.shape[0]
        z = x @ self.w  # n, c

        ex = np.exp(z)

        norm = np.sum(ex, axis=1, keepdims=True)
        y_hat = ex / (norm + 1e-30)
        y_onehot = np.zeros_like(y_hat)
        for i, j in enumerate(y):
            y_onehot[i, j] = 1

        loss = cross_entropy(y_hat, y_onehot)
        acc = np.mean((np.argmax(y_hat, axis=-1) == y))

        dz = (y_hat - y_onehot) / n  # n, c
        dw = (x.T @ dz + weight_decay * self.w)  # h, c

        hessian = np.zeros([self.in_size, self.out_size, self.in_size, self.out_size])

        for alpha in range(hessian.shape[0]):
            for beta in range(hessian.shape[1]):
                for gamma in range(hessian.shape[2]):
                    for miu in range(hessian.shape[3]):
                        hessian[alpha, beta, gamma, miu] = -np.mean(x[:, alpha] * x[:, gamma] * y_hat[:, beta] * y_hat[:, miu])
                        if beta == miu:
                            hessian[alpha, beta, gamma, miu] += np.mean(x[:, alpha] * x[:, gamma] * y_hat[:, beta])

        hessian = hessian.reshape([self.in_size * self.out_size, -1])
        hessian += max(weight_decay, 0.5) * np.eye(self.in_size * self.out_size)

        hessian_inv = np.linalg.inv(hessian)
        dw = dw.reshape([-1, 1])
        ddw = hessian_inv @ dw

        w = np.copy(self.w)

        self.w = w - lr * ddw.reshape([self.in_size, self.out_size])

        # print(np.sum(np.isnan(self.w)))

        return loss, acc

    def conjugate_gradient(self, x, y, weight_decay=0., lr=1e-6):
        x, y = self.get_x(x, y)

        n = x.shape[0]

        z = x @ self.w  # n, c

        ex = np.exp(z)

        norm = np.sum(ex, axis=1, keepdims=True)
        y_hat = ex / (norm + 1e-30)

        y_onehot = np.zeros_like(y_hat)
        for i, j in enumerate(y):
            y_onehot[i, j] = 1

        loss = cross_entropy(y_hat, y_onehot)

        acc = np.mean((np.argmax(y_hat, axis=-1) == y))

        dz = (y_hat - y_onehot) / n  # n, c

        dw = x.T @ dz + weight_decay * self.w  # h, c

        g_k = dw.reshape([-1, 1])

        if self.g is not None:

            g_k_ = self.g

            beta = (g_k.T @ (g_k - g_k_)) / (g_k_.T @ g_k_)

            self.d = g_k_ - beta * self.d

            self.w -= lr * self.d.reshape([self.in_size, self.out_size])
        else:
            self.d = g_k

        self.g = g_k

        return loss, acc


if __name__ == '__main__':
    import data

    pca = PCA()

    loader = data.DataLoader(pca(data.train_x, k=20), data.train_y, 128)

    print("loader")

    model = Logistic(20, 10)
    i = 0

    for data in loader:
        x, y = data
        loss, acc = model.conjugate_gradient(x, y, weight_decay=0.)
        print("loss:{}, acc:{}".format(loss, acc))
        i += 1
        if i == 600:
            break

