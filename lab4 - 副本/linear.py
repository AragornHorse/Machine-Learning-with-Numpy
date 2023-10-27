import numpy as np
import random


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


def gaussian(x, mean, cov):
    n, d = x.shape
    dis = x - mean.reshape([1, -1])

    dis = dis.reshape([n, 1, d]) @ np.linalg.inv(cov) @ dis.reshape([n, d, 1])
    det = np.linalg.det(cov)
    norm = 1 / ((2 * np.pi) ** (d / 2) * det ** 0.5)
    return norm * np.exp(-0.5 * dis.reshape([-1]))


class LDA:
    def __init__(self):
        self.cov = []
        self.means = []
        self.py = []
        self.k = 0

    def fit(self, x, y):
        n, h = x.shape
        self.k = np.max(y) + 1
        self.means = np.zeros([self.k, h])
        self.py = np.zeros([self.k])

        self.cov = (x.T @ x) / x.shape[0]

        for k in range(self.k):
            x_ = x[np.nonzero(y==k)[0], :]
            # print("x_")
            self.means[k] = np.mean(x_, axis=0)
            # print(self.covs[k])
            # print("cov")
            self.py[k] = x_.shape[0]
            # print(k)
        self.py = self.py / np.sum(self.py)

    def __call__(self, x):
        n, h = x.shape
        k = self.k

        p_cls_x = np.zeros([n, k])

        for i in range(k):
            p_cls_x[:, i] = gaussian(x, self.means[i], self.cov) * self.py[i]

        cls = np.argmax(p_cls_x, axis=-1)

        return cls


def linear_kernel(x1, x2):
    return x1 @ x2.T


def gaussian_kernel(x1, x2, sigma=5e8):
    n1, h = x1.shape
    n2, h = x2.shape
    x1 = x1.reshape([n1, h, 1])
    x2 = x2.T.reshape([1, h, n2])
    dis = np.sum((x1 - x2) ** 2, axis=1).reshape([n1, n2])
    y = np.exp(- dis / sigma)
    # print(y)
    return y


class SVM:
    def __init__(self):
        self.alpha = None
        self.w = None
        self.b = None
        self.kernel = None
        self.x = None
        self.y = None

    def SMO(self, x, y, C=1e3, kernel=gaussian_kernel, max_iter=1000):

        # print(y)

        if np.min(y) == 0:
            y = y.astype(int) * 2 - 1

        # print(y)

        y = y.reshape([-1, 1])
        self.y = y

        self.kernel = kernel
        self.x = x

        n, h = x.shape

        K = kernel(x, x)  # n, n
        # print(K.shape)

        self.alpha = np.zeros([n, 1])
        self.w = x.T @ (self.alpha * y)

        self.b = 0

        for epoch in range(max_iter):
            i = 0
            for _ in range(self.alpha.shape[0]):
                if self.alpha[_] != 0 and y[_, 0] * self.__call__(None, K[:, _].reshape([-1, 1])) != 1:
                    i = _
                    break
                elif self.alpha[_] == 0 and y[_, 0] * self.__call__(None, K[:, _].reshape([-1, 1])) < 1:
                    i = _
                    break
                elif self.alpha[_] == C and y[_, 0] * self.__call__(None, K[:, _].reshape([-1, 1])) > 1:
                    i = _
                    break

            idxs = random.sample(list(range(x.shape[0])), k=10)

            if i in idxs:
                idxs.remove(i)

            f1 = self.__call__(x[i].reshape([1, -1]), k=K[i].reshape([-1, 1]))
            f2 = self.__call__(x[idxs], k=K[:, idxs])
            E1 = f1 - y[i]
            E2s = f2 - y[idxs, :]
            if E1 >= 0:
                idx = np.argmin(E2s.reshape([-1]))
                j = idxs[idx]
            else:
                idx = np.argmax(E2s.reshape([-1]))
                j = idxs[idx]
            f2 = f2[idx]
            E2 = E2s[idx]

            aj_o = self.alpha[j, 0]
            ai_o = self.alpha[i, 0]

            if y[i, 0] * y[j, 0] == -1:
                L = max(0, -(ai_o - aj_o))
                H = min(C, C - (ai_o - aj_o))
                # print(L, H)
                # print(C, C + ai_o - aj_o)
            else:
                L = max(0, ai_o + aj_o - C)
                H = min(C, ai_o + aj_o)
                # print(L, H)
                # print(C, ai_o + aj_o)

            aj = np.clip(aj_o + (E1 - E2) / (K[i, i] + K[j, j] - 2 * K[i, j]), L, H)
            ai = ai_o + y[i, 0] * y[j, 0] * (aj_o - aj)

            # print(ai, aj)
            # print('')

            self.alpha[i, 0] = ai
            self.alpha[j, 0] = aj

            self.w = self.w + (ai - ai_o) * y[i, 0] * x[i].reshape([-1, 1]) + (aj - aj_o) * y[j, 0] * x[j].reshape([-1, 1])

            b1 = y[i, 0] - self.__call__(None, K[i].reshape([-1, 1]))
            b2 = y[j, 0] - self.__call__(None, K[j].reshape([-1, 1]))
            if 0 <= ai <= C:
                self.b = b1
            elif 0 <= aj <= C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

        self.b = np.mean(y - self.__call__(None, K))

    def __call__(self, x, k=None):
        if k is None:
            k = self.kernel(x, self.x).T   # n, n1

        return k.T @ (self.alpha * self.y) + self.b


if __name__ == '__main__':

    from data import *
    # pca = PCA()
    # pca.get_w(train_x, k=100)
    #
    # train_x = pca(train_x, k=100, update_w=False)
    # test_x = pca(test_x, k=100, update_w=False)
    #
    # print(train_x.shape, test_x.shape)

    # cov = (train_x.T @ train_x) / train_x.shape[0]

    # U, D, _ = np.linalg.svd(cov)
    #
    # D = np.diag(D ** -0.5)
    #
    # train_x = (D @ U.T @ train_x.T).T
    # test_x = (D @ U.T @ test_x.T).T

    # model = LDA()
    # model.fit(train_x, train_y)
    #
    # y_hat = model(test_x)
    #
    # acc = np.mean((y_hat == test_y).astype(int))
    #
    # print(acc)
    #
    # y_hat = model(train_x)
    #
    # acc = np.mean((y_hat == train_y).astype(int))
    #
    # print(acc)

    model = SVM()

    x = train_x[:10000]
    y = train_y[:10000]

    idxs = np.nonzero((y == 0) + (y == 1))[0]

    x = x[idxs]
    y = y[idxs]

    model.SMO(x, y, kernel=linear_kernel, max_iter=1000)

    y_hat = model(x)
    y_hat = (y_hat > 0).reshape([-1]).astype(int)

    print(np.mean(y==y_hat))

    x = test_x[:1000]
    print(x.shape)
    y = test_y[:1000]

    idxs = np.nonzero((y == 0) + (y == 1))[0]

    x = x[idxs]
    y = y[idxs]

    # print(x)
    print(y)



    y_hat = model(x)
    y_hat = (y_hat > 0).reshape([-1]).astype(int)

    print(y_hat)
    print(np.mean(y==y_hat))







